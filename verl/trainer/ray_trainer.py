# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from collections import defaultdict, Counter
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum, IntEnum, auto
from typing import Any, Callable, Dict, List, Optional, Tuple, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from ray.experimental.tqdm_ray import tqdm
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import PreTrainedTokenizer, ProcessorMixin

from ..protocol import DataProto, pad_dataproto_to_divisor, unpad_dataproto
from ..single_controller.base import Worker
from ..single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from ..single_controller.ray.base import create_colocated_worker_cls
from ..utils import torch_functional as VF
from ..utils.checkpoint import CHECKPOINT_TRACKER, remove_obsolete_ckpt
from ..utils.dataset import RLHFDataset, collate_fn
from ..utils.logger import Tracker
from ..utils.py_functional import convert_dict_to_str
from ..utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from ..workers.fsdp_workers import FSDPWorker
from . import core_algos
from .config import PPOConfig
from .metrics import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics
from .buffer import Buffer

class Role(IntEnum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """

    Actor = auto()
    Rollout = auto()
    ActorRollout = auto()
    Critic = auto()
    RefPolicy = auto()
    RewardModel = auto()
    Teacher = auto()
    ActorRolloutRef = auto()


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """

    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REMAX = "remax"
    RLOO = "rloo"


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker."""
        return self.resource_pool_dict[self.mapping[role]]

    def get_num_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        gpus_available = ray.available_resources().get("GPU", 0)
        gpus_required = self.get_num_gpus()
        if gpus_available < gpus_required:
            raise ValueError(f"Total available GPUs {gpus_available} is less than total desired GPUs {gpus_required}.")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.KLController, kl_penalty="kl"):
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    response_mask = data.batch["response_mask"]

    # compute kl between ref_policy and current policy
    kld = core_algos.compute_kl(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)
    kld = kld * response_mask  # (batch_size, response_length)

    data.batch["token_level_rewards"] = token_level_scores - kl_ctrl.kl_coef * kld

    current_kl = VF.masked_mean(kld, mask=response_mask, dim=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    metrics = {"critic/kl": current_kl, "critic/kl_coef": kl_ctrl.kl_coef}

    # According to https://github.com/huggingface/trl/blob/v0.11.0/trl/trainer/ppo_trainer.py#L880
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    return data, metrics


def compute_advantage(data: DataProto, adv_estimator: AdvantageEstimator, gamma: float = 1.0, lam: float = 1.0):
    token_level_rewards = data.batch["token_level_rewards"]
    response_mask = data.batch["response_mask"]
    index = data.non_tensor_batch["uid"]
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch["values"]
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards, values, response_mask, gamma, lam
        )
    elif adv_estimator == AdvantageEstimator.GRPO:
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards, response_mask, index)
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards, response_mask, gamma
        )
    elif adv_estimator == AdvantageEstimator.REMAX:
        reward_baselines = data.batch["reward_baselines"]
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards, reward_baselines, response_mask
        )
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards, response_mask, index)
    else:
        raise NotImplementedError

    data.batch["advantages"] = advantages
    data.batch["returns"] = returns
    return data

def _select_by_advantage(batch: DataProto, threshold = 0, absolute = True):
    advantages = batch.batch["advantages"][:,0]
    if absolute:
        selected_mask = advantages.abs() > threshold
    else:
        selected_mask = advantages > threshold
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    print("select samples by advantage", threshold, ":", len(selected_indices))
    if len(selected_indices) > 0:
        return batch.select_by_index(index_list=selected_indices.tolist())
    else:
        return None

def _select_by_advantage_reverse(batch: DataProto, threshold = 0, absolute = True):
    advantages = batch.batch["advantages"][:,0]
    if absolute:
        selected_mask = advantages.abs() < threshold
    else:
        selected_mask = advantages < threshold
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    print("select samples by advantage", threshold, ":", len(selected_indices))
    if len(selected_indices) > 0:
        return batch.select_by_index(index_list=selected_indices.tolist())
    else:
        return None
    

def _select_positive(batch: DataProto, adv_threshold: float = 0.0, reward_threshold: float = 0.0):
    advantages = batch.batch["advantages"][:, 0]   # [B]
    values = batch.batch["token_level_scores"].sum(dim=1)
    bool_reward = (values > reward_threshold)
    batch.batch["bool_reward"] = bool_reward.to(torch.bool)
    selected_mask = (advantages > adv_threshold) & bool_reward
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1) 
    print(f"select samples by (adv>{adv_threshold}) & (reward>{reward_threshold}) : {selected_indices.numel()}")

    if selected_indices.numel() > 0:
        return batch.select_by_index(index_list=selected_indices.tolist())
    else:
        return None



    
def _select_negative(batch: DataProto, threshold = 0):
    advantages = batch.batch["advantages"][:,0]

    selected_mask = advantages < threshold
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    print("select negative samples:", len(selected_indices))
    if len(selected_indices) > 0:
        return batch.select_by_index(index_list=selected_indices.tolist())
    else:
        return None    
    
def _compute_bool_reward_by_advantage(batch: DataProto, threshold = 0, absolute = True):
    # 原advantage如果大于threshold，则标记为True，否则为False。直接重建一个和advantage同shape的bool tensor放到batch中
    advantages = batch.batch["advantages"][:,0]
    if absolute:
        selected_mask = advantages.abs() > threshold
    else:
        selected_mask = advantages > threshold
    bool_tensor = selected_mask.to(torch.bool)
    batch.batch["bool_reward"] = bool_tensor
    return batch

def _compute_bool_reward_by_score(batch: DataProto, threshold = 0, absolute = True):
    # 原advantage如果大于threshold，则标记为True，否则为False。直接重建一个和advantage同shape的bool tensor放到batch中
    values = batch.batch["token_level_scores"].sum(dim = 1)
    if absolute:
        selected_mask = values.abs() > threshold
    else:
        selected_mask = values > threshold
    bool_tensor = selected_mask.to(torch.bool)
    batch.batch["bool_reward"] = bool_tensor
    return batch

def _select_easy_prompts(batch: DataProto, rollout_n: int, eps: float = 1e-6):
    advantages = batch.batch["advantages"][:, 0]
    values = batch.batch["token_level_scores"].sum(dim = 1)
    selected_mask = (advantages.abs() < eps) & (values > eps)
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    #n1 = len(selected_indices)
    #selected_indices = [i.item() // rollout_n for i in selected_indices if i.item() % rollout_n == 0]
    #n2 = len(selected_indices)
    #print(f"select easy prompts:", len(selected_indices))
    #assert n1 == n2 * rollout_n
    return selected_indices

def set_easy_advantages(batch: DataProto, easy_indices: List[int], adv: int):
    advantages = batch.batch["advantages"]  # shape: [B, T]
    for idx in easy_indices:
        advantages[idx] = adv

def _select_hard_prompts(batch: DataProto, rollout_n: int, difficult_record, eps: float = 1e-6):
    advantages = batch.batch["advantages"][:, 0]
    values = batch.batch["token_level_scores"].sum(dim = 1)    
    selected_mask = (advantages.abs() < eps) & (values < eps)
    selected_indices = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    n1 = len(selected_indices)
    selected_indices = [i.item() // rollout_n for i in selected_indices if i.item() % rollout_n == 0]
    n2 = len(selected_indices)
    difficult_record.append(len(selected_indices))
    print(f"select hard prompts:", len(selected_indices))
    assert n1 == n2 * rollout_n
    return selected_indices


def _filtering_overlong(batch: DataProto, overlong_record, overlong_positive_record):
    advantages = batch.batch["advantages"][:, 0]
    response_mask_last_col = batch.batch["response_mask"][:, -1]
    selected_mask = ~((response_mask_last_col == 1) & (advantages > 0))
    selected_mask = selected_mask.nonzero(as_tuple=False).squeeze(-1)
    overlong_positive_record.append(len(batch) - len(selected_mask))

    selected_overlong = (response_mask_last_col == 0)
    selected_overlong = selected_overlong.nonzero(as_tuple=False).squeeze(-1)
    overlong_record.append(len(batch) - len(selected_overlong))

    if len(selected_mask) > 0:
        return batch.select_by_index(index_list=selected_mask.tolist())
    else:
        return None  

def count_occurrences(batch: DataProto, name):
    occurrences = batch.batch[name][:, 0]  # shape: (B,)
    occurrences_list = occurrences.cpu().numpy().tolist()
    occurrences_counter = Counter(occurrences_list)
    for k,v in occurrences_counter.items():
        print(name, k, v)
    return occurrences_counter

def _align_batch_with_buffer(batch: DataProto, buf: Buffer, s):
    n1 = len(batch)
    if buf.size < len(batch) % s:
        batch = trim_to_multiple(batch, s)
    else:
        batch = DataProto.concat([batch, buf.sample((s - len(batch) % s) % s)])
    n2 = len(batch)
    print(f"align num of samples from {n1} to {n2}")
    return batch

def trim_to_multiple(batch: DataProto, s: int):
    total = len(batch)
    target_size = (total // s) * s
    if total == target_size:
        return batch

    start_idx = total - target_size
    return batch.select_by_index(list(range(start_idx, total)))

@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield

    timing_raw[name] = timer.last

def print_batch(batch: DataProto, s = ""):
    for k, v in batch.batch.items():
        print(s,k,v.size())

def duplicate_gen_batch(batch: DataProto, n: int):
    if len(batch) % n > 0:
        sample_num = (n - len(batch) % n) % n
        fill_batch = batch.select_by_index([i for i in range(sample_num)])
        return DataProto.concat([batch, fill_batch])
    else:
        return batch

    

class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    def __init__(
        self,
        config: PPOConfig,
        tokenizer: PreTrainedTokenizer,
        processor: Optional[ProcessorMixin],
        role_worker_mapping: dict[Role, Type[Worker]],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: Type[RayWorkerGroup] = RayWorkerGroup,
        reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
        val_reward_fn: Optional[Callable[[DataProto], Tuple[torch.Tensor, Dict[str, List[float]]]]] = None,
    ):
        self.tokenizer = tokenizer
        if config.trainer.project_name == 'llama':
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "{{ '<|start_header_id|>system<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'user' %}"
                "{{ '<|start_header_id|>user<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% elif message['role'] == 'assistant' %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' + message['content'] + '<|eot_id|>' }}"
                "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}"
                "{% endif %}"
            )
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.worker.hybrid_engine
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, (
                f"ActorRollout should be included in {role_worker_mapping.keys()}."
            )
        else:
            raise NotImplementedError

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reward_model = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if Role.RefPolicy in role_worker_mapping and not config.algorithm.disable_kl:
            self.use_reference_policy = True
            self.kl_ctrl = core_algos.get_kl_controller(config.algorithm)
        else:
            self.use_reference_policy = False
            self.kl_ctrl = core_algos.FixedKLController(init_kl_coef=0.0)
            print("KL is disabled, no KL metrics will be logged. Please set `kl_coef=0` to log KL metrics.")
        self.use_teacher_policy = (
            self.config.worker.teacher.use_teacher
            and Role.Teacher in role_worker_mapping
        )

        if config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        else:
            self.use_critic = False

        if config.algorithm.adv_estimator not in list(AdvantageEstimator):
            raise NotImplementedError(f"Unknown advantage estimator: {config.algorithm.adv_estimator}.")

        if config.data.rollout_batch_size % config.worker.actor.global_batch_size != 0:
            raise ValueError("Rollout batch size must be divisible by actor global batch size.")

        if (
            config.data.rollout_batch_size * config.worker.rollout.n
        ) % config.worker.actor.micro_batch_size_per_device_for_experience != 0:
            raise ValueError(
                "Rollout batch size * rollout.n must be divisible by actor micro batch size for experience."
            )

        if self.use_critic:
            if config.data.rollout_batch_size % config.worker.critic.global_batch_size != 0:
                raise ValueError("Rollout batch size must be divisible by critic global batch size.")

            if (
                config.data.rollout_batch_size * config.worker.rollout.n
            ) % config.worker.critic.micro_batch_size_per_device_for_experience != 0:
                raise ValueError(
                    "Rollout batch size * rollout.n must be divisible by critic micro batch size for experience."
                )

        if (
            config.algorithm.adv_estimator in (AdvantageEstimator.GRPO, AdvantageEstimator.RLOO)
            and config.worker.rollout.n == 1
        ):
            raise ValueError("GRPO and RLOO algorithm need `config.worker.rollout.n > 1`.")

        self._create_dataloader()

    def _create_dataloader(self) -> None:
        self.train_dataset = RLHFDataset(
            data_path=self.config.data.train_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            format_prompt=self.config.data.format_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.seed)
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.rollout_batch_size,
            sampler=sampler,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=True,
        )

        self.val_dataset = RLHFDataset(
            data_path=self.config.data.val_files,
            tokenizer=self.tokenizer,
            processor=self.processor,
            prompt_key=self.config.data.prompt_key,
            answer_key=self.config.data.answer_key,
            image_key=self.config.data.image_key,
            max_prompt_length=self.config.data.max_prompt_length,
            truncation="right",
            format_prompt=self.config.data.format_prompt,
            min_pixels=self.config.data.min_pixels,
            max_pixels=self.config.data.max_pixels,
        )
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=len(self.val_dataset)
            if self.config.data.val_batch_size == -1
            else self.config.data.val_batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=collate_fn,
            pin_memory=False,
            drop_last=False,
        )

        assert len(self.train_dataloader) >= 1
        assert len(self.val_dataloader) >= 1
        print(f"Size of train dataloader: {len(self.train_dataloader)}")
        print(f"Size of val dataloader: {len(self.val_dataloader)}")

        if self.config.trainer.max_steps is not None:
            training_steps = self.config.trainer.max_steps
        else:
            training_steps = len(self.train_dataloader) * self.config.trainer.total_episodes

        self.training_steps = training_steps
        self.config.worker.actor.optim.training_steps = training_steps
        self.config.worker.critic.optim.training_steps = training_steps
        print(f"Total training steps: {self.training_steps}")

    def _maybe_log_val_generations(
        self, inputs: List[str], outputs: List[str], labels: List[str], scores: List[float]
    ) -> None:
        """Log a table of validation samples"""
        if self.config.trainer.val_generations_to_log <= 0:
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, labels, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        samples = samples[: self.config.trainer.val_generations_to_log]
        self.logger.log_generation(samples, self.global_step)

    def _validate(self) -> Dict[str, Any]:
        reward_tensor_lst = []
        # Lists to collect samples for the table
        sample_inputs, sample_outputs, sample_labels, sample_scores = [], [], [], []
        reward_metrics_lst = defaultdict(list)
        for batch_dict in self.val_dataloader:
            test_batch = DataProto.from_single_dict(batch_dict)
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if "multi_modal_data" in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=["input_ids", "attention_mask", "position_ids"],
                    non_tensor_batch_keys=["raw_prompt_ids"],
                )

            test_gen_batch.meta_info = self.config.worker.rollout.val_override_config
            test_gen_batch, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch = self.actor_rollout_wg.generate_sequences(test_gen_batch)
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch, pad_size=pad_size)
            print("validation generation end")

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            sample_labels.extend(test_batch.non_tensor_batch["ground_truth"].tolist())
            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor, reward_metrics = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            for key, value in reward_metrics.items():
                reward_metrics_lst[key].extend(value)

        self._maybe_log_val_generations(sample_inputs, sample_outputs, sample_labels, sample_scores)
        reward_score = torch.cat(reward_tensor_lst, dim=0).sum(-1).mean().item()
        val_reward_metrics = {f"val/{key}_reward": value for key, value in reduce_metrics(reward_metrics_lst).items()}
        return {"val/reward_score": reward_score, **val_reward_metrics}

    def init_workers(self) -> None:
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout], config=self.config.worker, role="actor_rollout"
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.worker, role="critic"
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy], config=self.config.worker, role="ref"
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        if self.use_teacher_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Teacher)
            teacher_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.Teacher],
                config=self.config.worker,
                role="teacher",
            )
            self.resource_pool_to_cls[resource_pool]["teacher"] = teacher_policy_cls

        # create a reward model if reward_fn is None
        if self.use_reward_model:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.RewardModel], config=self.config.worker, role="reward"
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg: Dict[str, FSDPWorker] = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
            
        if self.use_teacher_policy:
            self.teacher_policy_wg = all_wg["teacher"]
            self.teacher_policy_wg.init_model()

        if self.use_reward_model:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self) -> None:
        # path: {save_checkpoint_path}/global_step_{global_step}/{actor,critic}
        remove_obsolete_ckpt(
            self.config.trainer.save_checkpoint_path, self.global_step, self.config.trainer.save_limit
        )
        folder_path = os.path.join(self.config.trainer.save_checkpoint_path, f"global_step_{self.global_step}")
        actor_path = os.path.join(folder_path, "actor")
        self.actor_rollout_wg.save_checkpoint(actor_path)

        if self.use_critic:
            critic_path = os.path.join(folder_path, "critic")
            self.critic_wg.save_checkpoint(critic_path)

        dataloader_path = os.path.join(folder_path, "dataloader.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_path)

        last_global_step_path = os.path.join(self.config.trainer.save_checkpoint_path, CHECKPOINT_TRACKER)
        with open(last_global_step_path, "w") as f:
            f.write(str(self.global_step))

    def _load_checkpoint(self) -> None:
        if self.config.trainer.load_checkpoint_path is None:
            return

        if "global_step_" not in self.config.trainer.load_checkpoint_path.strip(os.path.sep).split(os.path.sep)[-1]:
            raise ValueError("`load_checkpoint_path` should end with `global_step_*`.")

        print(f"Load from checkpoint: {self.config.trainer.load_checkpoint_path}.")
        self.global_step = int(self.config.trainer.load_checkpoint_path.strip(os.path.sep).split("global_step_")[-1])
        actor_path = os.path.join(self.config.trainer.load_checkpoint_path, "actor")
        self.actor_rollout_wg.load_checkpoint(actor_path)
        if self.use_critic:
            critic_path = os.path.join(self.config.trainer.load_checkpoint_path, "critic")
            self.critic_wg.load_checkpoint(critic_path)

        dataloader_path = os.path.join(self.config.trainer.load_checkpoint_path, "dataloader.pt")
        if os.path.exists(dataloader_path):
            dataloader_state_dict = torch.load(dataloader_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"No dataloader state found at {dataloader_path}, will start from scratch.")

    def _balance_batch(self, batch: DataProto, metrics: Dict[str, Any], logging_prefix: str = "global_seqlen") -> None:
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(
            global_seqlen_lst, k_partitions=world_size, equal_size=True
        )
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)

    def apply_eos_boost_and_early_stop(
        model_inputs,
        meta_info,
        response_length,
        pad_to_length=None,
    ):
        eos_token_id = meta_info["eos_token_id"]
        pad_token_id = meta_info.get("pad_token_id", eos_token_id)  # fallback to eos if missing

        logits = model_inputs["logits"]        # (B, L_resp, V)
        responses = model_inputs["responses"]  # (B, L_resp)

        assert logits.shape[:2] == responses.shape, "logits and responses must match in B and L_resp"
        B, L_resp, V = logits.shape

        # Step 1: boost eos_token_id logits
        logits[:, :, eos_token_id] *= 2.0

        # Step 2: find first early-stop position per sample
        eos_is_argmax = logits.argmax(dim=-1) == eos_token_id          # (B, L_resp)
        not_already_eos = responses != eos_token_id                     # (B, L_resp)
        candidate = eos_is_argmax & not_already_eos                     # (B, L_resp)

        positions = torch.arange(L_resp, device=responses.device).unsqueeze(0)  # (1, L_resp)
        first_eos_pos = torch.where(
            candidate, positions, L_resp
        ).min(dim=1).values  # (B,)

        # Build mask: from first_eos_pos onward → set to eos_token_id
        idx_range = torch.arange(L_resp, device=responses.device).unsqueeze(0)  # (1, L_resp)
        should_replace = idx_range >= first_eos_pos.unsqueeze(1)                # (B, L_resp)
        valid_mask = (first_eos_pos < L_resp).unsqueeze(1)                      # (B, 1)
        final_mask = should_replace & valid_mask                                # (B, L_resp)

        # ✅ Early-stop fill: use eos_token_id (semantic termination)
        responses.masked_fill_(final_mask, eos_token_id)

        # Optional: length alignment padding — use pad_token_id (not eos!)
        if pad_to_length is not None:
            if pad_to_length > L_resp:
                # Right-pad with pad_token_id (standard practice)
                padded = torch.full(
                    (B, pad_to_length),
                    pad_token_id,  # ← key change: use pad_token_id here
                    dtype=responses.dtype,
                    device=responses.device
                )
                padded[:, :L_resp] = responses
                model_inputs["responses"] = padded
            elif pad_to_length < L_resp:
                model_inputs["responses"] = responses[:, :pad_to_length]
            else:
                model_inputs["responses"] = responses
        else:
            model_inputs["responses"] = responses  # keep original

    def _update_critic(self, batch: DataProto, timing_raw, metrics):
        if self.use_critic:
            with _timer("update_critic", timing_raw):
                critic_output = self.critic_wg.update_critic(batch)

            critic_metrics = reduce_metrics(critic_output.non_tensor_batch)
            metrics.update(critic_metrics)

    def _update_actor(self, batch: DataProto, timing_raw, metrics):
        if self.config.trainer.critic_warmup <= self.global_step:
            with _timer("update_actor", timing_raw):
                actor_output = self.actor_rollout_wg.update_actor(batch)

            actor_metrics = reduce_metrics(actor_output.non_tensor_batch)
            metrics.update(actor_metrics)

    def _compute_reward(self, batch: DataProto, timing_raw, metrics):
        with _timer("reward", timing_raw):
            reward_tensor, reward_metrics, length_tensor, accuracy_tensor = self.reward_fn(batch)
            batch.batch["token_level_scores"] = reward_tensor
            batch.batch["acc_scores"] = accuracy_tensor
            batch.batch["len_scores"] = length_tensor

            reward_metrics = {
                f"reward/{key}": value for key, value in reduce_metrics(reward_metrics).items()
            }
            metrics.update(reward_metrics)

            group_n = self.config.worker.rollout.n

            def _to_sample_scores(x):
                """
                Convert token-level reward tensor to per-sample scalar scores.
                If x is already [B], return directly.
                If x is [B, T], sum over T.
                """
                if x.dim() == 1:
                    return x.float()
                return x.sum(dim=-1).float()

            def _mean_group_var(token_level_rewards):
                scores = _to_sample_scores(token_level_rewards)   # [B]
                B = scores.shape[0]
                assert B % group_n == 0, f"Batch size {B} is not divisible by group_n {group_n}"
                scores = scores.view(B // group_n, group_n)
                return scores.var(dim=1, unbiased=False).mean().item()

            def _mean_group_cov(x_token_level_rewards, y_token_level_rewards):
                x = _to_sample_scores(x_token_level_rewards)   # [B]
                y = _to_sample_scores(y_token_level_rewards)   # [B]
                B = x.shape[0]
                assert B % group_n == 0, f"Batch size {B} is not divisible by group_n {group_n}"
                x = x.view(B // group_n, group_n)  # [num_groups, group_n]
                y = y.view(B // group_n, group_n)
                x_centered = x - x.mean(dim=1, keepdim=True)
                y_centered = y - y.mean(dim=1, keepdim=True)
                cov = (x_centered * y_centered).mean(dim=1)  # unbiased=False
                return cov.mean().item()

            def _mean_group_corr(x_token_level_rewards, y_token_level_rewards, eps=1e-8):
                x = _to_sample_scores(x_token_level_rewards)   # [B]
                y = _to_sample_scores(y_token_level_rewards)   # [B]
                B = x.shape[0]
                assert B % group_n == 0, f"Batch size {B} is not divisible by group_n {group_n}"
                x = x.view(B // group_n, group_n)
                y = y.view(B // group_n, group_n)
                x_centered = x - x.mean(dim=1, keepdim=True)
                y_centered = y - y.mean(dim=1, keepdim=True)
                cov = (x_centered * y_centered).mean(dim=1)
                std_x = x_centered.pow(2).mean(dim=1).sqrt()
                std_y = y_centered.pow(2).mean(dim=1).sqrt()
                corr = cov / (std_x * std_y + eps)
                return corr.mean().item()

            def _batch_cov_from_scores(x_scores, y_scores):
                """
                x_scores, y_scores: [B]
                """
                x = x_scores.float()
                y = y_scores.float()
                x_centered = x - x.mean()
                y_centered = y - y.mean()
                cov = (x_centered * y_centered).mean()   # unbiased=False
                return cov.item()

            def _batch_corr_from_scores(x_scores, y_scores, eps=1e-8):
                """
                x_scores, y_scores: [B]
                """
                x = x_scores.float()
                y = y_scores.float()
                x_centered = x - x.mean()
                y_centered = y - y.mean()
                cov = (x_centered * y_centered).mean()
                std_x = x_centered.pow(2).mean().sqrt()
                std_y = y_centered.pow(2).mean().sqrt()
                corr = cov / (std_x * std_y + eps)
                return corr.item()

            def _mean_group_cov_from_scores(x_scores, y_scores):
                """
                x_scores, y_scores: [B]
                Compute covariance within each group, then average over groups.
                """
                x = x_scores.float()
                y = y_scores.float()
                B = x.shape[0]
                assert B % group_n == 0, f"Batch size {B} is not divisible by group_n {group_n}"

                x = x.view(B // group_n, group_n)  # [num_groups, group_n]
                y = y.view(B // group_n, group_n)

                x_centered = x - x.mean(dim=1, keepdim=True)
                y_centered = y - y.mean(dim=1, keepdim=True)
                cov = (x_centered * y_centered).mean(dim=1)  # [num_groups]
                return cov.mean().item()

            def _mean_group_corr_from_scores(x_scores, y_scores, eps=1e-8):
                """
                x_scores, y_scores: [B]
                Compute correlation within each group, then average over groups.
                """
                x = x_scores.float()
                y = y_scores.float()
                B = x.shape[0]
                assert B % group_n == 0, f"Batch size {B} is not divisible by group_n {group_n}"

                x = x.view(B // group_n, group_n)
                y = y.view(B // group_n, group_n)

                x_centered = x - x.mean(dim=1, keepdim=True)
                y_centered = y - y.mean(dim=1, keepdim=True)

                cov = (x_centered * y_centered).mean(dim=1)
                std_x = x_centered.pow(2).mean(dim=1).sqrt()
                std_y = y_centered.pow(2).mean(dim=1).sqrt()
                corr = cov / (std_x * std_y + eps)
                return corr.mean().item()

            metrics["reward_var/reward"] = _mean_group_var(reward_tensor)

            var = False
            if var:
                metrics["reward_var/accuracy"] = _mean_group_var(accuracy_tensor)
                metrics["reward_var/length"] = _mean_group_var(length_tensor)
                # 保留原来的：长度奖励 vs 准确性奖励（group级）
                metrics["reward_var/acc_length_cov"] = _mean_group_cov(accuracy_tensor, length_tensor)
                metrics["reward_var/acc_length_corr"] = _mean_group_corr(accuracy_tensor, length_tensor)
                # 新增：真实 response length vs 准确性奖励
                resp_lens = self._get_resp_lens(batch).float()   # [B]
                acc_scores = _to_sample_scores(accuracy_tensor) # [B]
                # batch级
                metrics["reward_var/acc_raw_length_cov_batch"] = _batch_cov_from_scores(resp_lens, acc_scores)
                metrics["reward_var/acc_raw_length_corr_batch"] = _batch_corr_from_scores(resp_lens, acc_scores)
                # group级
                metrics["reward_var/acc_raw_length_cov_group"] = _mean_group_cov_from_scores(resp_lens, acc_scores)
                metrics["reward_var/acc_raw_length_corr_group"] = _mean_group_corr_from_scores(resp_lens, acc_scores)
                

    def _compute_old_log_probs(self, batch: DataProto, timing_raw, temperature = None, eos_args = None):
        if temperature is not None:
            with _timer(f"old-t{temperature}", timing_raw):
                batch_t = deepcopy(batch)
                batch_t.meta_info["temperature"] = temperature
                old_log_probs = self.actor_rollout_wg.compute_log_probs(batch_t)
                batch.batch["old_log_probs_t"] = old_log_probs.batch["old_log_probs"]
        else:
            with _timer("old", timing_raw):
                batch.meta_info["temperature"] = 1.0
                old_log_probs = self.actor_rollout_wg.compute_log_probs(batch)
                batch = batch.union(old_log_probs)



    def _compute_ref_log_probs(self, batch: DataProto, timing_raw):
        if self.use_reference_policy:
            with _timer("ref", timing_raw):
                ref_log_probs = self.ref_policy_wg.compute_ref_log_probs(batch)
                batch = batch.union(ref_log_probs)

    def _compute_teacher_log_probs(self, batch: DataProto, timing_raw):
        if self.use_teacher_policy:
            with _timer("teacher", timing_raw):
                if "temperature" not in batch.meta_info:
                    batch.meta_info["temperature"] = 1.0
                teacher_log_probs = self.teacher_policy_wg.compute_teacher_log_probs(batch)
                batch = batch.union(teacher_log_probs)
        return batch

    def _compute_values(self, batch: DataProto, timing_raw):
        if self.use_critic:
            with _timer("values", timing_raw):
                values = self.critic_wg.compute_values(batch)
                batch = batch.union(values)

    def _compute_adv(self, batch: DataProto, timing_raw):
        with _timer("adv", timing_raw):
            # apply kl penalty if available
            if not self.config.algorithm.use_kl_loss and self.use_reference_policy:
                # apply kl penalty to reward
                batch, kl_metrics = apply_kl_penalty(
                    batch, kl_ctrl=self.kl_ctrl, kl_penalty=self.config.algorithm.kl_penalty
                )
            else:
                batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
            batch = compute_advantage(
                batch,
                adv_estimator=self.config.algorithm.adv_estimator,
                gamma=self.config.algorithm.gamma,
                lam=self.config.algorithm.lam,
            )
    def _get_resp_lens(self, dp: DataProto) -> torch.Tensor:
        if "response_mask" in dp.batch:
            return dp.batch["response_mask"].sum(dim=1)
        elif "responses" in dp.batch:
            return (dp.batch["responses"] != self.tokenizer.eos_token_id).sum(dim=1)
        else:
            return torch.zeros(
                len(dp),
                dtype=torch.long,
                device=dp.batch["advantages"].device
            )
    def _apply_drl_importance_weight(
        self,
        batch: DataProto,
        metrics: dict,
        clip_ratio: float = 3.0,
        rho_apply_to: str = "positive",  # "all", "positive", "negative"
        if_normalized: bool = True,
        temperature_as_teacher: bool = False
    ):
        old_log_probs = batch.batch["old_log_probs"]              # [B, T]
        if temperature_as_teacher:
            teacher_log_probs = batch.batch["old_log_probs_t"]    # [B, T] 
        else:
            teacher_log_probs = batch.batch["teacher_log_probs"]  # [B, T]
        advantages = batch.batch["advantages"]                    # [B, T]
        response_mask = batch.batch["response_mask"]              # [B, T]

        valid_mask = response_mask.bool()
        # token-level log rho: [B, T]
        token_log_rho = teacher_log_probs - old_log_probs
        # token-level raw rho: [B, T]
        rho_raw = torch.exp(token_log_rho.clamp(min=-80.0, max=80.0))

        clip_low = 1.0 / clip_ratio
        clip_high = clip_ratio
        # token-level clipped rho: [B, T]
        rho_clipped = rho_raw.clamp(min=clip_low, max=clip_high)

        if if_normalized:
            eps = 1e-12
            mask_float = valid_mask.to(rho_clipped.dtype)

            # 在 log 空间计算每条 response 的平均 log-rho
            log_rho_clipped = torch.log(rho_clipped.clamp_min(eps))

            mean_log_rho_per_response = (
                (log_rho_clipped * mask_float).sum(dim=-1, keepdim=True)
                / mask_float.sum(dim=-1, keepdim=True).clamp_min(1.0)
            )

            # 除以每条 response 的 rho 几何均值
            rho_clipped = torch.exp(
                log_rho_clipped - mean_log_rho_per_response
            )

            # padding 位置保持权重为 1
            rho_clipped = torch.where(
                valid_mask,
                rho_clipped,
                torch.ones_like(rho_clipped),
            )
        # -------- choose where to apply rho --------
        if rho_apply_to == "all":
            apply_mask = valid_mask
        elif rho_apply_to == "positive":
            apply_mask = valid_mask & (advantages > 0)
        elif rho_apply_to == "negative":
            apply_mask = valid_mask & (advantages < 0)
        else:
            raise ValueError(
                f"Unknown rho_apply_to={rho_apply_to}. "
                "Expected one of: 'all', 'positive', 'negative'."
            )

        # positions not selected use weight 1.0
        effective_rho = torch.ones_like(rho_clipped)
        effective_rho = torch.where(apply_mask, rho_clipped, effective_rho)

        # token-level reweighting
        batch.batch["advantages"] = advantages * effective_rho

        # -------- metrics --------
        with torch.no_grad():
            eps = 1e-12

            valid_token_log_rho = token_log_rho[valid_mask].detach()
            valid_rho_raw = rho_raw[valid_mask].detach()
            valid_rho_clipped = rho_clipped[valid_mask].detach()

            selected_rho = rho_clipped[apply_mask].detach()
            selected_raw_rho = rho_raw[apply_mask].detach()

            metrics["rho/apply_to"] = {"all": 0, "positive": 1, "negative": 2}[rho_apply_to]
            metrics["rho/apply_frac"] = apply_mask.float().sum().item() / valid_mask.float().sum().clamp_min(1.0).item()

            metrics["rho/token_rho_raw_min"] = valid_rho_raw.min().item()
            metrics["rho/token_rho_raw_max"] = valid_rho_raw.max().item()
            metrics["rho/token_rho_raw_mean"] = valid_rho_raw.mean().item()
            metrics["rho/token_rho_raw_geomean"] = torch.exp(valid_token_log_rho.mean()).item()

            metrics["rho/token_rho_clipped_min"] = valid_rho_clipped.min().item()
            metrics["rho/token_rho_clipped_max"] = valid_rho_clipped.max().item()
            metrics["rho/token_rho_clipped_mean"] = valid_rho_clipped.mean().item()
            metrics["rho/token_rho_clipped_geomean"] = torch.exp(
                torch.log(valid_rho_clipped.clamp_min(eps)).mean()
            ).item()

            metrics["rho/token_clip_low_frac"] = (valid_rho_raw < clip_low).float().mean().item()
            metrics["rho/token_clip_high_frac"] = (valid_rho_raw > clip_high).float().mean().item()
            metrics["rho/token_clip_frac"] = (
                (valid_rho_raw < clip_low) | (valid_rho_raw > clip_high)
            ).float().mean().item()

            metrics["rho/token_log_rho_min"] = valid_token_log_rho.min().item()
            metrics["rho/token_log_rho_max"] = valid_token_log_rho.max().item()
            metrics["rho/token_log_rho_mean"] = valid_token_log_rho.mean().item()

            # selected-only metrics
            if selected_rho.numel() > 0:
                metrics["rho/selected_rho_mean"] = selected_rho.mean().item()
                metrics["rho/selected_rho_min"] = selected_rho.min().item()
                metrics["rho/selected_rho_max"] = selected_rho.max().item()
                metrics["rho/selected_clip_frac"] = (
                    (selected_raw_rho < clip_low) | (selected_raw_rho > clip_high)
                ).float().mean().item()
            else:
                metrics["rho/selected_rho_mean"] = 1.0
                metrics["rho/selected_rho_min"] = 1.0
                metrics["rho/selected_rho_max"] = 1.0
                metrics["rho/selected_clip_frac"] = 0.0

            # sequence-level log rho only for debugging, not used for clipping
            seq_log_rho = (token_log_rho * response_mask.to(token_log_rho.dtype)).sum(dim=-1)
            metrics["rho/seq_log_rho_min"] = seq_log_rho.detach().min().item()
            metrics["rho/seq_log_rho_max"] = seq_log_rho.detach().max().item()
            metrics["rho/seq_log_rho_mean"] = seq_log_rho.detach().mean().item()

        return batch

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        self.logger = Tracker(loggers=self.config.trainer.logger, config=self.config.to_dict())
        self.global_step = 0
        replay_buffer = Buffer(capacity=0)
        batch_buffer = Buffer(capacity=0)
        difficult_record = []
        overlong_record = []
        overlong_positive_record = []
        entropy = 0.5
        entropy_delta_list = []
        
        val_metrics: Optional[Dict[str, Any]] = None

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.val_before_train:
            val_metrics = self._validate()
            self.logger.log(data=val_metrics, step=self.global_step)
            if self.config.trainer.val_only:
                return

        for _ in tqdm(range(self.config.trainer.total_episodes), desc="Episode", position=0):
            for batch_dict in tqdm(self.train_dataloader, desc="Running step", position=1):
                self.global_step += 1
                if self.global_step > self.training_steps:
                    break

                metrics, timing_raw = {}, {}
                initial_batch: DataProto = DataProto.from_single_dict(batch_dict)

                 # pop those keys for generation
                if "multi_modal_data" in initial_batch.non_tensor_batch.keys():
                    gen_batch = initial_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids", "multi_modal_data"],
                    )
                else:
                    gen_batch = initial_batch.pop(
                        batch_keys=["input_ids", "attention_mask", "position_ids"],
                        non_tensor_batch_keys=["raw_prompt_ids"],
                    )
                gen_batch_bak = deepcopy(gen_batch)
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):  # wg: worker group
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    

                    initial_batch.non_tensor_batch["uid"] = np.array(
                        [str(uuid.uuid4()) for _ in range(len(initial_batch.batch))], dtype=object
                    )
                    batch = initial_batch.repeat(repeat_times=self.config.worker.rollout.n, interleave=True)
                    batch_bak = deepcopy(batch)
                    batch = batch.union(gen_batch_output)
                    initial_batch_size = len(batch)
                    self._compute_reward(batch, timing_raw, metrics)
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                    self._compute_values(batch, timing_raw)
                    self._compute_adv(batch, timing_raw)
                    count_occurrences(batch, "advantages")
                    try:
                        if "responses" in batch.batch:
                            first_resp_ids = batch.batch["responses"][0]
                            if hasattr(first_resp_ids, "cpu"):
                                first_resp_ids = first_resp_ids.cpu().tolist()
                            decoded_first_response = self.tokenizer.decode(first_resp_ids, skip_special_tokens=True)
                            print("First response (decoded):", decoded_first_response)
                        else:
                            print("No 'responses' key in batch to decode.")
                    except Exception as e:
                        print("Failed to decode first response:", e)


                    alg = self.config.trainer.algorithm
                    if alg=="Distilled-RL":                   
                        self._compute_old_log_probs(batch, timing_raw)
                        self._compute_ref_log_probs(batch, timing_raw)
                        self._compute_teacher_log_probs(batch, timing_raw)
                        batch = self._apply_drl_importance_weight(
                            batch=batch,
                            metrics=metrics,
                            clip_ratio=self.config.trainer.rho_clip,
                            rho_apply_to=self.config.trainer.rho_sample,   # "all" / "positive" / "negative"
                            if_normalized=self.config.trainer.rho_normalized,
                            temperature_as_teacher=self.config.trainer.temperature_as_teacher,
                        )
                        self._update_critic(batch, timing_raw, metrics)
                        self._update_actor(batch, timing_raw, metrics)
                    elif alg=="GRPO":                   
                        self._compute_old_log_probs(batch, timing_raw)
                        self._compute_ref_log_probs(batch, timing_raw)
                        self._update_critic(batch, timing_raw, metrics)
                        self._update_actor(batch, timing_raw, metrics)


                      

                    # validate
                    if (
                        self.val_reward_fn is not None
                        and self.config.trainer.val_freq > 0
                        and self.global_step % self.config.trainer.val_freq == 0
                        ):
                        with _timer("validation", timing_raw):
                            val_metrics = self._validate()

                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and self.global_step % self.config.trainer.save_freq == 0:
                        with _timer("save_checkpoint", timing_raw):
                            self._save_checkpoint()

                # collect metrics
                num_gpus = self.resource_pool_manager.get_num_gpus()
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, num_gpus=num_gpus))
                self.logger.log(data=metrics, step=self.global_step)

        # perform validation after training
        if self.val_reward_fn is not None:
            if (
                val_metrics is None
                or self.config.trainer.val_freq <= 0
                or self.global_step % self.config.trainer.val_freq != 0
            ):
                val_metrics = self._validate()
                self.logger.log(data=val_metrics, step=self.global_step)

            print(f"Final validation metrics: {convert_dict_to_str(val_metrics)}")

        #if self.config.trainer.save_freq <= 0 or self.global_step % self.config.trainer.save_freq != 0:
        #    self._save_checkpoint()
