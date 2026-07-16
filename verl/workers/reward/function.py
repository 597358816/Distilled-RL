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

import importlib.util
import os
import sys
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from typing import Callable, Dict, List, Optional, Tuple, TypedDict

from sympy import group

import torch
from transformers import PreTrainedTokenizer

from ...protocol import DataProto
from .config import RewardConfig


class RewardScore(TypedDict):
    overall: float
    format: Optional[float]
    accuracy: Optional[float]


ScoreFunction = Callable[[str, str], RewardScore]




@dataclass
class FunctionRewardManager:
    config: RewardConfig
    tokenizer: PreTrainedTokenizer

    def __post_init__(self):
        """Load score function."""
        if ":" not in self.config.score_function:
            file_path = self.config.score_function
            function_name = "main"
        else:
            file_path, function_name = self.config.score_function.split(":", maxsplit=1)

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Score function file {file_path} not found.")

        spec = importlib.util.spec_from_file_location("custom_score_fn", file_path)
        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules["custom_score_fn"] = module
            spec.loader.exec_module(module)
        except Exception as e:
            raise RuntimeError(f"Failed to load score function: {e}")

        if not hasattr(module, function_name):
            raise AttributeError(f"Module {module} does not have function {function_name}.")

        score_fn: ScoreFunction = getattr(module, function_name)
        print(f"Using score function `{function_name}` from `{file_path}`.")
        self.score_fn = partial(score_fn, **self.config.score_function_kwargs)
        
        
    def __call__(self, data: DataProto) -> Tuple[torch.Tensor, Dict[str, List[float]]]:
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        length_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        accuracy_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_metrics = defaultdict(list)
        n = 8
        group = []
        for i in range(len(data)):
            data_item = data[i]  # DataProtoItem
            response_ids = data_item.batch["responses"]
            response_mask = data_item.batch["response_mask"]
            valid_response_length = response_mask.sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(
                valid_response_ids, skip_special_tokens=self.config.skip_special_tokens
            )
            ground_truth = data_item.non_tensor_batch["ground_truth"]
            score = self.score_fn(response_str, ground_truth)

            if self.config.length_reward == "LP1":
                length_min = 4096
                length_max = 8192
                length_reward = 0.0
                if valid_response_length >= length_min:
                    length_reward = -1 * (valid_response_length - length_min) / (length_max - length_min)
                score["length"] = float(length_reward)
                score["overall"] = float(score["overall"]) #+ float(length_reward)
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                length_tensor[i, valid_response_length - 1] = score["length"]
                accuracy_tensor[i, valid_response_length - 1] = float(score["accuracy"])
                for key, value in score.items():
                    reward_metrics[key].append(float(value))
            elif self.config.length_reward == "LP2": 
                lp2_weight = 1.0
                correct01 = 1 if float(score["overall"]) > 0.5 else 0 
                group.append((i, valid_response_length, score, correct01)) 
                if len(group) == n: 
                    self._flush_group_lp2(group, reward_tensor, reward_metrics, lp2_weight)
                continue
            elif self.config.length_reward == "LP22": 
                lp2_weight = 1.0
                correct01 = 1 if float(score["overall"]) > 0.5 else 0 
                group.append((i, valid_response_length, score, correct01)) 
                if len(group) == n: 
                    self._flush_group_lp22(group, reward_tensor, reward_metrics, length_tensor, accuracy_tensor, lp2_weight)
                continue
            elif self.config.length_reward == "ShorterBetter":
                sb_alpha = 1.0
                sb_beta  = float(1e-3)
                correct01 = 1 if float(score["overall"]) > 0.5 else 0
                group.append((i, valid_response_length, score, correct01))
                if len(group) == n:
                    self._flush_group_shorterbetter(
                        group, reward_tensor, reward_metrics, alpha=sb_alpha, beta=sb_beta
                    )
                continue
            elif self.config.length_reward == "LC-R1":
                lcr1_length_w = 1.0
                lcr1_compress_w = 1.0
                lcr1_think_start_ids = self.tokenizer.encode("<think>", add_special_tokens=False)
                lcr1_think_end_ids = self.tokenizer.encode("</think>", add_special_tokens=False)
                lcr1_valid_len = valid_response_length
                lcr1_valid_ids = valid_response_ids
                acc = self._as_float_accuracy(score)
                self._lcr1_step_masked(
                    group=group,
                    idx=i,
                    response_mask_1d=response_mask,     # 直接原地修改 mask，实现 o'
                    valid_len=lcr1_valid_len,
                    valid_ids=lcr1_valid_ids,
                    accuracy=acc,
                    ground_truth=ground_truth,
                    reward_tensor=reward_tensor,
                    reward_metrics=reward_metrics,
                    n=n,
                    length_weight=lcr1_length_w,
                    compress_weight=lcr1_compress_w,
                    think_start_ids=lcr1_think_start_ids,
                    think_end_ids=lcr1_think_end_ids,
                )
                continue
            else:                       
                reward_tensor[i, valid_response_length - 1] = score["overall"]
                accuracy_tensor[i, valid_response_length - 1] = float(score["accuracy"])
                for key, value in score.items():
                    reward_metrics[key].append(value)

        return reward_tensor, reward_metrics, length_tensor, accuracy_tensor
    def _find_subseq(self, haystack, needle):
        """Return start idx of needle in haystack, else -1. Both are Python lists[int]."""
        if not needle or len(needle) > len(haystack):
            return -1
        L = len(needle)
        for i in range(len(haystack) - L + 1):
            if haystack[i:i + L] == needle:
                return i
        return -1
        
    def _as_float_accuracy(self, score) -> float:
        """score_fn 只返回 accuracy：兼容 float / dict。"""
        if isinstance(score, dict):
            if "accuracy" in score:
                return float(score["accuracy"])
            if "overall" in score:
                return float(score["overall"])
            # 兜底
            return float(next(iter(score.values())))
        return float(score)

    def _last_one_pos(self, mask_1d: torch.Tensor) -> int:
        """返回最后一个 mask==1 的位置；如果全 0 返回 -1。"""
        idx = torch.nonzero(mask_1d > 0, as_tuple=False)
        return int(idx[-1].item()) if idx.numel() > 0 else -1

    def _lcr1_step_masked(
        self,
        *,
        group: list,
        idx: int,
        response_mask_1d: torch.Tensor,  # 1D view of data_item.batch["response_mask"]
        valid_len: int,
        valid_ids: torch.Tensor,         # 1D ids already truncated to valid_len
        accuracy: float,
        ground_truth,
        reward_tensor: torch.Tensor,
        reward_metrics: dict,
        n: int,
        length_weight: float,
        compress_weight: float,          # 对应论文 gamma
        think_start_ids: list,
        think_end_ids: list,
    ):
        correct01 = 1 if float(accuracy) > 0.5 else 0

        # ---- locate <think> ... </think> spans ----
        ids_list = valid_ids.tolist()
        start_tag_start = self._find_subseq(ids_list, think_start_ids)
        end_tag_start   = self._find_subseq(ids_list, think_end_ids)

        think_end_pos = None
        orig_think_len = 0
        comp_think_len = 0
        r_comp = 0.0

        if start_tag_start >= 0 and end_tag_start >= 0:
            think_content_start = start_tag_start + len(think_start_ids)
            think_content_end   = end_tag_start  # exclusive
            end_tag_end = end_tag_start + len(think_end_ids)
            think_end_pos = min(valid_len - 1, end_tag_end - 1)  # end-tag 最后一个 token

            if think_content_end >= think_content_start:
                orig_think_len = think_content_end - think_content_start

            # ---- compression by masking redundant thinking tokens ----
            # 仍用你原来的“GT token 子串匹配”方式（简单但可能偏保守）
            if correct01 == 1 and orig_think_len > 0:
                gt_ids = self.tokenizer.encode(str(ground_truth), add_special_tokens=False)
                thinking_ids = ids_list[think_content_start:think_content_end]
                pos = self._find_subseq(thinking_ids, gt_ids) if gt_ids else -1

                if pos >= 0:
                    comp_think_len = pos + len(gt_ids)

                    # mask 冗余 thinking: [think_content_start + comp_think_len, think_content_end)
                    cut_l = think_content_start + comp_think_len
                    cut_r = think_content_end
                    if cut_l < cut_r:
                        response_mask_1d[cut_l:cut_r] = 0  # 关键：用 mask 实现 o'

                    # r_comp = 1 - |t(o')|/|t(o)|
                    r_comp = 1.0 - float(comp_think_len) / float(max(1, orig_think_len))
                else:
                    # 论文：correct 但 ans 不在 t(o') => -1 penalty
                    r_comp = -1.0
            else:
                r_comp = 0.0

        # 压缩后的长度：|o'|
        comp_total_len = int(response_mask_1d.sum().item())
        terminal_pos = self._last_one_pos(response_mask_1d)

        group.append({
            "idx": idx,
            "valid_len": int(valid_len),
            "terminal_pos": int(terminal_pos),
            "accuracy": float(accuracy),
            "overall": float(accuracy),   # base reward
            "correct01": int(correct01),
            "comp_total_len": int(comp_total_len),
            "think_end_pos": think_end_pos,  # compress reward 的落点
            "r_comp": float(r_comp),
            "orig_think_len": int(orig_think_len),
            "comp_think_len": int(comp_think_len),
        })

        if len(group) == n:
            self._flush_group_lcr1_masked(
                group, reward_tensor, reward_metrics,
                length_weight=length_weight,
                compress_weight=compress_weight,
            )

    def _flush_group_lcr1_masked(
        self,
        group: list,
        reward_tensor: torch.Tensor,
        reward_metrics: dict,
        *,
        length_weight: float,
        compress_weight: float,
    ):
        """
        LC-R1 flush（按论文）：
        r_len 只对 correct 且基于压缩长度 |o'|
        r_tilde = r_base + α r_len
        r_combine = r_tilde - mean(r_tilde)   (group mean subtraction)
        token reward: terminal += r_combine; </think> += γ r_comp
        """
        if not group:
            return

        # max_{j in C} |o'_j|
        correct_lens = [g["comp_total_len"] for g in group if g["correct01"] == 1]
        max_len_c = max(correct_lens) if correct_lens else None

        # r_tilde list for mean subtraction
        r_tilde_list = []
        for g in group:
            if g["correct01"] == 1 and max_len_c is not None and max_len_c > 0:
                r_len = 1.0 - float(g["comp_total_len"]) / float(max_len_c)
            else:
                r_len = 0.0
            g["r_len"] = float(r_len)
            r_tilde = float(g["overall"]) + float(length_weight) * float(r_len)
            g["r_tilde"] = float(r_tilde)
            r_tilde_list.append(float(r_tilde))

        mean_r = sum(r_tilde_list) / float(max(1, len(r_tilde_list)))

        for g in group:
            idx = g["idx"]
            terminal_pos = g["terminal_pos"]
            think_end_pos = g["think_end_pos"]

            r_combine = float(g["r_tilde"]) - float(mean_r)

            # terminal reward 放在“最后一个未被 mask 的 token”
            if terminal_pos is not None and terminal_pos >= 0:
                reward_tensor[idx, terminal_pos] += float(r_combine)

            # compress reward 只打在 </think>（end-tag 最后 token）
            if think_end_pos is not None and 0 <= think_end_pos < g["valid_len"]:
                reward_tensor[idx, think_end_pos] += float(compress_weight) * float(g["r_comp"])

            # metrics
            reward_metrics["accuracy"].append(float(g["accuracy"]))
            reward_metrics["overall"].append(float(g["overall"]))
 

        group.clear()

    def _lcr1_finalize_masked(
        self,
        group: list,
        reward_tensor: torch.Tensor,
        reward_metrics: dict,
        *,
        length_weight: float,
        compress_weight: float,
    ):
        """修复 bug：最后不足 n 的 group 也要 flush。"""
        if group:
            self._flush_group_lcr1_masked(
                group, reward_tensor, reward_metrics,
                length_weight=length_weight,
                compress_weight=compress_weight,
            )



    def _flush_group_lp2(self, group, reward_tensor, reward_metrics, lp2_weight: float):
        if not group:
            return group
        lens = [g[1] for g in group]
        min_len = min(lens)
        max_len = max(lens)
        if max_len == min_len:
            len_rewards = [0.0] * len(group)
        else:
            denom = float(max_len - min_len)
            len_rewards = []
            for (_, l, _, correct01) in group:
                lam = 0.5 - (float(l - min_len) / denom)  # in [-0.5, 0.5]
                if correct01 == 1:
                    lr = lam
                else:
                    lr = min(0.0, lam) 
                len_rewards.append(lr)
        for (idx, valid_len, score, _), lr in zip(group, len_rewards): 
            score["overall"] = float(score["overall"]) + lp2_weight * float(lr) 
            reward_tensor[idx, valid_len - 1] = score["overall"] 
            for key, value in score.items(): 
                reward_metrics[key].append(float(value))
        group.clear()
        return group

    def _flush_group_lp22(self, group, reward_tensor, reward_metrics, length_tensor, accuracy_tensor, lp2_weight: float):
        if not group:
            return group
        lens = [g[1] for g in group]
        min_len = min(lens)
        max_len = max(lens)
        if max_len == min_len:
            len_rewards = [0.0] * len(group)
        else:
            denom = float(max_len - min_len)
            len_rewards = []
            for (_, l, _, correct01) in group:
                lam = 0.5 - (float(l - min_len) / denom)  # in [-0.5, 0.5]
                if correct01 == 1:
                    lr = lam
                else:
                    lr = min(0.0, lam) 
                len_rewards.append(lr)
        for (idx, valid_len, score, _), lr in zip(group, len_rewards): 
            score["length"] = lp2_weight * float(lr) 
            score["overall"] = float(score["overall"]) + lp2_weight * float(lr) 
            reward_tensor[idx, valid_len - 1] = score["overall"] 
            length_tensor[idx, valid_len - 1] = score["length"] 
            accuracy_tensor[idx, valid_len - 1] = score["accuracy"]
            for key, value in score.items(): 
                reward_metrics[key].append(float(value))
        group.clear()
        return group
    
    def _flush_group_shorterbetter(
        self,
        group,
        reward_tensor,
        reward_metrics,
        alpha: float,
        beta: float,
    ):

        if not group:
            return group

        lens = [int(g[1]) for g in group]
        correct_lens = [int(l) for (_, l, _, c) in group if int(c) == 1]

        if len(correct_lens) > 0:
            sol_len = min(correct_lens)
        else:
            sol_len = sum(lens) / float(len(lens))  # mean length when all wrong

        for (idx, valid_len, score, correct01) in group:
            valid_len = int(valid_len)
            correct01 = int(correct01)

            sb_reward = float(alpha) * float(correct01) - float(beta) * abs(float(valid_len) - float(sol_len))

            score["overall"] = float(sb_reward)
            reward_tensor[idx, valid_len - 1] = float(score["overall"])
            for k, v in score.items():
                reward_metrics[k].append(float(v))
        group.clear()
        return group