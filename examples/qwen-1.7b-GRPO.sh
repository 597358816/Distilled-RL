set -x

MODEL_PATH=Qwen3/Qwen3-1.7B
NAME="Qwen3-1.7B-Distilled-RL"

FORMAT_PROMPT="""You FIRST think about the reasoning process as an internal monologue and then provide the final answer. The reasoning process MUST BE enclosed within <think> </think> tags. The final answer MUST BE put in \boxed{}."""


/vepfs-mlp2/c20250203/250602012/Anaconda/envs/easyr1/bin/python -m verl.trainer.main \
    config=config.yaml \
    worker.actor.model.model_path="${MODEL_PATH}" \
    data.train_files=./train_dapo.parquet \
    data.max_response_length=8192 \
    data.rollout_batch_size=128 \
    data.format_prompt="${FORMAT_PROMPT}" \
    worker.rollout.n=8 \
    worker.rollout.max_num_batched_tokens=10240 \
    worker.rollout.gpu_memory_utilization=0.6 \
    trainer.experiment_name="${NAME}" \
    trainer.project_name="Distilled-RL" \
    trainer.val_freq=-1 \
    trainer.save_limit=8 \
    trainer.save_freq=20 \
    trainer.total_episodes=2 \
    trainer.max_steps=160 \
    trainer.val_before_train=false \
    worker.actor.micro_batch_size_per_device_for_update=2 \
    worker.actor.micro_batch_size_per_device_for_experience=4 \
    worker.actor.global_batch_size=64 \
    worker.teacher.use_teacher=false \
    trainer.algorithm="GRPO" \
    trainer.disable_kl=false \




