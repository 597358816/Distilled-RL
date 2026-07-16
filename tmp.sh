# for step in $(seq 200 40 340)
# do
#     echo ${step}
#     python3 /home/dataset-assist-0/wc/EasyR1/scripts/model_merger.py --local_dir /home/dataset-assist-0/wc/checkpoints/Qwen2.5-3B/qwen2.5-3b-Ins-head/global_step_${step}/actor/
# done


python3 /vepfs-mlp2/c20250203/250602012/EasyR1/scripts/model_merger.py --local_dir /vepfs-mlp2/c20250203/250602012/checkpoints/Qwen3-4B/qwen3-4b-GRPO2/global_step_139/actor
python3 /vepfs-mlp2/c20250203/250602012/EasyR1/scripts/model_merger.py --local_dir /vepfs-mlp2/c20250203/250602012/checkpoints/Qwen3-8B/qwen3-8b-GRPO/global_step_139/actor


