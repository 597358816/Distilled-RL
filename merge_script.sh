MODEL_NAME="Qwen3-1.7B"


#for step in $(seq 200 10 270)
#for step in $(seq 50 10 120)
#for step in $(seq 40 10 150)
#for step in $(seq 60 10 130)
for step in $(seq 20 20 160)
do
    echo ${step}
    RUN_NAME="qwen3-1.7b-GRPO-ref"
    python3 ./scripts/model_merger.py --local_dir "/vepfs-mlp2/c20250203/250602012/checkpoints/OPD/${MODEL_NAME}/${RUN_NAME}/global_step_${step}/actor/"
done

#rclone copy "/home/dataset-assist-0/wc/checkpoints/${MODEL_NAME}/${RUN_NAME}" "beijing11:bucket-c20250203/wc/checkpoints/${MODEL_NAME}/${RUN_NAME}"  --progress --transfers=48

