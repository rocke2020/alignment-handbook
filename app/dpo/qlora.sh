export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/nas2/huggingface
gpu=$1
if [ -z $gpu ]; then
    gpu=0,1
fi
export CUDA_VISIBLE_DEVICES=$gpu
# --main_process_port=29555 \
# --use_flash_attention_2=false \
# 2>&1  </dev/null | tee app/dpo/qlora2.log
nohup accelerate launch \
--config_file recipes/accelerate_configs/multi_gpu.yaml \
--num_processes=2 \
scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_qlora.yaml \
> app/dpo/qlora-3.log 2>&1 &