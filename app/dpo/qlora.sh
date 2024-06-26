export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/nas2/huggingface
gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu
# --main_process_port=29555 \
# --use_flash_attention_2=false \
# deepspeed_zero3 multi_gpu
nohup accelerate launch \
--config_file recipes/accelerate_configs/deepspeed_zero2.yaml \
--num_processes=2 \
--main_process_port=29555 \
scripts/run_dpo.py recipes/zephyr-7b-beta/dpo/config_qlora.yaml \
> app/dpo/qlora.log 2>&1 &