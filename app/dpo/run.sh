export HF_ENDPOINT=https://hf-mirror.com
gpu=$1
if [ -z $gpu ]; then
    gpu=0
fi
export CUDA_VISIBLE_DEVICES=$gpu

# merge_dpo quick_start
file=app/dpo/compare/quick_start.py
python $file \
    2>&1  </dev/null | tee $file.log
