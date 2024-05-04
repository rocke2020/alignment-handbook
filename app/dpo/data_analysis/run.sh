export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/mnt/nas2/huggingface
# 
file=app/dpo/data_analysis/dpo_data_analysis.py
python $file \
    2>&1  </dev/null | tee $file.log