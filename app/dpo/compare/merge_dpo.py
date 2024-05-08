from pathlib import Path
import shutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizer

from loguru import logger
from peft import PeftModel
from icecream import ic
ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120


base_model = AutoModelForCausalLM.from_pretrained(
    "/mnt/nas1/models/alignment-handbook/zephyr-7b-sft-full",
    torch_dtype=getattr(torch, "bfloat16"),
)
peft_model_id = "data/zephyr-7b-dpo-qlora/checkpoint-7000"
model = PeftModel.from_pretrained(base_model, peft_model_id)
model = model.merge_and_unload()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(peft_model_id, trust_remote_code=True)
messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, return_tensors="pt"
)
ic(type(input_ids), input_ids)

input_ids = input_ids.to(device)
generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.01,
    # "top_p": 0.9,
    "pad_token_id": tokenizer.eos_token_id,
}

model = model.to(device)
outputs = model.generate(
    input_ids,
    **generate_kwargs,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))

save = 0
if save:
    out_dir = Path("/mnt/nas1/models/alignment-handbook/zephyr-7b-sft-full_with_dpo")
    shutil.rmtree(out_dir, ignore_errors=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)

logger.info("end")
