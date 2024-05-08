import logging
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from icecream import ic
from loguru import logger


sys.path.append(os.path.abspath("."))


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

messages = [
    {
        "role": "system",
        "content": "You are a pirate chatbot who always responds in pirate speak!",
    },
    {"role": "user", "content": "Who are you?"},
]
model_id = "/mnt/nas1/models/alignment-handbook/zephyr-7b-sft-full_with_dpo"
# model_id = "data/zephyr-7b-dpo-qlora/checkpoint-7000"
# model_id = "/mnt/nas1/models/alignment-handbook/zephyr-7b-sft-full"

ic(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)

"""  
    input_str: <|system|>
               You are a pirate chatbot who always responds in pirate speak!</s>
               <|user|>
               Who are you?</s>

"""
input_str = tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, tokenize=False
)
# ic(input_str)

"""  
    input_str: <|system|>
               You are a pirate chatbot who always responds in pirate speak!</s>
               <|user|>
               Who are you?</s>
               <|assistant|>

"""
input_str = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=False
)
# ic(input_str)

# return_tensors="pt", returns 2d tensor [[1, 2, 3]], here Must return tensor
input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=False, return_tensors="pt"
)
# ic(type(input_ids), input_ids)

terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
# terminators: [128001, 128009]
# ic(terminators)

enable_generate = 1
if not enable_generate:
    logger.info("end")
    sys.exit(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
input_ids = input_ids.to(device)

use_quantization = 1
logger.info(f"{use_quantization = }")
if use_quantization:
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
        # attn_implementation="flash_attention_2",
        quantization_config=quantization_config,
    )
else:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map=device,
    )

config_dict = model.config.to_dict()
model_context_window = int(config_dict.get("max_position_embeddings", 100))
# logger.info(f"{model_context_window = }")
generate_kwargs = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.01,
    # "top_p": 0.9,
    "pad_token_id": tokenizer.eos_token_id,
}
outputs = model.generate(
    input_ids,
    **generate_kwargs,
)
response = outputs[0][input_ids.shape[-1] :]
print(tokenizer.decode(response, skip_special_tokens=True))
logger.info("end")
