import json
import math
import os
import random
import re
import shutil
import sys
import warnings
from collections import defaultdict
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm
from transformers import AutoTokenizer

from icecream import ic
from loguru import logger


sys.path.append(os.path.abspath("/home/qcdong/codes/alignment-handbook"))


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

# model_path = "/mnt/nas1/models/meta-llama/Meta-Llama-3-8B-Instruct"
model_path = "/mnt/nas1/models/alignment-handbook/zephyr-7b-sft-full"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tests = [
    "Percy urges James to go forward, and after James is hesitant to do so, **Percy** decides that he must be brave as he buffers up to James to get him to safety.",
]

# false
ic("<|im_start|>" in tokenizer.chat_template)

"""
zephyr-7b-sft-full
{% for message in messages %}
{% if message['role'] == 'user' %}
{{ '<|user|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'system' %}
{{ '<|system|>
' + message['content'] + eos_token }}
{% elif message['role'] == 'assistant' %}
{{ '<|assistant|>
'  + message['content'] + eos_token }}
{% endif %}
{% if loop.last and add_generation_prompt %}
{{ '<|assistant|>' }}
{% endif %}
{% endfor %}

llama3-8b-instruct
{% set loop_messages = messages %}{% for message in loop_messages %}{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>'+ message['content'] | trim + '<|eot_id|>' %}{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}{{ '<|start_header_id|>assistant<|end_header_id|>

' }}

"""
print(tokenizer.chat_template)

def test_tokenizer():
    for s in tests:
        r = tokenizer(s, add_special_tokens=False)
        ic(r)
        r = tokenizer(s, add_special_tokens=True)
        ic(r)
        ss = tokenizer.convert_ids_to_tokens([1])
        ic(ss)
        ic(tokenizer.bos_token_id, tokenizer.bos_token)


if __name__ == "__main__":
    pass
