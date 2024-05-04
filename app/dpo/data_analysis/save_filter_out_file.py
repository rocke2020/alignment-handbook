import json
import os
import random
import sys

import numpy as np

from icecream import ic


sys.path.append(os.path.abspath("."))
from src.alignment.decontaminate import FILTER_OUT


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)


# HumanEval solutions that are considered simple/generic enough to be kept in the training dataset
HUMAN_EVAL_STRINGS_OK = [
    "return x + y",
    "return len(string)",
    "return n**2",
    "return " ".join(strings)",
]

out_file = "app/dpo/data/filter_out.json"
with open(out_file, "w", encoding="utf-8") as f:
    json.dump(FILTER_OUT, f, ensure_ascii=False, indent=4)
