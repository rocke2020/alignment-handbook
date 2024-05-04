import json
import math
import os
import random
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from datasets import (
    DatasetDict,
    concatenate_datasets,
    load_dataset,
    load_dataset_builder,
    load_from_disk,
)
from pandas import DataFrame
from tqdm import tqdm

from icecream import ic
from loguru import logger


sys.path.append(os.path.abspath("."))


ic.configureOutput(includeContext=True, argToStringFunction=str)
ic.lineWrapWidth = 120
SEED = 0
random.seed(SEED)
np.random.seed(SEED)

dataset_mixer = ["HuggingFaceH4/ultrafeedback_binarized"]


def get_general_info() -> DatasetDict:
    for dset_name in dataset_mixer:
        # dataset = load_dataset(dset_name, split="test")
        ds_builder = load_dataset_builder(dset_name)
        ic(ds_builder.info)
        ic(ds_builder.info.features)
        break
    return ds_builder


def check_item():
    out_file = "app/dpo/data/ultrafeedback_binarized/test_prefs.json"
    Path(out_file).parent.mkdir(parents=True, exist_ok=True)
    dataset = load_dataset(dataset_mixer[0], split="test_prefs")
    
    dataset.to_json(out_file)


def main():
    get_general_info()
    check_item()
    logger.info("end")


if __name__ == "__main__":
    main()
