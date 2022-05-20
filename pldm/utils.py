from pathlib import Path
import random
import re
import os

import torch
import numpy as np
from dataclasses import fields


def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def pick_latest_model(path):
    # Finds the model in path with the largeest epoch value
    paths = list(Path(path).glob("epoch=*.ckpt"))
    rx = re.compile(".*epoch=(?P<epoch>\d+).*")
    max_epoch = 0
    max_p = None
    for p in paths:
        m = rx.match(str(p))
        epoch = int(m.group("epoch"))
        if epoch > max_epoch:
            max_epoch = epoch
            max_p = p
    return max_p


def fix_nvidia_ld_path():
    STR = "/usr/lib/nvidia"
    if STR not in os.environ.get("LD_LIBRARY_PATH", ""):
        os.environ["LD_LIBRARY_PATH"] = (
            os.environ.get("LD_LIBRARY_PATH", "") + ":" + STR
        )
        print("Fixed LD_LIBRARY_PATH to have nvidia")
        return True
    return False


def format_seconds(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


def calculate_conv_out_dim(in_dim, stride, padding, kernel_size):
    return ((in_dim - kernel_size + 2 * padding) / stride) + 1


def dict_to_namespace(d):
    """
    # Function to convert dictionary to SimpleNamespace
    """
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = dict_to_namespace(value)  # Recursively handle nested dictionaries
    return SimpleNamespace(**d)


def update_config_from_yaml(config_class, yaml_data):
    """
    Create an instance of `config_class` using default values, but override
    fields with those provided in `yaml_data`.
    """
    config_field_names = {f.name for f in fields(config_class)}

    relevant_yaml_data = {
        key: value for key, value in yaml_data.items() if key in config_field_names
    }
    return config_class(**relevant_yaml_data)
