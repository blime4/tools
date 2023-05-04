from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import os
import yaml
from pathlib import Path

def check_suffix(file="demo.yaml", suffix=('.yaml,'), msg=''):
    # Check file(s) for acceptable suffix
    if file and suffix:
        if isinstance(suffix, str):
            suffix = [suffix]
        for f in file if isinstance(file, (list, tuple)) else [file]:
            s = Path(f).suffix.lower()  # file suffix
            if len(s):
                assert s in suffix, f"{msg}{f} acceptable suffix is {suffix}"

def handle_config(config):
    if isinstance(config, str):
        check_suffix(config, suffix=('.yaml', '.yml'))
        def join_path(loader, node):
            file_path = os.path.dirname(loader.name)
            path = loader.construct_scalar(node)
            return os.path.join(file_path, path)

        yaml.add_constructor('!join_path', join_path)
        with open(config, 'r', encoding='utf-8') as file:
            file_data = file.read()
            config = yaml.safe_load(file_data)

    assert isinstance(config, dict), f"unacceptable cofiguration! "
    return config

def ordered(obj):
    if isinstance(obj, dict):
        return sorted((k, ordered(v)) for k, v in obj.items())
    if isinstance(obj, list):
        return sorted(ordered(x) for x in obj)
    else:
        return obj

def get_file_list(path, endswith="pt"):
    return [os.path.join(path, pb) for pb in sorted(os.listdir(path)) if pb.endswith(endswith)]


def is_non_nn_module(module):
    if hasattr(module, "is_non_nn_module"):
        return True
    return False