from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np  # type: ignore

import os
import yaml
from pathlib import Path
import torch
import inspect

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

class NewHookData(object):

    def __init__(self, module, input, output, classify="", timestamp="", tag=""):
        self.module_name = str(module)
        self.classify = str(classify)
        self.timestamp = str(timestamp)
        self.tag = str(tag)
        self.input = input
        self.output = output

    def __repr__(self) -> str:
        return f"module_name    \t: {self.module_name},     \
                \nclassify      \t: {self.classify},        \
                \ninput.type    \t: {type(self.input)},     \
                \noutput.type   \t: {type(self.output)},    \
                \ntimestamp     \t: {self.timestamp},       \
                \ntag           \t: {self.tag}"

def convert_to_numpy(data):
    if type(data) in (int, float):
        return np.array(data)
    elif type(data) == torch.Tensor:
        return data.data.cpu().numpy()
    elif type(data) in (list, tuple):
        return [convert_to_numpy(d) for d in data]
    elif isinstance(data, NewHookData) :
        # TODO: check if have input output
        return [convert_to_numpy(data.input), convert_to_numpy(data.output)]
    else:
        raise TypeError(f"Unsupported data type : {type(data)}")

def calculate_absolute_error(data1, data2):
    if isinstance(data1, (list, tuple)):
        return [calculate_absolute_error(d1, d2) for d1, d2 in zip(data1, data2)]
    error = np.abs(data1 - data2)
    if isinstance(error, np.ndarray):
        return [calculate_absolute_error(d1, d2) for d1, d2 in zip(data1, data2)]
    else:
        return error