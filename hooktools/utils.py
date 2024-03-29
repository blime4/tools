from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import yaml
from pathlib import Path
import time
import torch.nn as nn
from collections import defaultdict

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


def get_classify(module):
    if hasattr(module, "is_non_nn_module"):
        return "non nn.module"
    else:
        return "nn.module"

def is_need_to_filter_specifiy_modules(module_name, specifiy_modules_options):
    if not specifiy_modules_options or len(specifiy_modules_options) == 0:
        return True
    if "starts_with" in specifiy_modules_options:
        if any(str(module_name).startswith(specifiy_module) for specifiy_module in specifiy_modules_options["starts_with"]):
            return True
    if "regular_expression" in specifiy_modules_options:
        import re
        if any(re.match(rf"{pattern}", str(module_name)) for pattern in specifiy_modules_options["regular_expression"]):
            return True
    return False

class NewHookData(nn.Module):

    def __init__(self, module,
                 input=None, input_grad=None,
                 output=None, output_grad=None,
                 gradient=None, gradient_grad=None,
                 pt_file=None,
                 tag="", formal=False):
        super(NewHookData, self).__init__()
        self.module_name = str(module)
        self.formal = formal
        self.pt_file = pt_file
        self.have_children = len(list(module.children()))>0
        self.ast_tree = None
        if self.have_children:
            self.ast_tree = self.get_sublayers(module)

        if input is not None:
            self.input = input
        if input_grad is not None:
            self.input_grad = input_grad
        if output is not None:
            self.output = output
        if output_grad is not None:
            self.output_grad = output_grad
        if gradient is not None:
            self.gradient = gradient
        if gradient_grad is not None:
            self.gradient_grad = gradient_grad

        if self.formal:
            self.classify = get_classify(module) if gradient is None else "gradient"
            self.timestamp = str(int(time.time()))
            self.tag = str(tag)

    def __repr__(self) -> str:
        r = f"module_name    \t: {self.module_name}"
        r += f",\n pt_file   \t: {self.pt_file}"
        if hasattr(self, 'input'):
            r += f",\n input.type    \t: {type(self.input)}"
        if hasattr(self, 'output'):
            r += f",\n output.type   \t: {type(self.output)}"
        if hasattr(self, 'gradient'):
            r += f",\n gradient.type \t: {type(self.gradient)}"

        if hasattr(self, 'input_grad'):
            r += f",\n input_grad.type    \t: {type(self.input_grad)}"
        if hasattr(self, 'output_grad'):
            r += f",\n output_grad.type   \t: {type(self.output_grad)}"
        if hasattr(self, 'gradient_grad'):
            r += f",\n gradient_grad.type \t: {type(self.gradient_grad)}"

        if self.formal:
            r += f",\n classify      \t: {self.classify}"
            r += f",\n timestamp     \t: {self.timestamp}"
            r += f",\n tag           \t: {self.tag}"
        return r

    def get_sublayers(self, layer):
        sublayers = defaultdict(list)
        for name, module in layer.named_children():
            sublayers[name].append(str(module))
            if len(list(module.children())) > 0:
                sublayers[name].append(self. get_sublayers(module))
        return sublayers