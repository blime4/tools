from contextlib import contextmanager
import torch
import io
import os
import copy

import time
import json
import hashlib
import yaml
import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import yaml
from functools import wraps

from hooktools.trace_pb2 import HookData, MetaData
from hooktools.utils import from_array, handle_config

class WrapModule(nn.Module):
    def __init__(self, orig_fn, user_forward_hook, user_backward_hook) -> None:
        super().__init__()
        if isinstance(orig_fn, str):
            orig_fn = eval(orig_fn)
        self.orig_fn = orig_fn
        self.register_forward_hook(user_forward_hook)
        self.register_backward_hook(user_backward_hook)

    def forward(self, *args, **kwargs):
        return self.orig_fn(*args, **kwargs)

    def __repr__(self):
        return str(self.orig_fn.__module__) + '.' + str(self.orig_fn.__name__)


class HackerBase(object):
    def __init__(self, config):
        config = handle_config(config)
        self.hacker_options = config.get("hacker_options", {})
        self.verbose = self.hacker_options.get("verbose", False)
        self.fw_hook = self.default_forward_hook
        self.bw_hook = self.default_backward_hook

    def default_forward_hook(self, module, input, output):
        print("[hack] [forward] [module] : ", module)
        print("[input] : ", input)
        print("[output] : ", output)

    def default_backward_hook(self, module, input, output):
        print("[hack] [backward] [module] : ", module)
        print("[input] : ", input)
        print("[output] : ", output)

    def parse_yaml(self, yaml_file):
        with open(yaml_file, 'r') as file:
            data = yaml.safe_load(file)

        apis = set()
        for key, values in data.items():
            for value in values:
                apis.add(key+"."+value)
        return list(apis)

    def detail_op(self, detail, fn):
        def wrapper(*args, **kwargs):
            if self.verbose:
                print(detail)
            fn(*args, **kwargs)
        return wrapper

    def wrap_setattr(self, apis: list, fw_fn=None, bw_fn=None):
        if not fw_fn:
            fw_fn = self.default_forward_hook
        if not bw_fn:
            bw_fn = self.default_backward_hook
        for api in apis:
            base, function = eval(".".join(api.split(".")[:-1])), api.split(".")[-1]
            fw_fn = self.detail_op(api, fw_fn)
            bw_fn = self.detail_op(api, bw_fn)
            setattr(base, function, WrapModule(api, fw_fn, bw_fn))

class Hacker(HackerBase):
    def __init__(self, config):
        super().__init__(config)
        self.supported_apis_yaml = self.hacker_options.get("supported_apis_yaml", "")

    def __call__(self):
        self.wrap_supported_apis(self.supported_apis_yaml)

    def hack(self, fw_fn=None, bw_fn=None):
        self.wrap_supported_apis(self.supported_apis_yaml, fw_fn, bw_fn)

    def wrap_supported_apis(self, yaml_file, fw_fn=None, bw_fn=None):
        supported_apis = self.parse_yaml(yaml_file)
        self.wrap_setattr(supported_apis, fw_fn, bw_fn)

    def unhack(self):
        # TODO: unhack
        pass


