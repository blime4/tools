
import torch
import torch
import torch.nn as nn
from functools import wraps
from collections import OrderedDict, namedtuple
import itertools
import functools
import torch.utils.hooks as hooks
from typing import Any, Callable, Optional, Dict
import transformers

from hooktools.utils import handle_config

UNSUPPORT_CLASS_TYPE = ["type", "method_descriptor"]
BASE_TYPE = [int, float, bool]

_global_hacker_backward_hooks: Dict[int, Callable] = OrderedDict()
_global_hacker_is_full_backward_hook: Optional[bool] = None
_global_hacker_forward_pre_hooks: Dict[int, Callable] = OrderedDict()
_global_hacker_forward_hooks: Dict[int, Callable] = OrderedDict()


class HackedNNModule(nn.Module):
    def __init__(self):
        super().__init__()

    def _call_impl(self, *input, **kwargs):
        # Do not call functions when jit is used
        full_backward_hooks, non_full_backward_hooks = [], []
        if len(self._backward_hooks) > 0 or len(_global_hacker_backward_hooks) > 0:
            full_backward_hooks, non_full_backward_hooks = self._get_backward_hooks()

        for hook in itertools.chain(
                _global_hacker_forward_pre_hooks.values(),
                self._forward_pre_hooks.values()):
            result = hook(self, input)
            if result is not None:
                if not isinstance(result, tuple):
                    result = (result,)
                input = result

        bw_hook = None
        if len(full_backward_hooks) > 0:
            bw_hook = hooks.BackwardHook(self, full_backward_hooks)
            input = bw_hook.setup_input_hook(input)

        if torch._C._get_tracing_state():
            result = self._slow_forward(*input, **kwargs)
        else:
            result = self.forward(*input, **kwargs)
        for hook in itertools.chain(
                _global_hacker_forward_hooks.values(),
                self._forward_hooks.values()):
            hook_result = hook(self, input, result)
            if hook_result is not None:
                result = hook_result

        if bw_hook:
            result = bw_hook.setup_output_hook(result)

        # Handle the non-full backward hooks
        if len(non_full_backward_hooks) > 0:
            var = result
            while not isinstance(var, torch.Tensor):
                if isinstance(var, dict):
                    var = next((v for v in var.values() if isinstance(v, torch.Tensor)))
                elif isinstance(var, (list, tuple)):
                    var = var[0]
                else:
                    return result
            grad_fn = var.grad_fn
            if grad_fn is not None:
                for hook in non_full_backward_hooks:
                    wrapper = functools.partial(hook, self)
                    functools.update_wrapper(wrapper, hook)
                    grad_fn.register_hook(wrapper)
                self._maybe_warn_non_full_backward_hook(input, result, grad_fn)

        return result

    __call__ : Callable[..., Any] = _call_impl

class WrapModule(HackedNNModule):
    def __init__(self, orig_fn:str, user_forward_hook, user_backward_hook) -> None:
        super().__init__()
        if not isinstance(orig_fn, str):
            raise TypeError('Expected a string for orig_fn')
        self.module_name = orig_fn
        self.orig_fn = eval(orig_fn)
        self.register_forward_hook(user_forward_hook)
        self.register_backward_hook(user_backward_hook)
        self.is_non_nn_module = True

    def forward(self, *args, **kwargs):
        try:
            return self.orig_fn(*args, **kwargs)
        except:
            try:
                new_fn = functools.partial(self.orig_fn, self)  # 使用functools.partial来包装原始函数
                return new_fn(*args, **kwargs)
            except:
                print("[error] self.orig_fn:", self.orig_fn.__name__, ", type is:", type(self.orig_fn))
                return self.orig_fn(*args, **kwargs)


    def __repr__(self):
        return self.module_name

class HackerBase(object):
    def __init__(self, config):
        config = handle_config(config)
        self.hacker_options = config.get("hacker_options", {})
        self.supported_apis = self.hacker_options.get("supported_apis", {})
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

    def detail_op(self, detail, fn):
        def wrapper(*args, **kwargs):
            if self.verbose:
                print(detail)
            fn(*args, **kwargs)
        return wrapper

    def is_unsupport_class_type(self, api):
        if isinstance(api, str):
            api = eval(api)
        if type(api).__name__ in UNSUPPORT_CLASS_TYPE:
            return True
        return False

    def wrap_setattr(self, apis: list, fw_fn=None, bw_fn=None):
        if not fw_fn:
            fw_fn = self.default_forward_hook
        if not bw_fn:
            bw_fn = self.default_backward_hook
        for api_class, api_list in apis.items():
            api_list = set(api_list)
            for api in api_list:
                if self.is_unsupport_class_type(api):
                    continue
                base, function = eval(".".join(api.split(".")[:-1])), api.split(".")[-1]
                fw_fn = self.detail_op(api, fw_fn)
                bw_fn = self.detail_op(api, bw_fn)
                setattr(base, function, WrapModule(api, fw_fn, bw_fn))

class Hacker(HackerBase):
    def __init__(self, config):
        super().__init__(config)

    def hack(self, fw_fn=None, bw_fn=None):
        self.wrap_setattr(self.supported_apis, fw_fn, bw_fn)

    def unhack(self):
        pass


# TODO :
# 1. unhack
# 2. 能否将调用到函数的文件，行号打出来。