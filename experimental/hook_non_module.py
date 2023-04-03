import torch
import torch.nn as nn
import torch.nn.functional as F
import inspect
import yaml
from functools import wraps

print(torch.__version__)
class WrapModule(nn.Module):
    def __init__(self, orig_fn, user_forward_hook, user_backward_hook) -> None:
        super().__init__()
        self.orig_fn = orig_fn
        self.register_forward_hook(user_forward_hook)
        self.register_backward_hook(user_backward_hook)

    def forward(self, *args, **kwargs):
        return self.orig_fn(*args, **kwargs)

def my_forward_hook(module, features_in, features_out):
    print("print my_forward_hook:")
    print("features_in:", features_in)
    print("features_out:", features_out)

def my_backward_hook(module, features_in, features_out):
    print("print my_backward_hook:")
    print("features_in:", features_in)
    print("features_out:", features_out)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()

    def forward(self, x):
        output = torch.pow(x, 2)
        output = torch.atan(output)
        return output
class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(2, 5)

    def forward(self, x):
        output = self.fc(x)
        return output

def test_torch_op():
    op_name_torch = torch.pow
    setattr(torch, "pow", WrapModule(op_name_torch, my_forward_hook, my_backward_hook))
    model = Net()
    input = torch.tensor([2., 3], requires_grad=True)
    print("input:", input)
    output = model(input)
    output.sum().backward()

def test_module():
    op_name_functional = F.linear
    setattr(F, "linear", WrapModule(op_name_functional, my_forward_hook, my_backward_hook))
    model = Net2()
    input = torch.tensor([1., 2], requires_grad=True)
    print("input:", input)
    output = model(input)
    output.sum().backward()

def is_startwith__(msg: str, prefix: str = '__') -> bool:
    return msg.startswith(prefix)

def get_all_ops():
    torch_ops = []
    for name, obj in torch.__dict__.items():
        if callable(obj) and '__name__' in dir(obj) and not is_startwith__(name):
            torch_ops.append(name)
    return torch_ops


def categorize_functions(module_name : str, builtin_funcs : list, funcs : list, ops : list):
    module = eval(module_name)
    stack = [(module, module_name)]
    while stack:
        module, prefix = stack.pop()
        for name in dir(module):
            if is_startwith__(name): continue
            obj = getattr(module, name)
            full_name = prefix + "." + name if prefix else name
            if inspect.isbuiltin(obj):
                builtin_funcs.append(full_name)
            elif inspect.isfunction(obj):
                funcs.append(full_name)
            else:
                if full_name.count(".") > 5 : continue
                if callable(eval(full_name)):
                    ops.append(full_name)
                if inspect.ismodule(obj):
                    stack.append((obj, full_name))

TO_WRAPPER = [
    "torch.nn.functional",
]

RECURSION_ERROR_OPS = [
    'ceil',
    'ceil_',
    'isfinite',
    'masked_select',
    'masked_select_backward'
]

def remove_unnecessary_funcs(builtin_funcs : list, funcs : list, ops : list):
    # TODO : remove the no need to wrapper functions
    pass

def get_non_nn_module_function():
    builtin_funcs = []
    default_funcs = []
    other_funcs = []
    for to in TO_WRAPPER:
        categorize_functions(to, builtin_funcs, default_funcs, other_funcs)
    remove_unnecessary_funcs(builtin_funcs, default_funcs, other_funcs)
    return builtin_funcs, default_funcs, default_funcs

def get_ops_in_native_functions(yaml_file):
    with open(yaml_file, 'r') as file:
        data = yaml.safe_load(file)["ops"]
    data = [i for i in data if i not in RECURSION_ERROR_OPS]
    return data

def detail_op(detail, fn):
    def wrapper(*args, **kwargs):
        print(detail)
        fn(*args, **kwargs)
    return wrapper

def wrap_torch_ops():
    ops = get_ops_in_native_functions("ops_in_native_functions.yaml")
    for op in ops:
        if getattr(torch, op, False):
            op_name = eval("torch." + op)
            fw_fn = detail_op(op_name.__name__, my_forward_hook)
            bw_fn = detail_op(op_name.__name__, my_backward_hook)
            setattr(torch, op, WrapModule(op_name, fw_fn, bw_fn))

    print("wrap done!")

def test_wrap_torch_ops():
    wrap_torch_ops()
    model = Net()
    input = torch.tensor([2., 3], requires_grad=True)
    print("input : ", input)
    output = model(input)
    output.sum().backward()

def main():
    # test_torch_op()
    # test_module()
    # print(get_non_nn_module_function())
    # get_ops_in_native_functions("ops_in_native_functions.yaml")
    test_wrap_torch_ops()
    pass

if __name__ == '__main__':
    main()
