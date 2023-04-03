import torch
import torch.nn as nn
import torch.nn.functional as F
print(torch.__version__)
# 类似的torch的方法，也可以用于tensor 和 functional
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

def main():
    test_torch_op()
    test_module()

if __name__ == '__main__':
    main()
