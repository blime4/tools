import torch
import torch.nn as nn
import os
import hooktools
from hooktools import Tracer

class BatchNorm1dNet(nn.Module):
    def __init__(self):
        super(BatchNorm1dNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        self.bn1 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        return x


def test_batchnorm():
    net = BatchNorm1dNet()
    x = torch.randn((32, 10))
    output = net(x)
    print(output)

config_path = os.path.join(hooktools.__path__[0], "config")

if __name__ == "__main__":
    trace = Tracer(os.path.join(config_path, "tracer_demo.yaml"))
    trace.trace()
    test_batchnorm()
    trace.untrace()