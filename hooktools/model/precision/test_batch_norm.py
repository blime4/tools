import torch
import torch.nn as nn
import torch.optim as optim
import os
import hooktools
from hooktools import Tracer

class BatchNorm1dNet(nn.Module):
    def __init__(self):
        super(BatchNorm1dNet, self).__init__()
        self.fc1 = nn.Linear(10, 20)
        # self.bn1 = nn.BatchNorm1d(20)

    def forward(self, x):
        x = self.fc1(x)
        # x = self.bn1(x)
        return x

def test_batchnorm1dnet():
    net = BatchNorm1dNet()

    input = torch.randn(32, 10)
    target = torch.randn(32, 20)

    optimizer = optim.SGD(net.parameters(), lr=0.01)
    criterion = nn.MSELoss()

    for i in range(10):
        optimizer.zero_grad()
        output = net(input)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    input.requires_grad_()
    output = net(input)
    grad_output = torch.randn(output.size())
    output.backward(grad_output)

config_path = os.path.join(hooktools.__path__[0], "config")

if __name__ == "__main__":
    trace = Tracer(os.path.join(config_path, "tracer_demo.yaml"))
    trace.trace()
    test_batchnorm1dnet()
    trace.untrace()