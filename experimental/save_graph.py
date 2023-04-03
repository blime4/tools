import torch
from torch.nn import Module
class Graph():
    def __init__(self):
        self.graph = {}
    def add_edge(self, u, v):
        if u not in self.graph:
            self.graph[u] = []
        self.graph[u].append(v)
    def __str__(self):
        return str(self.graph)

class SaveGraph():
    def __init__(self, model: Module):
        self.model = model
        self.graph = Graph()
        self.handles = []
    def register_hooks(self, module: Module):
        handle = module.register_forward_hook(self.forward_hook)
        self.handles.append(handle)
    def forward_hook(self, module: Module, inputs, outputs):
        input_names = []
        for input in inputs:
            if hasattr(input, 'grad_fn'):
                input_names.append(input.grad_fn.__class__.__name__)
            else:
                input_names.append(type(input).__name__)
        self.graph.add_edge(module.__class__.__name__, input_names)
    def visualize(self):
        print(self.graph)
    def close(self):
        for handle in self.handles:
            handle.remove()

model = torch.nn.Sequential(
    torch.nn.Linear(10, 5),
    torch.nn.ReLU(),
    torch.nn.Linear(5, 1)
)

save_graph = SaveGraph(model)

for module in model.modules():
    save_graph.register_hooks(module)

x = torch.randn(1, 10)
y = model(x)
save_graph.visualize()
save_graph.close()