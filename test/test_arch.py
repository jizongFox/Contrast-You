import torch
from torch.nn import Module, Linear


class A(Module):
    def __init__(self, a=1) -> None:
        super().__init__()
        self.a = Linear(1, 1)
        self.register_buffer("b", torch.Tensor([0]))
        self.b = torch.Tensor([0])
        self.d = a


A1 = A(1)
A2 = A(2)
A1.a
b = A1.a
c = A2.b
d = A1.d
d = A1.d
