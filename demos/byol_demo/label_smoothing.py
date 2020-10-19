from itertools import chain

import torch
from torch import Tensor
from torch import nn
from torch.optim import Adam

from deepclustering2.utils import simplex, class2one_hot


class Mapping(nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.head = nn.Linear(3, 3)

    def forward(self, input):
        return self.head(input)


input = torch.randn(100, 3)
target = torch.randint(3, (100,))
net = Mapping()
optimizer = Adam(chain(net.parameters(), input), lr=1e-4)


def soft_convert(target, max_value=0.9, C=10):
    assert 1 / C <= max_value <= 1, (C, max_value)
    onehot_target = class2one_hot(target, C=C).float()
    if max_value == 1:
        return onehot_target
    min_value = (1 - max_value) / (C - 1)
    assert min_value <= max_value
    onehot_target[onehot_target == 1] = max_value
    onehot_target[onehot_target == 0] = min_value
    assert simplex(onehot_target)
    return onehot_target


def soft_ce_loss(probs: Tensor, targets: Tensor, reduction="mean"):
    b, c, *_ = probs.shape
    assert probs.shape == targets.shape, (probs.shape, targets.shape)
    assert simplex(probs) and simplex(targets)
    loss = ((probs + 1e-10).log() * targets).sum(1)
    if reduction == "mean":
        return -loss.mean()
    if reduction == "sum":
        return -loss.sum()
    return -loss


soft_target = soft_convert(target, C=3, max_value=1)
for i in range(100000):
    logits = net(input)
    loss = soft_ce_loss(logits.softmax(1), soft_target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print(torch.eq(logits.max(1)[1], target).float().mean().item(), loss.item())
