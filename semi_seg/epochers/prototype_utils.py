import torch
from torch import nn
from torch.nn import functional as F


def pairwise_distances(x, y=None):
    '''
    Input: x is a Nxd matrix
           y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
            if y is not given then use 'y=x'.
    i.e. dist[i,j] = ||x[i,:]-y[j,:]||^2
    '''
    x_norm = (x ** 2).sum(1).view(-1, 1)
    if y is not None:
        y_norm = (y ** 2).sum(1).view(1, -1)
    else:
        y = x
        y_norm = x_norm.view(1, -1)

    dist = x_norm + y_norm - 2.0 * torch.mm(x, torch.transpose(y, 0, 1))
    return dist


class PrototypeProjector(nn.Module):
    def __init__(self, num_clusters, input_dim, z_dim):
        super().__init__()
        self._num_clusters = num_clusters
        self._input_dim = input_dim
        self._z_dim = z_dim
        self._prototype_vector = nn.Parameter(
            F.normalize(torch.Tensor(num_clusters, z_dim), dim=1),
            requires_grad=True
        )
        self._projector = nn.Sequential(
            nn.Linear(self._input_dim, self._z_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(self._z_dim, self._z_dim)
        )

    def forward(self, tensor1, tensor2):
        tensor1 = self._projector(tensor1.squeeze())
        tensor2 = self._projector(tensor2.squeeze())
        tensor1 = F.normalize(tensor1, dim=1)
        tensor2 = F.normalize(tensor2, dim=1)
        dist1 = tensor1 - self._prototype_vector


