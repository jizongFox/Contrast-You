import random
from contextlib import contextmanager
from itertools import chain
from unittest import TestCase

import matplotlib.pyplot as plt
import torch
from torch import Tensor
from tqdm import tqdm


class TestPairwiseDistianceOptimization(TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._random_vectors = torch.randn(1000, 2, requires_grad=True, device="cuda")  # 3 dimensional features

        self._prototype_vectors = torch.randn(10, 2, requires_grad=True, device="cuda")  # 10 prototypes

        self._optimizer = torch.optim.Adam(chain((self._random_vectors,), (self._prototype_vectors,)), lr=1e-3)

    def test_pairwise_loss(self, mapping_func=lambda x: x):
        distance = self._distance_fun(self._random_vectors, self._prototype_vectors)
        distance = mapping_func(distance)
        for _ in range(100):
            i, j = random.randint(0, int(len(self._random_vectors))) - 1, \
                   random.randint(0, int(len(self._prototype_vectors)) - 1)
            elementi, elementj = self._random_vectors[i], self._prototype_vectors[j]
            distance2 = ((elementi - elementj) ** 2).sum()
            assert torch.allclose(distance[i][j], mapping_func(distance2)), (distance[i][j], mapping_func(distance2))

    def test_exponental_pairwise_loss(self):
        self.test_pairwise_loss(mapping_func=lambda x: torch.exp(-x))

    def test_visualizing_clustering(self):
        indicator = tqdm(range(10000))
        plt.ion()
        for i in indicator:
            self._optimizer.zero_grad()
            with self.disable_grad(self._prototype_vectors):
                distance = self._distance_fun(self._random_vectors, self._prototype_vectors)
                distance = torch.exp(-distance)
                loss1 = -distance.mean()

            distance_prototype = self._distance_fun(self._prototype_vectors, self._prototype_vectors)
            distance2 = torch.exp(-distance_prototype)
            loss2 = distance2.mean()

            loss = loss1 + loss2
            loss.backward()
            self._optimizer.step()
            indicator.set_postfix({"loss": loss.item()})
            if i % 10 == 0:
                self.plot_features(self._random_vectors, self._prototype_vectors)
        plt.ioff()

    def plot_features(self, features, prototypes):
        features, prototypes = features.detach().cpu(), prototypes.detach().cpu()
        plt.clf()
        plt.scatter(features[:, 0], features[:, 1])
        plt.scatter(prototypes[:, 0], prototypes[:, 1], c="r")
        plt.show()
        plt.pause(0.0001)

    @contextmanager
    def disable_grad(self, vector:Tensor):
        prev_flag = vector.requires_grad
        vector.requires_grad=False
        yield
        vector.requires_grad= prev_flag

    @staticmethod
    def _distance_fun(x, y):
        """
        ∥Ai−Bj∥22=⟨Ai−Bj,Ai−Bj⟩=∥Ai∥22+∥Bj∥22−2⟨Ai,Bj⟩
        :param x:
        :param y:
        :return:
        """
        assert x.shape[1] == y.shape[1], (x.shape, y.shape)
        x, y = x.unsqueeze(1), y.unsqueeze(1)
        x_norm = (x ** 2).sum(2).view(x.shape[0], 1)
        y_norm = (y ** 2).sum(2).view(y.shape[0], 1).transpose(1, 0)
        dist = x_norm + y_norm - 2.0 * torch.mm(x.squeeze(), y.squeeze().transpose(1, 0))
        # dist[dist != dist] = 0  # replace nan values with 0
        return dist
