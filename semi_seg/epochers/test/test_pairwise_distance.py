import random
from contextlib import contextmanager
from unittest import TestCase

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from torch import Tensor
from tqdm import tqdm

from contrastyou.helper import pairwise_distances, plt_interactive


class TestPairwiseDistanceOptimization(TestCase):
    def setUp(self) -> None:
        super().setUp()
        dim = 3
        self._random_vectors = torch.randn(2000, dim, requires_grad=True, device="cuda")  # 3 dimensional features

        self._prototype_vectors = torch.randn(20, dim, requires_grad=True, device="cuda")  # 10 prototypes

        self._optimizer = torch.optim.Adam((self._random_vectors,), lr=5e-2)
        self._optimizer.add_param_group({"params": (self._prototype_vectors,), "lr": 5e-3, "weight_decay": 1e-4})

    def test_pairwise_loss(self, mapping_func=lambda x: x):
        distance = pairwise_distances(self._random_vectors, self._prototype_vectors)
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
        with plt_interactive():
            for i in indicator:
                self._optimizer.zero_grad()

                # cluster features
                # with self.disable_grad(self._prototype_vectors):
                distance = pairwise_distances(self._random_vectors, self._prototype_vectors.detach())
                distance = torch.exp(-distance * 1)
                loss1 = -distance.mean().log()

                # scatter prototypes
                distance_prototype = pairwise_distances(self._prototype_vectors, self._prototype_vectors)

                distance2 = torch.exp(-distance_prototype)
                loss2 = distance2.mean().log()

                # scatter features
                distance_features = pairwise_distances(self._random_vectors, self._random_vectors)

                distance3 = torch.exp(-distance_features)
                loss3 = distance3.mean().log()

                # normalized prototypes:
                # loss4 = (self._prototype_vectors.norm(p=2, dim=1) - 1).pow(2).mean()

                loss = loss1 * 2 + loss2 + loss3 * 0.5

                loss.backward()
                self._optimizer.step()
                indicator.set_postfix({"loss": loss.item()})

                if i % 10 == 0:
                    self.plot_features(self._random_vectors, self._prototype_vectors)

    def plot_features(self, features, prototypes):
        features, prototypes = features.detach().cpu(), prototypes.detach().cpu()
        fig = plt.figure(0)
        plt.clf()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(*features.transpose(1, 0), alpha=0.5, label="feature")
        ax.scatter(*prototypes.transpose(1, 0), label="prototype")
        plt.legend()
        plt.show()
        plt.pause(0.0001)

    @contextmanager
    def disable_grad(self, vector: Tensor):
        prev_flag = vector.requires_grad
        vector.requires_grad = False
        yield
        vector.requires_grad = prev_flag
