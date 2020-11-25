import argparse
from itertools import cycle

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from sklearn.cluster import KMeans
from torch import nn

# variational prototype network for one shot classification.
from deepclustering2.tqdm import tqdm


def get_args():
    parser = argparse.ArgumentParser(  # noqa
        description="prototype demo",  # noqa
        formatter_class=argparse.ArgumentDefaultsHelpFormatter  # noqa
    )  # noqa
    parser.add_argument("--num_samples", default=100, help="num_samples")
    parser.add_argument("--device", default="cuda", help="training device")
    args = parser.parse_args()
    return args


def create_projector_head(input_dim, output_dim, nonlinearity="relu"):
    return nn.Sequential(nn.Linear(input_dim, output_dim))


def get_trainable_features(num_samples, z_dim, device="cuda"):
    return torch.randn(num_samples, z_dim, device=device)


def get_optimizer(features, lr, weight_decay):
    return torch.optim.Adam((features,), lr=lr, weight_decay=weight_decay)


def get_clusters(data, num_clusters=20):
    model = KMeans(n_clusters=num_clusters)
    model.fit(data.detach().cpu())
    return model.cluster_centers_, model.labels_


def plot(data, title=None, figure_num=0):
    fig = plt.figure(figure_num)
    ax = fig.add_subplot(111, projection='3d')
    # ax.plot_surface(
    #     x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    ax.scatter(data[:, 0], data[:, 1], data[:, 2], color="g", s=20)
    if title:
        plt.title(title)
    return plt.gcf()


def plot_prototype(data, centers, labels, figure_num=1):
    colors = cycle(["b", "g", "r", "c", "m", "y", "k"])
    if torch.is_tensor(data):
        data = data.detach().cpu().numpy()

    fig = plt.figure(figure_num)
    ax = fig.add_subplot(111, projection='3d')
    for i in sorted(np.unique(labels)):
        sub_data = data[labels == i]
        ax.scatter(sub_data[:, 0], sub_data[:, 1], sub_data[:, 2], color=next(colors), s=20)
    return fig


def train(data, optimizer, criterion, device, ):
    with tqdm(range(10000)) as indicator:
        for i in indicator:
            cluster_centers, cluster_labels = get_clusters(data, num_clusters=20)
            fig = plot(data, )
            plot_prototype(data, cluster_centers, cluster_labels)
            plt.show(block=False)
            plt.pause(0.001)

            # p_x_given_c =


if __name__ == '__main__':
    args = get_args()
    data = get_trainable_features(500, 3, device=args.device)
    optimizer = get_optimizer(data, lr=1e-2, weight_decay=1e-4)
    train(data, optimizer, None, device=args.device)

# device = torch.device("cuda")
# samples = torch.randn(NUM_SAMPLES, DIM, device=device)
# samples.requires_grad = True
# mask = torch.eye(NUM_SAMPLES, device=device)
