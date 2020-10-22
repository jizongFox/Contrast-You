import itertools

import matplotlib.pyplot as plt
import numpy as np
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

from contrastyou.losses.iic_loss import IIDLoss

DIM = 3
NUM_SAMPLES = 400
INTER = 5
OUT = 3
CLASS_OUT = 10
NUM_SUBHEAD = 10

ENABLE_CONTRAST = False
ENABLE_IIC = True
# Create a sphere
r = 0.98
pi = np.pi
cos = np.cos
sin = np.sin
phi, theta = np.mgrid[0.0:pi:100j, 0.0:2.0 * pi:100j]
x = r * sin(phi) * cos(theta)
y = r * sin(phi) * sin(theta)
z = r * cos(phi)


def uniform_loss(embeddings, t=2):
    distance_map = embeddings.mm(embeddings.T)
    exp_dis = torch.exp(distance_map * 2 * t - 2 * t)
    mask = 1 - torch.eye(embeddings.shape[0]).to(embeddings.device)
    return ((exp_dis * mask).mean() / 2).log()


def projection(tensor):
    return nn.functional.normalize(tensor, dim=1, p=2)


class Head(nn.Module):

    def __init__(self, OUT=OUT) -> None:
        super().__init__()
        self._head1 = nn.Linear(DIM, INTER)
        self._bn1 = nn.BatchNorm1d(INTER)
        self._head2 = nn.Linear(INTER, OUT)

    def forward(self, input):
        return self._head2(F.leaky_relu(self._bn1(self._head1(input)), 0.01))


class ClassificationHead(nn.Module):

    def __init__(self, num_head=5, OUT=CLASS_OUT) -> None:
        super().__init__()
        self._heads = nn.ModuleList([nn.Sequential(
            nn.Linear(DIM, OUT),
        ) for _ in range(num_head)])

    def forward(self, input):
        return [x(input) for x in self._heads]


device = torch.device("cuda")
samples = torch.randn(NUM_SAMPLES, DIM, device=device)
samples.requires_grad = True
mask = torch.eye(NUM_SAMPLES, device=device)

head = Head()
head.to(device)
head2 = ClassificationHead(OUT=CLASS_OUT, num_head=NUM_SUBHEAD)
head2.to(device)
optimizer = torch.optim.Adam(
    itertools.chain((samples,), head.parameters(), head2.parameters()),
    lr=1e-2, weight_decay=1e-5)
iic_criterion = IIDLoss(lamb=1.0)


def average_iter(a_list):
    return sum(a_list) / float(len(a_list))


def extract_iic(projected_classes):
    loss_iic_staff = iic_criterion(projected_classes.softmax(1),
                                   (projected_classes + 1 * torch.rand_like(projected_classes)).softmax(1))
    loss_iic, p_i_j = loss_iic_staff[0], loss_iic_staff[2]
    return loss_iic, p_i_j


with tqdm(range(10000)) as indicator:
    for i in indicator:
        contrast_loss = torch.tensor(0, dtype=torch.float, device=device)
        projected_features = head(samples)
        projected_vectors = projection(projected_features)
        if ENABLE_CONTRAST:
            distance_map = projected_vectors.mm(projected_vectors.T)
            distance_map_T = distance_map / 0.07
            distance_map_T = distance_map_T - distance_map_T.max().detach()
            logits_exp = distance_map_T.exp()
            contrast_loss = (logits_exp * mask / logits_exp.sum(1, keepdim=True)).sum(1)
            contrast_loss = -torch.log(contrast_loss).mean()

        loss_iic = torch.tensor(0, dtype=torch.float, device=device)
        if ENABLE_IIC:
            # projected_feature_norm = projected_features.norm(dim=1).mean()
            # aplitude_loss = nn.MSELoss()(projected_feature_norm, torch.ones_like(projected_feature_norm))
            projected_classes = head2(samples)
            iic_losses = []
            p_i_js = []
            for p_classes in projected_classes:
                _iicloss, _pij = extract_iic(p_classes)
                iic_losses.append(_iicloss)
                p_i_js.append(_pij.detach())
            loss_iic = average_iter(iic_losses)
            p_i_j = extract_iic(projected_classes[0])[1]
        optimizer.zero_grad()
        total_loss = 0
        if ENABLE_CONTRAST:
            total_loss += contrast_loss
        if ENABLE_IIC:
            total_loss += (loss_iic * -0.5)
        total_loss.backward()
        optimizer.step()
        indicator.set_postfix({"closs": contrast_loss.item(), "iic": loss_iic.item()})
        if i % 20 == 0:
            fig = plt.figure(0)
            ax = fig.add_subplot(111, projection='3d')
            # ax.plot_surface(
            #     x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
            ax.scatter(samples.cpu().detach()[:, 0], samples.cpu().detach()[:, 1], samples.cpu().detach()[:, 2],
                       color="g", s=20)
            plt.title("original feature space")
            plt.show(block=False)
            plt.pause(0.0001)
            if ENABLE_CONTRAST:
                plt.figure(1)
                plt.clf()
                plt.imshow(distance_map.cpu().detach())
                plt.colorbar()
                plt.title("distance map generated from contrastive loss")
                plt.show(block=False)
                plt.pause(0.001)

                projected_features = head(samples).cpu().detach()
                projected_vectors = projection(projected_features)

                fig = plt.figure(2)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(
                    x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
                ax.scatter(projected_features[:, 0], projected_features[:, 1], projected_features[:, 2], color="g",
                           s=20)
                plt.title("unnormalized features from contrastive loss")
                plt.xlim([-1.5, 1.5])
                plt.ylim([-1.5, 1.5])
                plt.show(block=False)
                plt.pause(0.001)

                fig = plt.figure(3)
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(
                    x, y, z, rstride=1, cstride=1, color='c', alpha=0.6, linewidth=0)
                ax.scatter(projected_vectors[:, 0], projected_vectors[:, 1], projected_vectors[:, 2], color="r", s=20)
                plt.title("normalized vectors from contrastive loss")
                plt.xlim([-1.5, 1.5])
                plt.ylim([-1.5, 1.5])
                plt.show(block=False)
                plt.pause(0.001)
            if ENABLE_IIC:
                fig = plt.figure(4)
                plt.clf()
                plt.imshow(average_iter(p_i_js).cpu().detach())
                plt.title("p_i_j")
                plt.colorbar()
                plt.show(block=False)
                plt.pause(0.001)

            # print(samples.norm(dim=1).mean().item(), samples.norm(dim=1).std().item())
            print(f"{i} iteration with uniform loss:{uniform_loss(projected_vectors, 2)}")
