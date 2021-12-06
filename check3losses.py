import os
import sys
import typing as t
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn, Tensor

from contrastyou.losses import LossClass
from contrastyou.losses.discreteMI import compute_joint
from contrastyou.losses.kl import KL_div
from contrastyou.mytqdm import tqdm
from contrastyou.utils import simplex, item2str, switch_plt_backend, fix_seed, class_name
from semi_seg.hooks.midl import entropy_criterion


def imsat_loss(prediction: Tensor):
    pred = prediction.moveaxis(0, 1).reshape(prediction.shape[1], -1)
    margin = pred.mean(1, keepdims=True)

    mi = -entropy_criterion(pred.t()).mean() + entropy_criterion(margin.t()).mean()

    return -mi


class IMSATLoss(nn.Module):

    def __init__(self, symmetric=False) -> None:
        super().__init__()
        self.symmetric = symmetric

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        self.x_out = x_out
        self.x_tf_out = x_tf_out
        return (imsat_loss(x_out) + imsat_loss(x_tf_out)) / 2 + KL_div()(x_out, x_tf_out)

    def get_joint(self):
        return compute_joint(self.x_out, self.x_tf_out, symmetric=False).squeeze().detach().cpu().numpy()


class IIDLoss(nn.Module, LossClass[Tensor]):
    def __init__(self, lamb: float = 1.0, eps: float = sys.float_info.epsilon, symmetric=False):
        """
        :param lamb:
        :param eps:
        """
        super().__init__()
        self.lamb = float(lamb)
        self.eps = float(eps)
        self.symmetric = symmetric

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :param x_tf_out:
        :return:
        """
        assert len(x_out.shape) == 2, x_out.shape
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out, symmetric=self.symmetric)
        assert p_i_j.size() == (k, k)
        self.p_i_j = p_i_j

        p_i = (p_i_j.sum(dim=1).view(k, 1).expand(k, k))  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric

        loss = -p_i_j * (
                torch.log(p_i_j + 1e-10) - self.lamb * torch.log(p_j + 1e-10) - self.lamb * torch.log(p_i + 1e-10)
        )
        loss = loss.sum()
        return loss

    def get_joint(self):
        return self.p_i_j.squeeze().detach().cpu().numpy()


class RRDLoss(nn.Module, LossClass[t.Tuple[Tensor, Tensor, Tensor]]):
    def __init__(self, lamb: float = 1.0, eps: float = sys.float_info.epsilon, symmetric=False, alpha: float = 1):
        """
        :param lamb:
        :param eps:
        """
        super().__init__()
        self.lamb = float(lamb)
        self._eps = float(eps)
        self.symmetric = symmetric
        self.alpha = alpha

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        """
        return the inverse of the MI. if the x_out == y_out, return the inverse of Entropy
        :param x_out:
        :param x_tf_out:
        :return:
        """
        assert len(x_out.shape) == 2, x_out.shape
        assert simplex(x_out), f"x_out not normalized."
        assert simplex(x_tf_out), f"x_tf_out not normalized."
        _, k = x_out.size()
        p_i_j = compute_joint(x_out, x_tf_out, symmetric=self.symmetric)
        assert p_i_j.size() == (k, k)

        self.p_i_j = p_i_j

        # mask = self.onehot_label(k, device=p_i_j.device)
        # diagonal_elements = p_i_j.masked_select(mask)
        # off_diagonal_elements = p_i_j.masked_select(~mask)
        p_i = (p_i_j.sum(dim=1).view(k, 1).expand(k, k))  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric
        constrained = (-p_i_j * (
                - self.lamb * torch.log(p_j + 1e-10) - self.lamb * torch.log(p_i + 1e-10)
        )).sum()
        target = (self.onehot_label(k=k, device=p_i_j.device) / k) * self.alpha + p_i_j * (1 - self.alpha)
        pseudo_loss = -(target * (p_i_j + 1e-8).log()).sum()
        return pseudo_loss + constrained

    @lru_cache()
    def onehot_label(self, k, device):
        label = torch.eye(k, device=device, dtype=torch.bool)
        return label

    def kl_criterion(self, dist: Tensor, prior: Tensor):
        return -(prior * (dist + self._eps).log() + (1 - prior) * (1 - dist + self._eps).log()).mean()

    def get_joint(self):
        return self.p_i_j.squeeze().detach().cpu().numpy()

    def set_ratio(self, alpha):
        assert 0 <= alpha <= 1
        self.alpha = alpha


def train(cur_loss, MAX_EPOCH):
    for i in range(MAX_EPOCH):
        optimizer.zero_grad()
        loss = cur_loss(input1.softmax(1), input2.softmax(1))
        loss.backward()
        optimizer.step()


@torch.no_grad()
def evaluate(out1, out2):
    iic_criterion = IIDLoss()
    imsat_criterion = IMSATLoss()
    rr_criterion = RRDLoss()

    iic_loss = iic_criterion(out1, out2).cpu().numpy()
    iic_joint = iic_criterion.get_joint()
    imsat_loss = imsat_criterion(out1, out2).cpu().numpy()
    imsat_joint = imsat_criterion.get_joint()
    rr_loss = rr_criterion(out1, out2).cpu().numpy()
    rr_joint = rr_criterion.get_joint()

    return {"iic": iic_loss,
            "imsat": imsat_loss,
            "barlow": rr_loss}, \
           {"iic": iic_joint,
            "imsat": imsat_joint,
            "barlow": rr_joint}, \
           {"agree": torch.eq(out1.max(1)[1], out2.max(1)[1]).float().mean().detach().cpu()}


def run(cur_criterion, max_round, epoch_2_evaluate):
    indicator = tqdm(range(max_round))
    indicator.set_description_str(f"{class_name(cur_criterion)}")
    record = defaultdict(lambda: list())
    record_plot = defaultdict(lambda: list())
    record_agree = defaultdict(lambda: list())
    for _ in indicator:

        train(cur_criterion, epoch_2_evaluate)
        result, joint_probs, agreement = evaluate(input1.softmax(1), input2.softmax(1))
        for k, v in result.items():
            record[k].append(v)
        for k, v in joint_probs.items():
            record_plot[k].append(v)
        for k, v in agreement.items():
            record_agree[k].append(v)
        indicator.set_postfix_str(item2str(result))
    return record, record_plot, record_agree


@switch_plt_backend()
def save_joint_plot(save_dir, plot_record):
    Path(save_dir, "image").mkdir(parents=True, exist_ok=True)
    keys = list(plot_record.keys())
    if len(keys) == 0:
        return
    for i in range(len(plot_record[keys[0]])):
        fig = plt.figure(figsize=(4, 4))

        for j, key in enumerate(keys[:1], start=1):
            plt.subplot(len(keys), 1, j)
            plt.imshow(plot_record[key][i])
            plt.axis('off')
        plt.savefig(os.path.join(save_dir, f"image/iter_{i:03d}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)


@switch_plt_backend()
def save_joint_curve(save_dir, curve_record, aggree, title: str = None, *, K: int):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    keys = list(curve_record.keys())
    if len(keys) == 0:
        return
    fig = plt.figure(figsize=(5, 3))

    for j, key in enumerate(keys, start=1):
        plt.plot(np.asarray(curve_record[key]).ravel(), label=key)
    plt.axhline(y=-np.log(K), color='r', linestyle='--')
    plt.axhline(y=-np.log(K) * 2, color='r', linestyle='--')

    plt.legend()
    plt.grid()
    if title:
        plt.title(title)
    plt.ylim([-np.log(K) * 2, 2])
    plt.savefig(os.path.join(save_dir, f"curve.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)
    fig = plt.figure()
    plt.plot(aggree["agree"], label="agreement")
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_dir, f"agreement.png"), dpi=300, bbox_inches='tight')
    plt.close(fig)


@fix_seed
def get_data(num_sample, K):
    input1 = torch.randn(num_sample, K, device="cuda", )
    input2 = torch.randn_like(input1)
    input1.requires_grad = True
    input2.requires_grad = True
    return input1, input2


if __name__ == '__main__':
    num_sample = 10000
    batches_run = 20
    K = 50
    save_dir = f"runs/sample_{num_sample}"
    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(IIDLoss(), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/iic", record_plot)
    save_joint_curve(f"{save_dir}/iic", record, record_aggrement, title="curve optimized by iic", K=K)

    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(IMSATLoss(), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/imsat", record_plot)
    save_joint_curve(f"{save_dir}/imsat", record, record_aggrement, title="curve optimized by imsat", K=K)
    #

    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(RRDLoss(alpha=0.0), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/rr/a_0.0", record_plot)
    save_joint_curve(f"{save_dir}/rr/a_0.0", record, record_aggrement, title="curve optimized by rr", K=K)

    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(RRDLoss(alpha=0.2), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/rr/a_0.2", record_plot)
    save_joint_curve(f"{save_dir}/rr/a_0.2", record, record_aggrement, title="curve optimized by rr", K=K)

    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(RRDLoss(alpha=0.4), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/rr/a_0.4", record_plot)
    save_joint_curve(f"{save_dir}/rr/a_0.4", record, record_aggrement, title="curve optimized by rr", K=K)

    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(RRDLoss(alpha=0.6), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/rr/a_0.6", record_plot)
    save_joint_curve(f"{save_dir}/rr/a_0.6", record, record_aggrement, title="curve optimized by rr", K=K)

    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(RRDLoss(alpha=0.8), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/rr/a_0.8", record_plot)
    save_joint_curve(f"{save_dir}/rr/a_0.8", record, record_aggrement, title="curve optimized by rr", K=K)

    input1, input2 = get_data(num_sample, K)
    optimizer = torch.optim.Adam((input1, input2), lr=1e-2)

    record, record_plot, record_aggrement = run(RRDLoss(alpha=1.0), 100, epoch_2_evaluate=batches_run)
    save_joint_plot(f"{save_dir}/rr/a_1.0", record_plot)
    save_joint_curve(f"{save_dir}/rr/a_1.0", record, record_aggrement, title="curve optimized by rr", K=K)
