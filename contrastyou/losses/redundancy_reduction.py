# this loss is adapted from https://arxiv.org/pdf/2103.03230.pdf
from functools import lru_cache

import torch
from loguru import logger
from torch import Tensor, nn

from contrastyou.losses import LossClass
from contrastyou.losses.discreteMI import compute_joint_2D_with_padding_zeros


class RedundancyCriterion(nn.Module, LossClass[Tensor]):

    def __init__(self, eps: float = 1e-8, symmetric: bool = True, lamda: float = 1) -> None:
        super().__init__()
        self._eps = eps
        self.symmetric = symmetric
        self.lamda = lamda
        self.alpha = 1

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        k = x_out.shape[1]
        p_i_j = compute_joint_2D_with_padding_zeros(x_out=x_out, x_tf_out=x_tf_out, symmetric=self.symmetric)
        p_i_j = p_i_j.view(k, k)
        self._p_i_j = p_i_j
        target = ((self.onehot_label(k=k, device=p_i_j.device) / k) * self.alpha + p_i_j * (1 - self.alpha))
        p_i = (p_i_j.sum(dim=1).view(k, 1).expand(k, k))  # p_i should be the mean of the x_out
        p_j = p_i_j.sum(dim=0).view(1, k).expand(k, k)  # but should be same, symmetric
        constrained = (-p_i_j * (
                - self.lamda * torch.log(p_j + 1e-10) - self.lamda * torch.log(p_i + 1e-10)
        )).sum()
        pseudo_loss = -(target * (p_i_j + 1e-8).log()).sum()
        return pseudo_loss + constrained

    @lru_cache()
    def onehot_label(self, k, device):
        label = torch.eye(k, device=device, dtype=torch.bool)
        return label

    def kl_criterion(self, dist: Tensor, prior: Tensor):
        return -(prior * (dist + self._eps).log() + (1 - prior) * (1 - dist + self._eps).log()).mean()

    def get_joint_matrix(self):
        if not hasattr(self, "_p_i_j"):
            raise RuntimeError()
        return self._p_i_j.detach().cpu().numpy()

    def set_ratio(self, alpha: float):
        assert 0 <= alpha <= 1, alpha
        if self.alpha != alpha:
            logger.trace(f"Setting alpha = {alpha}")
        self.alpha = alpha


if __name__ == '__main__':
    x_output = torch.rand(1, 5, 224, 224, requires_grad=True, device="cuda").softmax(1)
    x_output_tf = torch.rand(1, 5, 224, 224, requires_grad=True, device="cuda").softmax(1)
    torch.backends.cudnn.benchmark = True  # noqa

    """
    optimizer = torch.optim.Adam((x_output, x_output_tf), lr=1e-2)
    indicator = tqdm(range(100000))
    for i in indicator:
        loss = RedundancyCriterion(lamda=0.01)(x_output.softmax(1), x_output_tf.softmax(1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        max_values = x_output.softmax(1).max(1)[0].mean().item()
        max_values2 = x_output_tf.softmax(1).max(1)[0].mean().item()
        dict_ = {"loss": loss.item(), "marginal1": item2str(x_output.softmax(1).mean(dim=[0, 2, 3])),
                 "marginal2": item2str(x_output_tf.softmax(1).mean(dim=[0, 2, 3])), "max_1": max_values,
                 "max_2": max_values2, }
        indicator.set_postfix(dict_)
    """
