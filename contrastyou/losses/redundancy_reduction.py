# this loss is adapted from https://arxiv.org/pdf/2103.03230.pdf
from functools import lru_cache

import torch
from torch import Tensor, nn

from contrastyou.losses import LossClass
from contrastyou.losses.discreteMI import compute_joint_2D_with_padding_zeros


class RedundancyCriterion(nn.Module, LossClass[Tensor]):

    def __init__(self, eps: float = 1e-8, symmetric: bool = True, lamda: float = 1) -> None:
        super().__init__()
        self._eps = eps
        self.symmetric = symmetric
        self.lamda = lamda

    def forward(self, x_out: Tensor, x_tf_out: Tensor):
        k = x_out.shape[1]
        p_i_j = compute_joint_2D_with_padding_zeros(x_out=x_out, x_tf_out=x_tf_out, symmetric=self.symmetric)
        p_i_j = p_i_j.view(k, k)
        self._p_i_j = p_i_j
        mask = self.onehot_label(k, device=p_i_j.device)
        diagonal_elements = p_i_j.masked_select(mask)
        off_diagonal_elements = p_i_j.masked_select(~mask)
        return self.kl_criterion(diagonal_elements, torch.tensor(1 / k, dtype=p_i_j.dtype,
                                                                 device=p_i_j.device)) * self.lamda + self.kl_criterion(
            off_diagonal_elements, torch.tensor(0, dtype=p_i_j.dtype, device=p_i_j.device))

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
