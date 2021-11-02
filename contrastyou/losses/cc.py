import math
import typing as t

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class CCLoss(nn.Module):
    """
    Local (over window) normalized cross correlation loss.
    """

    def __init__(self, win: t.Tuple[int, int], *, eps: float = 1e-5):
        super(CCLoss, self).__init__()
        self.win = win
        self.register_buffer("_sum_filt", torch.ones([1, 1, *win]))
        self.win_size = np.prod(win)
        self.eps = eps

    def __call__(self, y_true, y_pred):

        Ii = y_true
        Ji = y_pred

        # get dimension of volume
        # assumes Ii, Ji are sized [batch_size, *vol_shape, nb_feats]
        ndims = Ii.ndim - 2
        assert ndims in [1, 2, 3], "volumes should be 1 to 3 dimensions. found: %d" % ndims

        # set window size
        win = self.win

        # compute filters

        pad_no = math.floor(win[0] / 2)

        if ndims == 1:
            stride = (1,)
            padding = (pad_no,)
        elif ndims == 2:
            stride = (1, 1)
            padding = (pad_no, pad_no)
        else:
            stride = (1, 1, 1)
            padding = (pad_no, pad_no, pad_no)

        # get convolution function
        conv_fn = getattr(F, 'conv%dd' % ndims)

        # compute CC squares
        I2 = Ii * Ii
        J2 = Ji * Ji
        IJ = Ii * Ji

        I_sum = conv_fn(Ii, self._sum_filt, stride=stride, padding=padding)
        J_sum = conv_fn(Ji, self._sum_filt, stride=stride, padding=padding)
        I2_sum = conv_fn(I2, self._sum_filt, stride=stride, padding=padding)
        J2_sum = conv_fn(J2, self._sum_filt, stride=stride, padding=padding)
        IJ_sum = conv_fn(IJ, self._sum_filt, stride=stride, padding=padding)

        u_I = I_sum / self.win_size
        u_J = J_sum / self.win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * self.win_size
        cross = torch.maximum(cross, torch.zeros_like(cross).fill_(self.eps))
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * self.win_size
        I_var = torch.maximum(I_var, torch.zeros_like(I_var).fill_(self.eps))
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * self.win_size
        J_var = torch.maximum(J_var, torch.zeros_like(J_var).fill_(self.eps))
        cc = (cross * cross) / (I_var * J_var)

        return -torch.mean(cc)
