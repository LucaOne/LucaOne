#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/6/15 22:53
@project: LucaOne
@file: regression_loss
@desc: regression loss for LucaOne
'''
import warnings
import numpy as np
import torch
import torch.nn as nn
from statsmodels.stats.stattools import durbin_watson
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from masked_loss import _MaskedLoss
except ImportError:
    from src.common.masked_loss import _MaskedLoss


def nanstd(input, dim=None, keepdim=False):
    mu = torch.nanmean(input, dim=dim, keepdim=True)
    return torch.sqrt(torch.nanmean((input - mu)**2, dim=dim, keepdim=keepdim))


def iqr(batch, dim=None, reduction='mean'):
    if dim is None:
        if len(batch.shape) == 1:
            dim = 0
        else:
            dim = 1
    if isinstance(batch, np.ndarray):
        out = np.quantile(batch, 0.75, axis=dim) - \
              np.quantile(batch, 0.25, axis=dim)
    elif isinstance(batch, torch.Tensor):
        out = torch.quantile(batch, 0.75, dim=dim) - \
              torch.quantile(batch, 0.25, dim=dim)
    if reduction == 'none':
        return out
    elif reduction == 'mean':
        return out.mean()
    else:
        raise NotImplementedError


def naniqr(batch, dim=None, reduction='none'):
    if dim is None:
        if len(batch.shape) == 1:
            dim = 0
        else:
            dim = 1
    if isinstance(batch, np.ndarray):
        out = np.nanquantile(batch, 0.75, axis=dim) - \
              np.nanquantile(batch, 0.25, axis=dim)
    elif isinstance(batch, torch.Tensor):
        out = torch.nanquantile(batch, 0.75, dim=dim) - \
              torch.nanquantile(batch, 0.25, dim=dim)
    if reduction == 'none':
        return out
    elif reduction == 'mean':
        return out.mean()
    elif reduction == 'nanmean':
        return torch.nanmean(out)
    else:
        raise NotImplementedError


def compute_dw(res, dim=1, replace_missing=0., reduction='none'):
    """Durbin-Watson statistics
    https://www.statsmodels.org/devel/generated/statsmodels.stats.stattools.durbin_watson.html
    """
    if isinstance(res, torch.Tensor):
        res = res.detach().cpu().numpy()
    if replace_missing is not None:
        res = res.copy()
        res[np.isnan(res)] = replace_missing
    out = durbin_watson(res, axis=dim)
    if reduction == 'mean':
        return out.mean()
    elif reduction == 'none':
        return out
    elif reduction == 'median':
        return np.median(out)


def estimate_noise(x, dim=1, window_size=10, step=5, reduce='nanmean', keepdim=True):
    noises = nanstd(x.unfold(dim, window_size, step), -1, keepdim=False)
    if reduce == 'nanmedian':
        return noises.nanmedian(dim, keepdim=keepdim).values
    if reduce == 'nanmean':
        return noises.nanmean(dim, keepdim=keepdim)
    if reduce == 'median':
        return noises.median(dim, keepdim=keepdim).values
    if reduce == 'mean':
        return noises.mean(dim, keepdim=keepdim)
    if reduce == 'none':
        return noises
    raise ValueError


class MaskedMSELoss(_MaskedLoss):
    """Masked MSE loss"""
    def __init__(self, reduction='mean', ignore_nans=True, ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.MSELoss(reduction='none')


class MaskedL1Loss(_MaskedLoss):
    """Masked L1 loss."""

    def __init__(self, reduction='mean', ignore_nans=True, ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.L1Loss(reduction='none')


class MaskedHuberLoss(_MaskedLoss):
    """Masked L1 loss."""

    def __init__(self, reduction='mean', ignore_nans=True, delta=1, ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.HuberLoss(reduction='none', delta=delta)


class IQRLoss(nn.Module):
    "IQR of the residuals"
    def __init__(self, reduction='nanmean', ignore_nans=True, ignore_value=-100.0):
        super().__init__()
        self.reduction = reduction
        self.ignore_nans = ignore_nans
        self.ignore_value = ignore_value

    def forward(self, input, target=0.):
        if isinstance(target, torch.Tensor) and not (target.size() == input.size()):
            warnings.warn(
                "Using a target size ({}) that is different to the input size ({}). "
                "This will likely lead to incorrect results due to broadcasting. "
                "Please ensure they have the same size.".format(
                    target.size(), input.size()),
                stacklevel=2,
            )
        if self.ignore_nans:
            return naniqr(target-input, reduction=self.reduction)
        else:
            return iqr(target-input, reduction=self.reduction)


class MaskedLogCoshLoss(_MaskedLoss):
    def __init__(self, reduction='mean', ignore_nans=True,  ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = LogCoshLoss(reduction='none')


class MaskedXTanhLoss(_MaskedLoss):
    def __init__(self, reduction='mean', ignore_nans=True,  ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = XTanhLoss(reduction='none')


class MaskedXSigmoidLoss(_MaskedLoss):
    def __init__(self, reduction='mean', ignore_nans=True,  ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = XSigmoidLoss(reduction='none')


class MaskedAlgebraicLoss(_MaskedLoss):
    def __init__(self, reduction='mean', ignore_nans=True,  ignore_value=-100.0):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = AlgebraicLoss(reduction='none')


class LogCoshLoss(torch.nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        if self.reduction == 'mean':
            return torch.mean(torch.log(torch.cosh(diff + 1e-12)))
        elif self.reduction == 'sum':
            return torch.sum(torch.log(torch.cosh(diff + 1e-12)))
        else:
            return torch.log(torch.cosh(diff + 1e-12))


class XTanhLoss(torch.nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        if self.reduction == 'mean':
            return torch.mean(diff * torch.tanh(diff))
        elif self.reduction == 'sum':
            return torch.sum(diff * torch.tanh(diff))
        else:
            return diff * torch.tanh(diff)


class XSigmoidLoss(torch.nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        if self.reduction == 'mean':
            return torch.mean(2 * diff * torch.sigmoid(diff) - diff)
        elif self.reduction == 'sum':
            return torch.sum(2 * diff * torch.sigmoid(diff) - diff)
        else:
            return 2 * diff * torch.sigmoid(diff) - diff


class AlgebraicLoss(torch.nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, input, target):
        diff = input - target
        if self.reduction == 'mean':
            return torch.mean(diff * diff / torch.sqrt(1 + diff * diff))
        elif self.reduction == 'sum':
            return torch.sum(diff * diff / torch.sqrt(1 + diff * diff))
        else:
            return diff * diff / torch.sqrt(1 + diff * diff)

