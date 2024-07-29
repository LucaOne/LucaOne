#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/3 20:35
@project: LucaOne
@file: classification_loss
@desc: classification loss for LucaOne
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from masked_loss import _MaskedLoss
except ImportError:
    from src.common.masked_loss import _MaskedLoss


class MaskedFocalLoss(_MaskedLoss):
    """Masked FocalLoss"""
    def __init__(self, alpha=1, gamma=2, normalization=False, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = FocalLoss(alpha=alpha, gamma=gamma, normalization=normalization, reduction='none')


class FocalLoss(nn.Module):
    '''
    Focal loss
    '''
    def __init__(self, alpha=1, gamma=2, normalization=False, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.normalization = normalization
        self.reduction = reduction

    def forward(self, inputs, targets):
        if self.normalization:
            '''
             reduction: the operation on the output loss, which can be set to 'none', 'mean', and 'sum'; 
            'none' will not perform any processing on the loss, 
            'mean' will calculate the mean of the loss, 
            'sum' will sum the loss, and the default is 'mean'
            '''
            bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
            probs = torch.sigmoid(inputs)
        else:
            bce = F.binary_cross_entropy(inputs, targets, reduction='none')
            probs = inputs
        pt = targets * probs + (1 - targets) * (1 - probs)
        modulate = 1 if self.gamma is None else (1 - pt) ** self.gamma

        focal_loss = modulate * bce

        if self.alpha is not None:
            assert 0 <= self.alpha <= 1
            alpha_weights = targets * self.alpha + (1 - targets) * (1 - self.alpha)
            focal_loss *= alpha_weights
        if self.reduction == "mean":
            # global mean
            return torch.mean(focal_loss)
        if self.reduction in ["summean", "meansum"]:
            # sum of all samples and calc the mean value
            return torch.mean(torch.sum(focal_loss, dim=1))
        elif self.reduction == "sum":
            return torch.sum(focal_loss, dim=1)
        else:
            return focal_loss


class MaskedMultiLabelCCE(_MaskedLoss):
    """Masked MultiLabel CCE"""
    def __init__(self, normalization=False, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = MultiLabelCCE(normalization=normalization, reduction='none')


class MultiLabelCCE(nn.Module):
    '''
    Multi Label CCE
    '''
    def __init__(self, normalization=False, reduction='mean'):
        super(MultiLabelCCE, self).__init__()
        self.normalization = normalization
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Cross entropy of multi-label classification
        Note：The shapes of y_true and y_pred are consistent, and the elements of y_true are either 0 or 1. 1 indicates
        that the corresponding class is a target class, and 0 indicates that the corresponding class is a non-target class.
        """
        if self.normalization:
            y_pred = torch.softmax(inputs, dim=-1)
        else:
            y_pred = inputs
        y_true = targets
        y_pred = (1 - 2 * y_true) * y_pred
        y_pred_neg = y_pred - y_true * 1e12
        y_pred_pos = y_pred - (1 - y_true) * 1e12
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat((y_pred_neg, zeros), axis=-1)
        y_pred_pos = torch.cat((y_pred_pos, zeros), axis=-1)
        neg_loss = torch.logsumexp(y_pred_neg,  axis=-1)
        pos_loss = torch.logsumexp(y_pred_pos,  axis=-1)
        if self.reduction == 'mean':
            return torch.mean(neg_loss + pos_loss)
        elif self.reduction == 'sum':
            return torch.sum(neg_loss + pos_loss)
        else:
            return neg_loss + pos_loss


class MaskedAsymmetricLoss(_MaskedLoss):
    """Masked AsymmetricLoss"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = AsymmetricLoss(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)


class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=True):
        super(AsymmetricLoss, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()


class MaskedAsymmetricLossOptimized(_MaskedLoss):
    """Masked ASLSingleLabel loss"""
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = AsymmetricLossOptimized(gamma_neg, gamma_pos, clip, eps, disable_torch_grad_focal_loss)


class AsymmetricLossOptimized(nn.Module):
    '''
    Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations
    '''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super(AsymmetricLossOptimized, self).__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()


class MaskedASLSingleLabel(_MaskedLoss):
    """Masked ASLSingleLabel loss"""
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = ASLSingleLabel(gamma_pos, gamma_neg, eps, reduction='none')


class ASLSingleLabel(nn.Module):
    '''
    This loss is intended for single-label classification problems（multi-class）
    '''
    def __init__(self, gamma_pos=0, gamma_neg=4, eps: float = 0.1, reduction='mean'):
        super(ASLSingleLabel, self).__init__()

        self.eps = eps
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.targets_classes = []
        self.gamma_pos = gamma_pos
        self.gamma_neg = gamma_neg
        self.reduction = reduction

    def forward(self, inputs, target):
        '''
        "input" dimensions: - (batch_size, number_classes)
        "target" dimensions: - (batch_size)
        '''
        num_classes = inputs.size()[-1]
        log_preds = self.logsoftmax(inputs)
        self.targets_classes = torch.zeros_like(inputs).scatter_(1, target.long().unsqueeze(1), 1)

        # ASL weights
        targets = self.targets_classes
        anti_targets = 1 - targets
        xs_pos = torch.exp(log_preds)
        xs_neg = 1 - xs_pos
        xs_pos = xs_pos * targets
        xs_neg = xs_neg * anti_targets
        asymmetric_w = torch.pow(1 - xs_pos - xs_neg, self.gamma_pos * targets + self.gamma_neg * anti_targets)
        log_preds = log_preds * asymmetric_w

        if self.eps > 0:
            # label smoothing
            self.targets_classes = self.targets_classes.mul(1 - self.eps).add(self.eps / num_classes)

        # loss calculation
        loss = - self.targets_classes.mul(log_preds)

        loss = loss.sum(dim=-1)
        if self.reduction == 'mean':
            loss = loss.mean()

        return loss


class MaskedBCEWithLogitsLoss(_MaskedLoss):
    """Masked BCE loss"""
    def __init__(self, pos_weight=None, weight=None, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight, weight=weight, reduction='none')


class MaskedCrossEntropyLoss(_MaskedLoss):
    """Masked CCE loss"""
    def __init__(self, weight=None, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__(reduction=reduction, ignore_nans=ignore_nans, ignore_value=ignore_value)
        self.criterion = nn.CrossEntropyLoss(weight=weight, reduction='none', ignore_index=ignore_value)

