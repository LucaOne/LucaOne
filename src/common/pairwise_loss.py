#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/8/14 22:24
@project: LucaOne
@file: pairwise_loss
@desc: pairwise loss for LucaOne
'''
import torch
import torch.nn as nn


class PairwiseLoss(nn.Module):
    def __init__(self, reduction='mean', ignore_nans=True, ignore_value=-100):
        super().__init__()
        self.ignore_value = ignore_value
        self.reduction = reduction
        self.ignore_nans = ignore_nans

    @classmethod
    def mse_loss(cls, targets, preds, ignore_index, reduction):
        mask = targets == ignore_index
        out = (preds[~mask]-targets[~mask])**2
        if reduction == "mean":
            return out.mean()
        elif reduction == "None":
            return out
        else:
            raise Exception("Not support reduction=%s" % reduction)

    def forward(self, preds, targets=None):
        outputs = (preds,)
        if targets is not None:
            contact_loss = self.mse_loss(
                targets=targets.type_as(preds),
                preds=preds,
                ignore_index=self.ignore_index,
                reduction=self.reduction)
            outputs = (contact_loss, ) + outputs

        return outputs



