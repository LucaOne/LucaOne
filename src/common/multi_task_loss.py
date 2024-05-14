#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/10 20:38
@project: LucaOne
@file: multi_task_loss
@desc: A PyTorch implementation of Liebel L, Körner M. Auxiliary tasks in multi-task learning[J]. arXiv preprint arXiv:1805.06334, 2018.
The above paper improves the paper "Multi-task learning using uncertainty to weigh losses for scene geometry and semantics" to avoid the loss of becoming negative during training.
'''
import numpy as np
import math
import torch
import torch.nn as nn

T = 20


class AutomaticWeightedLoss(nn.Module):
    """
    automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, loss_mum):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(loss_mum, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *losses):
        loss_sum = 0
        for i, loss in enumerate(losses):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


def dynamic_weight_average(loss_t1, loss_t2, task_num):
    if loss_t1 is None or loss_t2 is None:
        return [1] * task_num
    assert len(loss_t1) == len(loss_t2)
    task_num = len(loss_t1)
    w = [l_1 / l_2 for l_1, l_2 in zip(loss_t1, loss_t2)]
    lamb = [math.exp(v / T) for v in w]
    lamb_sum = sum(lamb)
    return [task_num * l / lamb_sum for l in lamb]


def l2_normalize(x):
    return torch.sqrt(torch.sum(torch.pow(x, 2)))


def grad_norm(model, task_loss_list, initial_task_loss, alpha=0.12):
    # get layer of shared weights
    W = model.get_last_shared_layer()

    # get the gradient norms for each of the tasks
    # G^{(i)}_w(t)
    norms = []
    for i in range(len(task_loss_list)):
        # get the gradient of this task loss with respect to the shared parameters
        gygw = torch.autograd.grad(task_loss_list[i], W.parameters(), retain_graph=True)
        # compute the norm
        norms.append(torch.norm(torch.mul(model.weights[i], gygw[0])))
    norms = torch.stack(norms)
    # print('G_w(t): {}'.format(norms))
    # compute the inverse training rate r_i(t)
    # \curl{L}_i
    if torch.cuda.is_available():
        loss_ratio = task_loss_list.data.cpu().numpy() / initial_task_loss
    else:
        loss_ratio = task_loss_list.data.numpy() / initial_task_loss
    # r_i(t)
    inverse_train_rate = loss_ratio / np.mean(loss_ratio)
    # print('r_i(t): {}'.format(inverse_train_rate))
    # compute the mean norm \tilde{G}_w(t)
    if torch.cuda.is_available():
        mean_norm = np.mean(norms.data.cpu().numpy())
    else:
        mean_norm = np.mean(norms.data.numpy())
    # print('tilde G_w(t): {}'.format(mean_norm))
    # compute the GradNorm loss
    # this term has to remain constant
    constant_term = torch.tensor(mean_norm * (inverse_train_rate ** alpha), requires_grad=False)
    if torch.cuda.is_available():
        constant_term = constant_term.cuda()
    # print('Constant term: {}'.format(constant_term))
    # this is the GradNorm loss itself
    grad_norm_loss = torch.sum(torch.abs(norms - constant_term))
    # print('GradNorm loss {}'.format(grad_norm_loss))

    # compute the gradient for the weights
    model.weights.grad = torch.autograd.grad(grad_norm_loss, model.weights)[0]
