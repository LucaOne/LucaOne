#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/3 20:35
@project: LucaOne
@file: loss
@desc: loss for LucaOne
'''
import torch, math
import torch.nn as nn
import sys
sys.path.append(".")
sys.path.append("..")
sys.path.append("../..")
sys.path.append("../../src")
try:
    from classification_loss import *
    from regression_loss import *
except ImportError:
    from src.common.classification_loss import *
    from src.common.regression_loss import *


class NewGELUActivation(nn.Module):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return 0.5 * input * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (input + 0.044715 * torch.pow(input, 3.0))))


def create_activate(activate_func):
    if activate_func:
        activate_func = activate_func.lower()
    if activate_func == "tanh":
        return nn.Tanh()
    elif activate_func == "relu":
        return nn.ReLU()
    elif activate_func == "leakyrelu":
        return nn.LeakyReLU()
    elif activate_func == "gelu":
        return nn.GELU()
    elif activate_func == "gelu_new":
        return NewGELUActivation()
    else:
        return nn.Tanh()


def create_loss_function(config,
                         args,
                         task_level_type,
                         task_level_name,
                         sigmoid,
                         output_mode,
                         num_labels,
                         loss_type,
                         ignore_index=-100,
                         pair_level=False,
                         return_types=["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
                         ):
    '''
    create the output layer and loss layer
    :param task_level_name:
    :param task_level_type:
    :param pair_level:
    :param config:
    :param args:
    :param sigmoid:
    :param output_mode:
    :param num_labels:
    :param loss_type:
    :param ignore_index:
    :param return_types:
    :return:
    '''
    '''
    print("task_level_type:%s, task_level_name:%s" %(task_level_type, task_level_name),
          ",sigmoid=", sigmoid, ",output_mode=", output_mode,
          ",num_labels=", num_labels, ",loss_type=", loss_type)
    '''
    dropout, hidden_layer, hidden_act, classifier, output, loss_fct = None, None, None, None, None, None
    if "dropout" in return_types:
        if hasattr(config, "classifier_dropout_prob"):
            dropout = nn.Dropout(config.classifier_dropout_prob)
        elif hasattr(config, "dropout_prob"):
            dropout = nn.Dropout(config.dropout_prob)
        else:
            dropout = nn.Dropout(0.1)

    if pair_level:
        hidden_size = 2 * config.hidden_size
    else:
        hidden_size = config.hidden_size
    if "hidden_layer" in return_types:
        hidden_layer_size = args.classifier_size[task_level_type][task_level_name]
        hidden_layer = nn.Linear(hidden_size, hidden_layer_size, bias=True)
        hidden_size = hidden_layer_size

    if "hidden_act" in return_types:
        if hasattr(args, "classifier_hidden_act"):
            hidden_act = create_activate(args.classifier_hidden_act)
        elif hasattr(config, "classifier_hidden_act"):
            hidden_act = create_activate(config.classifier_hidden_act)

    if "classifier" in return_types:
        if sigmoid:
            if output_mode in ["binary_class", "binary-class"]:
                classifier = nn.Linear(hidden_size, 1, bias=True)
            else:
                classifier = nn.Linear(hidden_size, num_labels, bias=True)
        else:
            classifier = nn.Linear(hidden_size, num_labels, bias=True)
    if "output" in return_types:
        if sigmoid or output_mode in ["multi_label", "multi-label", "binary_class", "binary-class"]:
            output = nn.Sigmoid()
        elif output_mode in ["multi_class", "multi-class"]:
            output = nn.Softmax(dim=-1)
        else:
            output = None

    if "loss" in return_types:
        # positive weight
        if hasattr(args, "pos_weight") and args.pos_weight:
            pos_weight = args.pos_weight
        elif hasattr(config, "pos_weight") and config.pos_weight:
            pos_weight = config.pos_weight
        else:
            pos_weight = None

        if hasattr(args, "weight") and args.weight is not None:
            weight = args.weight
        elif hasattr(config, "weight") and config.weight is not None:
            weight = config.weight
        else:
            weight = None

        reduction = config.loss_reduction if hasattr(config, "loss_reduction") else "meanmean"
        if output_mode in ["regression"]:
            if loss_type == "l2":
                loss_fct = MaskedMSELoss(reduction=reduction, ignore_nans=True,
                                         ignore_value=ignore_index * 1.0 if ignore_index else None)
            elif loss_type == "l1":
                loss_fct = MaskedL1Loss(reduction=reduction, ignore_nans=True,
                                        ignore_value=ignore_index * 1.0 if ignore_index else None)
        elif output_mode in ["multi_label", "multi-label"]:
            if loss_type == "bce":
                if pos_weight:
                    if isinstance(pos_weight, str) or isinstance(pos_weight, int):
                        pos_weight = [float(pos_weight)] * num_labels
                    elif isinstance(pos_weight, float):
                        pos_weight = [pos_weight] * num_labels
                    pos_weight = torch.tensor(pos_weight, dtype=torch.float32).to(args.device)
                    print("multi_label pos_weight:")
                    print(pos_weight)
                    assert pos_weight.ndim == 1 and pos_weight.shape[0] == num_labels
                    print("multi_label reduction:")
                    print(reduction)
                    loss_fct = MaskedBCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction,
                                                       ignore_nans=True, ignore_value=ignore_index)
                else:
                    loss_fct = MaskedBCEWithLogitsLoss(reduction=reduction,
                                                       ignore_nans=True, ignore_value=ignore_index)
            elif loss_type == "asl":
                loss_fct = MaskedAsymmetricLossOptimized(gamma_neg=args.asl_gamma_neg if hasattr(args, "asl_gamma_neg") else 4.0,
                                                         gamma_pos=args.asl_gamma_pos if hasattr(args, "asl_gamma_pos") else 1.0,
                                                         clip=args.clip if hasattr(args, "clip") else 0.05,
                                                         eps=args.eps if hasattr(args, "eps") else 1e-8,
                                                         disable_torch_grad_focal_loss=args.disable_torch_grad_focal_loss if hasattr(args, "disable_torch_grad_focal_loss") else False,
                                                         reduction=reduction,
                                                         ignore_nans=True,
                                                         ignore_value=ignore_index)
            elif loss_type == "focal_loss":
                loss_fct = MaskedFocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 0.7,
                                           gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 2.0,
                                           normalization=True,
                                           reduction=reduction,
                                           ignore_nans=True,
                                           ignore_value=ignore_index)
            elif loss_type == "multilabel_cce":
                loss_fct = MaskedMultiLabelCCE(normalization=True,
                                               reduction=reduction,
                                               ignore_nans=True,
                                               ignore_value=ignore_index)
        elif output_mode in ["binary_class", "binary-class"]:
            if loss_type == "bce":
                if pos_weight:
                    if isinstance(pos_weight, str) or isinstance(pos_weight, int):
                        pos_weight = torch.tensor([float(pos_weight)], dtype=torch.long).to(args.device)
                    elif isinstance(pos_weight, float):
                        pos_weight = torch.tensor([pos_weight], dtype=torch.float32).to(args.device)
                    print("binary_class pos_weight:")
                    print(pos_weight)
                    assert pos_weight.ndim == 1 and pos_weight.shape[0] == 1
                    loss_fct = MaskedBCEWithLogitsLoss(pos_weight=pos_weight, reduction=reduction, ignore_nans=True,
                                                       ignore_value=ignore_index)
                else:
                    loss_fct = MaskedBCEWithLogitsLoss(reduction=reduction, ignore_nans=True, ignore_value=ignore_index)
            elif loss_type == "focal_loss":
                loss_fct = MaskedFocalLoss(alpha=args.focal_loss_alpha if hasattr(args, "focal_loss_alpha") else 0.7,
                                           gamma=args.focal_loss_gamma if hasattr(args, "focal_loss_gamma") else 2.0,
                                           normalization=True,
                                           reduction=reduction,
                                           ignore_nans=True,
                                           ignore_value=ignore_index)
        elif output_mode in ["multi_class", "multi-class"]:
            if weight:
                # [1, 1, 1, ,1, 1...] length: num_labels
                if isinstance(weight, str) or isinstance(weight, int):
                    weight = [float(weight)] * num_labels
                elif isinstance(weight, float):
                    weight = [weight] * num_labels
                weight = torch.tensor(weight, dtype=torch.float32).to(args.device)
                print("multi_class weight:")
                print(weight)
                assert weight.ndim == 1 and weight.shape[0] == num_labels
                if ignore_index is None:
                    loss_fct = nn.CrossEntropyLoss(weight=weight, reduction=reduction)
                else:
                    loss_fct = MaskedCrossEntropyLoss(weight=weight, reduction=reduction, ignore_nans=True, ignore_value=ignore_index)
            else:
                if ignore_index is None:
                    loss_fct = nn.CrossEntropyLoss(reduction=reduction)
                else:
                    loss_fct = MaskedCrossEntropyLoss(reduction=reduction, ignore_nans=True, ignore_value=ignore_index)
        else:
            raise Exception("Not support output mode: %s." % output_mode)

    return dropout, hidden_layer, hidden_act, classifier, output, loss_fct
