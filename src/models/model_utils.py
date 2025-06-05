#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/10 11:15
@project: LucaOne
@file: model_utils
@desc: model uitils for LucaOne
'''
from typing import Optional, Tuple
from dataclasses import dataclass
from transformers.modeling_outputs import ModelOutput
import sys, copy, math
sys.path.append(".")
sys.path.append("..")
sys.path.append("../../src")
try:
    from .pooling import *
    from ..common.loss import *
except ImportError:
    from src.models.pooling import *
    from src.common.loss import *


@dataclass
class AllOutput(ModelOutput):
    losses: Optional[dict[str, dict[str, torch.FloatTensor]]] = None
    outputs: Optional[dict[str, dict[str, torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions: Optional[Tuple[torch.FloatTensor]] = None
    contacts: Optional[Tuple[torch.FloatTensor]] = None
    losses_b: Optional[dict[str, dict[str, torch.FloatTensor]]] = None
    outputs_b: Optional[dict[str, dict[str, torch.FloatTensor]]] = None
    hidden_states_b: Optional[Tuple[torch.FloatTensor]] = None
    attentions_b: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions_b: Optional[Tuple[torch.FloatTensor]] = None
    global_attentions_b: Optional[Tuple[torch.FloatTensor]] = None
    contacts_b: Optional[Tuple[torch.FloatTensor]] = None
    pair_outputs: Optional[Tuple[torch.FloatTensor]] = None
    pair_losses: Optional[dict[str, dict[str, torch.FloatTensor]]] = None


def create_pooler(task_level_type, task_level_name, config, args):
    '''
    pooler building
    :param task_level_type:
    :param task_level_name:
    :param config:
    :param args:
    :return:
    '''
    hidden_size = config.hidden_size[task_level_type][task_level_name]
    pooling_type = args.pooling_type[task_level_type][task_level_name]

    if pooling_type == "max":
        return GlobalMaskMaxPooling1D()
    elif pooling_type == "sum":
        return GlobalMaskSumPooling1D(axis=1)
    elif pooling_type == "avg":
        return GlobalMaskAvgPooling1D()
    elif pooling_type == "attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "context_attention":
        return GlobalMaskContextAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "weighted_attention":
        return GlobalMaskWeightedAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "value_attention":
        return GlobalMaskValueAttentionPooling1D(embed_size=hidden_size)
    elif pooling_type == "transformer":
        copy_config = copy.deepcopy(config)
        copy_config.hidden_size = hidden_size
        return GlobalMaskTransformerPooling1D(copy_config)
    else:
        return None


def create_output_loss_lucagplm(task_level_type, task_level_name, config, args):
    '''not cls module'''
    if not hasattr(args, "sigmoid"):
        args.sigmoid = {task_level_type: {}}
    elif task_level_type not in args.sigmoid:
        args.sigmoid[task_level_type] = {}
    args.sigmoid[task_level_type][task_level_name] = False if args.output_mode[task_level_type][task_level_name] \
                                                              in ["multi_class", "multi-class", "regression"] else True
    # 特殊情况，contact需要是sigmoid, 需要思考strcuture需不需要sigmoid
    if task_level_name == "prot_contact":
        args.sigmoid[task_level_type][task_level_name] = True
    config.num_labels = args.label_size[task_level_type][task_level_name]
    if task_level_type in ["token_level", "whole_level"]:
        return_types = ["output", "loss"]
    else:
        return_types = ["dropout", "hidden_layer", "hidden_act", "classifier", "output", "loss"]
    return create_loss_function(
        config,
        args,
        task_level_type=task_level_type,
        task_level_name=task_level_name,
        sigmoid=args.sigmoid[task_level_type][task_level_name],
        output_mode=args.output_mode[task_level_type][task_level_name],
        num_labels=config.num_labels,
        loss_type=args.loss_type[task_level_type][task_level_name],
        ignore_index=args.ignore_index,
        pair_level=True if task_level_type == "pair_level" else False,
        return_types=return_types
    )


def create_output_loss(task_level_type, task_level_name, cls_module, config, args):
    cls = None
    if task_level_type in ["token_level", "whole_level"]:
        cls = cls_module(config)
    dropout, hidden_layer, hidden_act, classifier, output, loss_fct = create_output_loss_lucagplm(
        task_level_type,
        task_level_name,
        config,
        args
    )
    return cls, dropout, hidden_layer, hidden_act, classifier, output, loss_fct
