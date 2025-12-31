#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/30 11:34
@project: lucaone
@file: tokenization_lucaone
@desc: tokenization_lucaone
'''

from typing import Literal
from transformers import PretrainedConfig

class LucaGPLMConfig(PretrainedConfig):
    model_type = "lucaone"
    
    def __init__(
        self,
        vocab_size: int = 39,
        pad_token_id: int = 0,
        unk_token_id: int = 1,
        bos_token_id: int = 2,
        eos_token_id: int = 3,
        sep_token_id: int = 3,
        mask_token_id: int = 4,
        hidden_act: str = "gelu",
        max_position_embeddings: int = 4096,
        type_vocab_size: int = 2,
        num_hidden_layers: int = 20,
        num_attention_heads: int = 40,
        hidden_size: int = 2560,
        ffn_dim: int = 10240,
        no_position_embeddings: bool = True,
        no_token_type_embeddings: bool = False,
        alphabet: str = "gene_prot",
        token_dropout: bool = False,
        attention_probs_dropout_prob: float = 0.0,
        hidden_dropout_prob: float = 0.0,
        use_embed_layer_norm: bool = False,
        use_last_layer_norm: bool = True,
        embed_scale: float = 1.0,
        ignore_index: int = -100,
        layer_norm_eps: float = 1e-12,
        initializer_range: float = 0.02,
        task_level: Literal["seq_level", "token_level"]  = "seq_level",
        task_type: Literal["embedding", "mlm", "multi_class", "binary_class", "regression", "multi_label"]  = "embedding",
        classifier_num_labels: int = -1,
        classifier_dropout_prob: float = 0.1,
        classifier_pooling_type: Literal["cls", "value_attention", "context_attention", "mean"]  = "value_attention",
        classifier_loss_type: Literal["binary_cross_entropy", "cross_entropy", "mse", "mae"]  = "cross_entropy",
        classifier_loss_reduction: Literal["mean", "sum", "none"]  = "mean",
        classifier_pos_weight: float=1.0,
        classifier_weight: list=None,
        tie_word_embeddings: bool=True,
        **kwargs
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            pad_token_id=pad_token_id,
            **kwargs
        )
        
        self.alphabet = alphabet
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.no_token_type_embeddings = no_token_type_embeddings
        self.no_position_embeddings = no_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.ffn_dim = ffn_dim
        self.token_dropout = token_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.ignore_index = ignore_index
        self.use_embed_layer_norm = use_embed_layer_norm
        self.use_last_layer_norm = use_last_layer_norm
        self.embed_scale = embed_scale
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.unk_token_id = unk_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.hidden_act = hidden_act
        self.classifier_num_labels = classifier_num_labels
        self.classifier_pooling_type = classifier_pooling_type
        self.task_level = task_level
        self.task_type = task_type
        self.classifier_loss_type = classifier_loss_type
        self.classifier_loss_reduction = classifier_loss_reduction
        self.classifier_pos_weight = classifier_pos_weight
        self.classifier_weight = classifier_weight


__all__ = ["LucaGPLMConfig"]