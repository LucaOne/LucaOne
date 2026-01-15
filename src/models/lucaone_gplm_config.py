#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 21:04
@project: LucaOne
@file: lucaone_gplm_config
@desc: LucaOne Config
'''
from transformers.configuration_utils import PretrainedConfig


class LucaGPLMConfig(PretrainedConfig):
    def __init__(
            self,
            vocab_size=-1,
            pad_token_id=0,
            max_position_embeddings: int = 4096,
            type_vocab_size: int = 2,
            num_hidden_layers: int = 20,
            hidden_size: int = 2560,
            num_attention_heads: int = 40,
            no_position_embeddings: bool = True,
            no_token_type_embeddings: bool = False,
            alphabet: str = "gene_prot",
            token_dropout: bool = False,
            attention_probs_dropout_prob=0.1,
            hidden_dropout_prob=0.1,
            classifier_dropout_prob=0.1,
            use_embed_layer_norm=False,
            use_last_layer_norm=True,
            embed_scale=1.0,
            ignore_index=-100,
            has_contact_head=False,
            **kwargs
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)
        self.alphabet = alphabet
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.no_token_type_embeddings = no_token_type_embeddings
        self.no_position_embeddings = no_position_embeddings
        self.num_hidden_layers = num_hidden_layers
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.token_dropout = token_dropout
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.hidden_dropout_prob = hidden_dropout_prob
        self.classifier_dropout_prob = classifier_dropout_prob
        self.ignore_index = ignore_index
        self.use_embed_layer_norm = use_embed_layer_norm
        self.use_last_layer_norm = use_last_layer_norm
        self.embed_scale = embed_scale
        self.has_contact_head = has_contact_head

