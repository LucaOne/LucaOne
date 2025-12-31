#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/30 11:32
@project: lucaone
@file: configuration_lucaone
@desc: configuration_lucaone
'''

from .configuration_lucaone import LucaGPLMConfig
from .tokenization_lucaone import LucaGPLMTokenizer, LucaGPLMTokenizerFast
from .modeling_lucaone import (
    LucaGPLMModel,
    LucaGPLMPreTrainedModel,
    LucaGPLMForMaskedLM,
    LucaGPLMForSequenceClassification,
    LucaGPLMForTokenClassification
)
from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForMaskedLM,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification
)

__all__ = [
    "LucaGPLMConfig",
    "LucaGPLMModel",
    "LucaGPLMPreTrainedModel",
    "LucaGPLMTokenizer",
    "LucaGPLMTokenizerFast",
    "LucaGPLMForMaskedLM",
    "LucaGPLMForSequenceClassification",
    "LucaGPLMForTokenClassification"
]


# 1. 注册配置类 (必选)
AutoConfig.register("lucaone", LucaGPLMConfig)

# 2. 注册基础模型 (用于 AutoModel.from_pretrained)
AutoModel.register(LucaGPLMConfig, LucaGPLMModel)

# 3. 注册序列分类模型 (用于 AutoModelForSequenceClassification)
AutoModelForSequenceClassification.register(LucaGPLMConfig, LucaGPLMForSequenceClassification)

# 4. 注册 Token 分类模型 (用于 AutoModelForTokenClassification)
AutoModelForTokenClassification.register(LucaGPLMConfig, LucaGPLMForTokenClassification)

# 5. 注册掩码语言模型 (用于 AutoModelForMaskedLM)
AutoModelForMaskedLM.register(LucaGPLMConfig, LucaGPLMForMaskedLM)