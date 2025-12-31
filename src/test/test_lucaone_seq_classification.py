#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2025, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/12/30 14:29
@project: lucaone
@file: test_lucaone_seq_classification
@desc: finetune lucaone for Seq Level Classification
'''
import sys
import torch
import lucaone
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer


# model_id
if len(sys.argv) < 2 or sys.argv[1] == "local":
    model_id = "../../checkpoints/LucaGroup/LucaOne-default-step36M"
else:
    model_id = "LucaGroup/LucaOne-default-step36M"


model = AutoModelForSequenceClassification.from_pretrained(
    model_id,
    task_level="seq_level",
    task_type="multi_class",
    classifier_num_labels=4,
    trust_remote_code=True
)

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    trust_remote_code=True
)
print(model)
print("*" * 50)

# finetune all parameters
for param in model.parameters():
    param.requires_grad = True

# create dataset and trainer for training...