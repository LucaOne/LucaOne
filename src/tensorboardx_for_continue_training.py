#!/usr/bin/env python
# encoding: utf-8
"""
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2025/5/27 14:02
@project: LucaGenPostTraining
@file: tensorboardx_for_continue_training.py
@desc: tensorboardx for continue training
"""
import math
import argparse
import os.path

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

parser = argparse.ArgumentParser(description='Get Values for Continue Training')
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    required=True,
    help="TensorBoard log dir of which dataset name."
)
parser.add_argument(
    "--dataset_version",
    type=str,
    default=None,
    required=True,
    help="TensorBoard log dir of which dataset version."
)
parser.add_argument(
    "--time_str",
    type=str,
    default=None,
    required=True,
    help="The values of when."
)
parser.add_argument(
    "--step",
    type=int,
    default=None,
    required=True,
    help="The values of which  steps."
)
args = parser.parse_args()

# 指定tensorboard文件路径
tb_log_dir = os.path.join(f"../tb-logs/{args.dataset_name}/{args.dataset_version}/token_level,span_level,seq_level,structure_level/lucaone_gplm/{args.time_str}")

# 创建EventAccumulator对象
event_acc = EventAccumulator(tb_log_dir)

# 加载事件数据
event_acc.Reload()

# 获取所有标签
tags = event_acc.Tags()['scalars']
print("All tags:")
for tag in tags:
    print(tag)
print("#" * 50)
# 读取事件数据
data = {}
min_diff_value = math.inf
min_diff_step = None
max_step = 0
for tag in [
    "logging_epoch",
    "logging_updated_lr",
    "logging_global_avg_loss",
    'logging_cur_epoch_avg_loss',

]:
    scalar_events = event_acc.Scalars(tag)
    print(tag + ":")
    for scalar_event in scalar_events:
        if max_step < scalar_event.step:
            max_step = scalar_event.step
        if abs(scalar_event.step - int(args.step)) < min_diff_value and args.step >= scalar_event.step:
            min_diff_value = abs(scalar_event.step - int(args.step))
            min_diff_step = scalar_event.step

        if scalar_event.step == int(args.step):
            if tag in ["epoch_epoch", "logging_epoch"]:
                print([scalar_event.step, int(scalar_event.value), scalar_event.wall_time])
            else:
                print([scalar_event.step, scalar_event.value, scalar_event.value * scalar_event.step, scalar_event.wall_time])
print("#" * 50)
print("max_step: %s" % max_step)
print("min_diff_value: %d" % min_diff_value)
print("min_diff_step: %d" % min_diff_step)
'''
python tensorboardx_for_continue_training.py --dataset_name lucagplm --dataset_version v2.0 --time_str 20240906230224 --step 65300000 

'''