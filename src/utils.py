#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/26 14:48
@project: LucaOne
@file: utils
@desc: utils for LucaOne
'''
import torch
import requests
import numpy as np
import random
import os, sys
import pynvml
from collections import OrderedDict
sys.path.append("..")
sys.path.append("../src")
sys.path.append("../src/common")
try:
    from .common.multi_label_metrics import metrics_multi_label, prob_2_pred, relevant_indexes
    from .common.metrics import metrics_multi_class, metrics_binary, metrics_regression
    from .common.multi_task_loss import *
except ImportError:
    from src.common.multi_label_metrics import metrics_multi_label, prob_2_pred, relevant_indexes
    from src.common.metrics import metrics_multi_class, metrics_binary, metrics_regression
    from src.common.multi_task_loss import *


def set_seed(args):
    '''
    set seed
    :param args:
    :return:
    '''
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)


def to_device(device, batch):
    '''
    input to device
    :param device:
    :param batch:
    :return:
    '''
    new_batch = {}
    sample_num = 0
    tens = None
    for item1 in batch.items():
        new_batch[item1[0]] = {}
        if isinstance(item1[1], dict):
            for item2 in item1[1].items():
                new_batch[item1[0]][item2[0]] = {}
                if isinstance(item2[1], dict):
                    for item3 in item2[1].items():
                        if item3[1] is not None:
                            new_batch[item1[0]][item2[0]][item3[0]] = item3[1].to(device)
                            tens = item3[1]
                        else:
                            new_batch[item1[0]][item2[0]][item3[0]] = item3[1]
                else:
                    if item2[1] is not None:
                        new_batch[item1[0]][item2[0]] = item2[1].to(device)
                        tens = item2[1]
                    else:
                        new_batch[item1[0]][item2[0]] = item2[1]
        else:
            if item1[1] is not None:
                new_batch[item1[0]] = item1[1].to(device)
                tens = item1[1]
            else:
                new_batch[item1[0]] = item1[1]
    if tens is not None:
        sample_num = tens.shape[0]
    return new_batch, sample_num


def print_batch_input1(batch):
    '''
    print input batch
    :param batch:
    :return:
    '''
    for item in batch.items():
        print(item[0] + ":")
        if isinstance(item[1], dict):
            for item2 in item[1].items():
                print(item2[0] + ":")
                if isinstance(item2[1], dict):
                    for item3 in item2[1].items():
                        print(item3[0] + ":")
                        print(item3[1].shape)
                        print(item3[1])
                else:
                    print(item2[1].shape)
                    print(item2[1])
        else:
            print(item[1].shape)
            print(item[1])


def print_batch_input(batch):
    '''
    print output batch
    :param batch:
    :return:
    '''
    if isinstance(batch, list):
        for item in batch:
            print_batch_output(item)
    elif isinstance(batch, dict):
        for item in batch.items():
            print(item[0] + ":")
            print_batch_output(item[1])
    else:
        print(batch.shape)
        print(batch)
        print(torch.nonzero(torch.ne(batch, -100)))


def print_batch_output(batch):
    '''
    print output batch
    :param batch:
    :return:
    '''
    if isinstance(batch, list):
        for item in batch:
            print_batch_output(item)
    elif isinstance(batch, dict):
        for item in batch.items():
            print(item[0] + ":")
            print_batch_output(item[1])
    else:
        print(batch.shape)
        print(batch)


def process_outputs(output_mode, truth, pred, output_truth, output_pred, ignore_index, keep_seq=False):
    # token_level/mask
    # truth: [N, max_seq_len] pred: [N, max_seq_len, vocab_size]
    # span_level/gene_type
    # truth: [N, max_seq_len] pred: [N, max_seq_len, label_size]
    # seq_level/gene_taxonomy
    # truth: [N, 1] pred: [N, label_size]

    # token_level/mask
    # truth: [N, max_seq_len] pred: [N, max_seq_len, vocab_size]
    # span_level/prot_homo
    # truth: [N, max_seq_len] pred: [N, max_seq_len, label_size]
    # span_level/prot_site
    # truth: [N, max_seq_len] pred: [N, max_seq_len, label_size]
    # seq_level/prot_taxonomy
    # truth: [N, 1] pred: [N, label_size]
    # seq_level/prot_keyword
    # truth: [N, label_size] pred: [N, label_size]
    # structure_level/prot_structure
    # truth: [N, max_seq_len, 3] pred: [N, max_seq_len, 3]

    # pair_level/trans
    # truth: [N, 1] pred: [N, 1]
    if keep_seq:
        # token_level/mask
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # span_level/gene_type
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # seq_level/gene_taxonomy
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # token_level/mask
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # span_level/prot_homo
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # span_level/prot_site
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # seq_level/prot_taxonomy
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1, pred.shape[-1])
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask, :]

        # seq_level/prot_keyword
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1)
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask]

        # structure_level/prot_structure
        truth = truth.view(-1, 3)
        truth_mask = truth[:, 0] != ignore_index
        pred = pred.view(-1, 3)
        masked_truth = truth[truth_mask, :]
        masked_pred = pred[truth_mask, :]

        # pair_level/trans
        truth = truth.view(-1)
        truth_mask = truth != ignore_index
        pred = pred.view(-1)
        masked_truth = truth[truth_mask]
        masked_pred = pred[truth_mask]
    else:
        for item1 in truth.items():
            task_type_level = item1[0]
            for item2 in item1[1].items():
                task_type_name = item2[0]
                cur_truth = truth[task_type_level][task_type_name]
                cur_pred = pred[task_type_level][task_type_name]
                if output_mode[task_type_level][task_type_name] in ["multi_class", "multi-class"]:
                    cur_truth = cur_truth.view(-1)
                    cur_mask = cur_truth != ignore_index
                    cur_pred = cur_pred.view(-1, cur_pred.shape[-1])
                    cur_truth = cur_truth[cur_mask]
                    cur_pred = cur_pred[cur_mask, :]
                elif output_mode[task_type_level][task_type_name] in ["multi_label", "multi-label", "binary_class", "binary-class"]:
                    cur_truth = cur_truth.view(-1)
                    cur_mask = cur_truth != ignore_index
                    cur_pred = cur_pred.view(-1)
                    cur_truth = cur_truth[cur_mask]
                    cur_pred = cur_pred[cur_mask]
                elif output_mode[task_type_level][task_type_name] in ["regression"]:
                    cur_truth = cur_truth.view(-1, 3)
                    cur_mask = (cur_truth[:, 0] != ignore_index) | (cur_truth[:, 1] != ignore_index) | (cur_truth[:, 2] != ignore_index)
                    cur_pred = cur_pred.view(-1, 3)
                    cur_truth = cur_truth[cur_mask, :]
                    cur_pred = cur_pred[cur_mask, :]
                else:
                    raise Exception("not output mode: task_type_level=%s, task_type_name=%s, mode:%s" % (task_type_level, task_type_name, output_mode[task_type_level][task_type_name]))
                    
                if cur_mask.sum().item() > 0:
                    cur_truth = cur_truth.detach().cpu().numpy()
                    cur_pred = cur_pred.detach().cpu().numpy()
                    if task_type_level not in output_truth:
                        output_truth[task_type_level] = {}
                        output_pred[task_type_level] = {}
                        output_truth[task_type_level][task_type_name] = cur_truth
                        output_pred[task_type_level][task_type_name] = cur_pred
                    elif task_type_name not in output_truth[task_type_level]:
                        output_truth[task_type_level][task_type_name] = cur_truth
                        output_pred[task_type_level][task_type_name] = cur_pred
                    else:
                        output_truth[task_type_level][task_type_name] = np.append(output_truth[task_type_level][task_type_name], cur_truth,  axis=0)
                        output_pred[task_type_level][task_type_name] = np.append(output_pred[task_type_level][task_type_name], cur_pred,  axis=0)
    return output_truth, output_pred


def concat_output(ground_truth, outputs, ground_truth_ids, pred_scores):
    '''
    :param ground_truth:
    :param outputs:
    :param ground_truth_ids:
    :param pred_scores:
    :return:
    '''
    if pred_scores is None:
        pred_scores = outputs.detach().cpu().numpy()
        ground_truth_ids = ground_truth.detach().cpu().numpy()
    else:
        pred_scores = np.append(pred_scores, outputs.detach().cpu().numpy(), axis=0)
        ground_truth_ids = np.append(ground_truth_ids, ground_truth.detach().cpu().numpy(), axis=0)
    return ground_truth_ids, pred_scores


def concat_output_tensor(ground_truth, outputs, ground_truth_ids, pred_scores):
    '''
    :param ground_truth:
    :param outputs:
    :param ground_truth_ids:
    :param pred_scores:
    :return:
    '''
    if pred_scores is None:
        pred_scores = outputs
        ground_truth_ids = ground_truth
    else:
        pred_scores = torch.cat((pred_scores, outputs), dim=0)
        ground_truth_ids = torch.cat((ground_truth_ids, ground_truth), dim=0)
    return ground_truth_ids, pred_scores


def get_labels(label_filepath, header=True):
    '''
    get labels from file, exists header
    :param label_filepath:
    :return:
    '''
    with open(label_filepath, "r") as fp:
        labels = []
        multi_cols = False
        cnt = 0
        for line in fp.readlines():
            cnt += 1
            if cnt == 1 and header:
                if line.find(",") > 0:
                    multi_cols = True
                continue
            line = line.strip()
            if multi_cols:
                idx = line.find(",")
                if idx > 0:
                    label_name = line[idx + 1:].strip()
                else:
                    label_name = line
            else:
                label_name = line
            labels.append(label_name)
        return labels


def available_gpu_id():
    """
    计算可用的GPU id
    :return:
    """
    pynvml.nvmlInit()
    if not torch.cuda.is_available():
        print("GPU not available")
        return -1
    # 获取GPU数量
    device_count = pynvml.nvmlDeviceGetCount()
    max_available_gpu = -1
    max_available_rate = 0

    # 遍历所有GPU并检查可用性
    for i in range(device_count):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        memory_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
        # 假设如果GPU利用率小于某个阈值（例如10%），我们认为这个GPU目前是空闲的
        if utilization.gpu < 10 and max_available_rate < 100 - utilization.gpu:
            max_available_rate = 100 - utilization.gpu
            max_available_gpu = i
    # 打印可用的GPU ID
    if max_available_gpu > -1:
        print("Available GPU ID: %d, Free Rate: %0.2f%%" % (max_available_gpu, max_available_rate))
    else:
        print("No Available GPU!")

    # Shutdown NVML
    pynvml.nvmlShutdown()
    return max_available_gpu


def eval_metrics(output_mode, truths, preds, threshold=0.5):
    '''
    eval metrics
    :param output_mode:
    :param truths:
    :param preds:
    :param threshold:
    :return:
    '''
    result = {}
    for item1 in truths.items():
        task_type_level = item1[0]
        result[task_type_level] = {}
        for item2 in item1[1].items():
            task_type_name = item2[0]
            cur_output_mode = output_mode[task_type_level][task_type_name]
            cur_truths = truths[task_type_level][task_type_name]
            cur_preds = preds[task_type_level][task_type_name]
            if cur_output_mode in ["multi-label", "multi_label"]:
                cur_result = metrics_binary(cur_truths, cur_preds, threshold=threshold)
            elif cur_output_mode in ["multi-class", "multi_class"]:
                cur_result = metrics_multi_class(cur_truths, cur_preds)
            elif cur_output_mode == "regression":
                cur_result = metrics_regression(cur_truths, cur_preds)
            elif cur_output_mode in ["binary-class", "binary_class"]:
                cur_result = metrics_binary(cur_truths, cur_preds, threshold=threshold)
            else:
                raise Exception("Not Support this output mode: %s, task_type_level=%s, task_type_name=%s" % (cur_output_mode, task_type_level, task_type_name))
            result[task_type_level][task_type_name] = cur_result

    return result


def metrics_merge(results, all_results):
    for item1 in results.items():
        if item1[0] not in all_results:
            all_results[item1[0]] = {}
        for item2 in item1[1].items():
            if item2[0] not in all_results[item1[0]]:
                all_results[item1[0]][item2[0]] = {}
            for item3 in item2[1].items():
                if item3[0] not in all_results[item1[0]][item2[0]]:
                    all_results[item1[0]][item2[0]][item3[0]] = item3[1]
                else:
                    all_results[item1[0]][item2[0]][item3[0]] += item3[1]
    return all_results


def eval_bak2(output_mode, task_type, ground_truth_ids, pred_scores, label_list=None, ignore_index=None, threshold=0.5,
              output_dir=None, output_filename=None):
    '''
    eval metrics
    :param output_mode:
    :param task_type:
    :param ground_truth_ids:
    :param pred_scores:
    :param label_list:
    :param ignore_index:
    :param threshold:
    :param output_dir:
    :return:
    '''
    if task_type in ["token_level", "whole_level", "span_level"]:
        if output_mode in ["multi-label", "multi_label"]:
            '''
            for example:
            gnore_index = -100 in masked llm
            ground_truth_ids:  N * seq_len * label_num
            pred_scores: N * seq_len * label_num
            '''
            if ignore_index is not None:
                selected_index = ground_truth_ids != ignore_index
                selected_ground_truth_ids = ground_truth_ids[selected_index]
                selected_pred_scores = pred_scores[selected_index]
            else:
                selected_index = ground_truth_ids != ignore_index if ignore_index else -100
                selected_ground_truth_ids = ground_truth_ids.reshape(-1, ground_truth_ids.shape[-1])
                selected_pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])
        else:
            '''
            for example:
            gnore_index = -100 in masked llm
            ground_truth_ids:  N * seq_len
            pred_scores: N * seq_len * label_num
            '''
            if ignore_index is not None:
                selected_index = ground_truth_ids != ignore_index
                selected_ground_truth_ids = ground_truth_ids[selected_index]
                selected_pred_scores = pred_scores[selected_index]
            else:
                selected_index = ground_truth_ids != ignore_index if ignore_index else -100
                selected_ground_truth_ids = ground_truth_ids.reshape(-1)
                selected_pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])
    else:
        # ground_truth_ids: N * 1 or N
        # pred_scores: N * label_num
        selected_index = ground_truth_ids != ignore_index if ignore_index else -100
        selected_ground_truth_ids = ground_truth_ids.reshape(-1)
        selected_pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])

    true_label_names, pred_label_names = None, None
    if output_mode in ["multi-label", "multi_label"]:
        result = metrics_multi_label(selected_ground_truth_ids, selected_pred_scores, threshold=threshold)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
    elif output_mode in ["multi-class", "multi_class"]:
        result = metrics_multi_class(selected_ground_truth_ids, selected_pred_scores)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index)
    elif output_mode == "regression":
        pass  # to do
    elif output_mode in ["binary-class", "binary_class"]:
        result = metrics_binary(selected_ground_truth_ids, selected_pred_scores, threshold=threshold,
                                savepath=os.path.join(output_dir, "dev_confusion_matrix.png") if output_dir else None)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
    else:
        raise Exception("Not Support this output mode: %s" % output_mode)

    if output_dir and output_filename and pred_label_names and true_label_names:
        with open(os.path.join(output_dir, output_filename), "w") as wfp:
            for idx in range(len(pred_label_names)):
                wfp.write("%d,%s,%s\n" % (idx, str(pred_label_names[idx]), str(true_label_names[idx])))

    return result, true_label_names, pred_label_names


def eval_bak(output_mode, task_type, ground_truth_ids, pred_scores, label_list=None, ignore_index=None, threshold=0.5,
             output_dir=None, output_filename=None):
    '''
    eval metrics
    :param output_mode:
    :param task_type:
    :param ground_truth_ids:
    :param pred_scores:
    :param label_list:
    :param ignore_index:
    :param threshold:
    :param output_dir:
    :return:
    '''
    if task_type in ["token_level", "whole_level", "span_level"]:
        if output_mode in ["multi-label", "multi_label"]:
            '''
            for example:
            gnore_index = -100 in masked llm
            ground_truth_ids:  N * seq_len * label_num
            pred_scores: N * seq_len * label_num
            '''
            if ignore_index is not None:
                selected_index = ground_truth_ids != ignore_index
                selected_ground_truth_ids = ground_truth_ids[selected_index]
                selected_pred_scores = pred_scores[selected_index]
            else:
                selected_index = ground_truth_ids != ignore_index if ignore_index else -100
                selected_ground_truth_ids = ground_truth_ids.reshape(-1, ground_truth_ids.shape[-1])
                selected_pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])
        else:
            '''
            for example:
            gnore_index = -100 in masked llm
            ground_truth_ids:  N * seq_len
            pred_scores: N * seq_len * label_num
            '''
            if ignore_index is not None:
                selected_index = ground_truth_ids != ignore_index
                selected_ground_truth_ids = ground_truth_ids[selected_index]
                selected_pred_scores = pred_scores[selected_index]
            else:
                selected_index = ground_truth_ids != ignore_index if ignore_index else -100
                selected_ground_truth_ids = ground_truth_ids.reshape(-1)
                selected_pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])
    else:
        # ground_truth_ids: N * 1 or N
        # pred_scores: N * label_num
        selected_index = ground_truth_ids != ignore_index if ignore_index else -100
        selected_ground_truth_ids = ground_truth_ids.reshape(-1)
        selected_pred_scores = pred_scores.reshape(-1, pred_scores.shape[-1])

    true_label_names, pred_label_names = None, None
    if output_mode in ["multi-label", "multi_label"]:
        result = metrics_multi_label(selected_ground_truth_ids, selected_pred_scores, threshold=threshold)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
    elif output_mode in ["multi-class", "multi_class"]:
        result = metrics_multi_class(selected_ground_truth_ids, selected_pred_scores)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index)
    elif output_mode == "regression":
        pass  # to do
    elif output_mode in ["binary-class", "binary_class"]:
        result = metrics_binary(selected_ground_truth_ids, selected_pred_scores, threshold=threshold,
                                savepath=os.path.join(output_dir, "dev_confusion_matrix.png") if output_dir else None)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
    else:
        raise Exception("Not Support this output mode: %s" % output_mode)

    if output_dir and output_filename and pred_label_names and true_label_names:
        with open(os.path.join(output_dir, output_filename), "w") as wfp:
            for idx in range(len(pred_label_names)):
                wfp.write("%d,%s,%s\n" % (idx, str(pred_label_names[idx]), str(true_label_names[idx])))

    return result, true_label_names, pred_label_names


def eval_tensor(output_mode, task_type, ground_truth_ids, pred_scores, label_list=None, ignore_index=None,
                threshold=0.5, output_dir=None, output_filename=None):
    '''
    eval metrics
    :param output_mode:
    :param task_type:
    :param ground_truth_ids:
    :param pred_scores:
    :param label_list:
    :param ignore_index:
    :param threshold:
    :param output_dir:
    :return:
    '''
    if task_type in ["token_level", "whole_level", "span_level"]:
        if output_mode in ["multi-label", "multi_label"]:
            '''
            for example:
            gnore_index = -100 in masked llm
            ground_truth_ids:  N * seq_len * label_num
            pred_scores: N * seq_len * label_num
            '''
            if ignore_index is not None:
                selected_index = ground_truth_ids != ignore_index
                selected_ground_truth_ids = ground_truth_ids[selected_index]
                selected_pred_scores = pred_scores[selected_index]
            else:
                selected_index = ground_truth_ids != ignore_index if ignore_index else -100
                selected_ground_truth_ids = ground_truth_ids.view(-1, ground_truth_ids.shape[-1])
                selected_pred_scores = pred_scores.view(-1, pred_scores.shape[-1])
        else:
            '''
            for example:
            gnore_index = -100 in masked llm
            ground_truth_ids:  N * seq_len
            pred_scores: N * seq_len * label_num
            '''
            if ignore_index is not None:
                selected_index = ground_truth_ids != ignore_index
                selected_ground_truth_ids = ground_truth_ids[selected_index]
                selected_pred_scores = pred_scores[selected_index]
            else:
                selected_index = ground_truth_ids != ignore_index if ignore_index else -100
                selected_ground_truth_ids = ground_truth_ids.view(-1)
                selected_pred_scores = pred_scores.view(-1, pred_scores.shape[-1])
    else:
        # ground_truth_ids: N * 1 or N
        # pred_scores: N * label_num
        selected_index = ground_truth_ids != ignore_index if ignore_index else -100
        selected_ground_truth_ids = ground_truth_ids.view(-1)
        selected_pred_scores = pred_scores.view(-1, pred_scores.shape[-1])
    selected_index = selected_index.detach().cpu().numpy()
    ground_truth_ids = ground_truth_ids.detach().cpu().numpy()
    pred_scores = pred_scores.detach().cpu().numpy()
    selected_ground_truth_ids = selected_ground_truth_ids.detach().cpu().numpy()
    selected_pred_scores = selected_pred_scores.detach().cpu().numpy()

    true_label_names, pred_label_names = None, None
    if output_mode in ["multi-label", "multi_label"]:
        result = metrics_multi_label(selected_ground_truth_ids, selected_pred_scores, threshold=threshold)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
    elif output_mode in ["multi-class", "multi_class"]:
        result = metrics_multi_class(selected_ground_truth_ids, selected_pred_scores)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index)
    elif output_mode == "regression":
        pass  # to do
    elif output_mode in ["binary-class", "binary_class"]:
        result = metrics_binary(selected_ground_truth_ids, selected_pred_scores, threshold=threshold,
                                savepath=os.path.join(output_dir, "dev_confusion_matrix.png") if output_dir else None)
        if label_list:
            pred_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=pred_scores,
                                                     ground_truth=None, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
            true_label_names = label_id_2_label_name(output_mode, label_list=label_list, prob=None,
                                                     ground_truth=ground_truth_ids, ignore_index=ignore_index,
                                                     selected_index=selected_index, threshold=0.5)
    else:
        raise Exception("Not Support this output mode: %s" % output_mode)

    if output_dir and output_filename and pred_label_names and true_label_names:
        with open(os.path.join(output_dir, output_filename), "w") as wfp:
            for idx in range(len(pred_label_names)):
                wfp.write("%d,%s,%s\n" % (idx, str(pred_label_names[idx]), str(true_label_names[idx])))

    return result, true_label_names, pred_label_names


def label_id_2_label_name(output_mode, label_list, prob, ground_truth, ignore_index, selected_index, threshold=0.5):
    '''
    convect label id to label name
    :param output_mode:
    :param label_list:
    :param prob:
    :param threshold:
    :return:
    '''
    if prob is not None:
        if torch.is_tensor(prob):
            prob = prob.detach().cpu().numpy()
        if selected_index is not None and torch.is_tensor(selected_index):
            selected_index = selected_index.detach().cpu().numpy()
        shape = prob.shape
        if output_mode in ["multi-label", "multi_label"]:
            res = []
            pred = prob_2_pred(prob, threshold)
            pred_index = relevant_indexes(pred)
            if prob.ndim == 3:  # N * seq_len * one_hot
                for x in range(shape[0]):
                    label_names_x = []
                    for y in range(shape[1]):
                        label_names_y = []
                        for label_idx in pred_index[x][y]:
                            if label_idx != ignore_index:
                                label_names_y.append(label_list[label_idx])
                            else:
                                label_names_y.append('N')
                        label_names_x.append(label_names_y)
                    res.append(label_names_x)
            elif prob.ndim == 2:  # N * one_hot
                for x in range(shape[0]):
                    label_names_x = []
                    for label_idx in pred_index[x]:
                        if label_idx != ignore_index:
                            label_names_x.append(label_list[label_idx])
                        else:
                            label_names_x.append('N')
                    res.append(label_names_x)
            return res
        elif output_mode in ["multi-class", "multi_class"]:
            res = []
            # N * seq_len * label_num
            if prob.ndim == 3:
                # N * seq_len
                pred = np.argmax(prob, axis=-1)
                for x in range(shape[0]):
                    label_name_x = []
                    for y in range(shape[1]):
                        if selected_index[x][y]:
                            label_name_x.append(label_list[pred[x][y]])
                        else:
                            label_name_x.append('N')
                    res.append(label_name_x)
            elif prob.ndim == 2:  # N * label_num
                pred = np.argmax(prob, axis=-1)
                for x in range(prob.shape[0]):
                    if selected_index[x]:
                        res.append(label_list[pred[x]])
                    else:
                        res.append('N')
            return res
        elif output_mode in ["binary-class", "binary_class"]:
            res = []
            if prob.ndim == 3 or prob.ndim == 2 and shape[1] > 1:  # N * Seq_len * 1 or N * Seq_len
                #  N * Seq_len
                prob = prob.reshape(-1, shape[1])
                pred = prob_2_pred(prob, threshold)
                for x in range(shape[0]):
                    label_name_x = []
                    for y in range(shape[1]):
                        label_idx = pred[x][y]
                        if label_idx != ignore_index:
                            label_name_x.append(label_list[label_idx])
                        else:
                            label_name_x.append('N')
                    res.append(label_name_x)
                return res
            elif prob.ndim == 2:  # N * 1 -> N
                prob = prob.flatten(order="C")
                selected_index = prob.flatten(order="C")
            pred = prob_2_pred(prob, threshold)
            for x in range(shape[0]):
                if selected_index[x]:
                    res.append(label_list[pred[x]])
                else:
                    res.append('N')
            return res
        else:
            raise Exception("Not support the output_mode: %s" % output_mode)
    else:
        if torch.is_tensor(ground_truth):
            ground_truth = ground_truth.detach().cpu().numpy()
        if torch.is_tensor(selected_index):
            selected_index = selected_index.detach().cpu().numpy()
        shape = ground_truth.shape

        if output_mode in ["multi-label", "multi_label"]:
            res = []
            ground_truth_index = relevant_indexes(ground_truth)
            # N * Seq_len * one_hot
            if ground_truth.ndim == 3:
                for x in range(shape[0]):
                    label_names_x = []
                    for y in range(shape[1]):
                        label_names_y = []
                        for label_idx in ground_truth_index[x][y]:
                            if label_idx != ignore_index:
                                label_names_y.append(label_list[label_idx])
                            else:
                                label_names_y.append('N')
                        label_names_x.append(label_names_y)
                    res.append(label_names_x)
            elif ground_truth.ndim == 2:  # N * one_hot
                for x in range(shape[0]):
                    label_names_x = []
                    for label_idx in ground_truth_index[x]:
                        if label_idx != ignore_index:
                            label_names_x.append(label_list[label_idx])
                        else:
                            label_names_x.append('N')
                    res.append(label_names_x)
            return res
        elif output_mode in ["multi-class", "multi_class"]:
            res = []
            # N * Seq_len * 1
            if ground_truth.ndim == 3:
                for x in range(shape[0]):
                    label_name_x = []
                    for y in range(shape[1]):
                        if selected_index[x][y][0]:
                            label_name_x.append(label_list[ground_truth[x][y][0]])
                        else:
                            label_name_x.append('N')
                    res.append(label_name_x)
            elif ground_truth.ndim == 2 and shape[1] != 1:  # N * Seq_len
                for x in range(shape[0]):
                    label_name_x = []
                    for y in range(shape[1]):
                        if selected_index[x][y]:
                            label_name_x.append(label_list[ground_truth[x][y]])
                        else:
                            label_name_x.append('N')
                    res.append(label_name_x)
            elif ground_truth.ndim == 2 and shape[1] == 1:  # N * 1
                for x in range(shape[0]):
                    if selected_index[x][0]:
                        res.append(label_list[ground_truth[x][0]])
                    else:
                        res.append('N')
            return res
        elif output_mode in ["binary-class", "binary_class"]:
            res = []
            # N * Seq_len * 1 or N * Seq_len
            if ground_truth.ndim == 3 or ground_truth.ndim == 2 and shape[1] != 1:
                # N * Seq_len
                ground_truth = ground_truth.reshape(-1, shape[1])
                # N * Seq_len
                ground_truth_index = prob_2_pred(ground_truth, threshold)
                for x in range(shape[0]):
                    label_name_x = []
                    for y in range(shape[1]):
                        label_idx = ground_truth_index[x][y]
                        if label_idx != ignore_index:
                            label_name_x.append(label_list[label_idx])
                        else:
                            label_name_x.append('N')
                    res.append(label_name_x)
                return res
            elif ground_truth.ndim == 2:  # N * 1 -> N
                ground_truth = ground_truth.flatten(order="C")
                selected_index = selected_index.flatten(order="C")
            ground_truth_index = prob_2_pred(ground_truth, threshold)
            for x in range(shape[0]):
                if selected_index[x]:
                    res.append(label_list[ground_truth_index[x]])
                else:
                    res.append('N')
            return res
        else:
            raise Exception("Not support the output_mode: %s" % output_mode)


def get_lr(optimizer):
    '''
    get learning rate
    :param optimizer:
    :return:
    '''
    for p in optimizer.param_groups:
        return p["lr"]


def get_parameter_number(model):
    '''
    colc the parameter number of the model
    :param model:
    :return:
    '''
    param_size = 0
    param_sum = 0
    trainable_size = 0
    trainable_num = 0
    for param in model.parameters():
        cur_size = param.nelement() * param.element_size()
        cur_num = param.nelement()
        param_size += cur_size
        param_sum += cur_num
        if param.requires_grad:
            trainable_size += cur_size
            trainable_num += cur_num
    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()
    '''
    total_num = sum(p.numel() for p in model.parameters())
    total_size = sum(p.numel() * p.element_size() for p in model.parameters())
    total_num += sum(p.numel() for p in model.buffers())
    total_size += sum(p.numel() * p.element_size() for p in model.buffers())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_size = sum(p.numel() * p.element_size() for p in model.parameters() if p.requires_grad)
    '''
    return {
        'total_num': "%fM" % round((buffer_sum + param_sum) / (1024 * 1024), 2),
        'total_size': "%fMB" % round((buffer_size + param_size) / (1024 * 1024), 2),
        'param_sum': "%fM" % round(param_sum / (1024 * 1024), 2),
        'param_size': "%fMB" % round(param_size / (1024 * 1024), 2),
        'buffer_sum': "%fM" % round(buffer_sum / (1024 * 1024), 2),
        'buffer_size': "%fMB" % round(buffer_size / (1024 * 1024), 2),
        'trainable_num': "%fM" % round(trainable_num / (1024 * 1024), 2),
        'trainable_size': "%fMB" % round(trainable_size / (1024 * 1024), 2)
    }


def calc_loss_index(args):
    if not hasattr(args, "index_list"):
        index_list = []
        if "token_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
            index_list.append(0)
        if "span_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
            index_list.append(1)
        if "seq_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
            index_list.append(2)
        args.index_list = index_list
        args.task_num = len(index_list)


def calc_loss(args, cur_losses, last_last_loss_list=None, last_loss_list=None):
    '''
    多loss的loss计算，用于反向传播
    :param args:
    :param cur_losses:
    :param last_last_loss_list:
    :param last_loss_list:
    :return:
    '''
    if args.multi_loss_strategy == "manual_weight":
        weights = args.loss_weights
        # print("args.loss_weights:")
        # print(args.loss_weights)
        cur_weight_losses = []
        if len(cur_losses) == 1:
            for item1 in cur_losses[0].items():
                key1 = item1[0]
                for item2 in item1[1].items():
                    key2 = item2[0]
                    if item2[1] is not None and not isinstance(item2[1], float):
                        '''
                        print(key1, key2, ":")
                        print(weights[key1][key2])
                        print(item2[1])
                        '''
                        cur_weight_losses.append(weights[key1][key2] * item2[1])
        else:
            for idx in range(len(cur_losses)):
                for item1 in cur_losses[idx].items():
                    key1 = item1[0]
                    for item2 in item1[1].items():
                        key2 = item2[0]
                        if item2[1] is not None and not isinstance(item2[1], float):
                            cur_weight_losses.append(weights[key1][key2] * item2[1])
        '''
        print("cur_weight_losses:")
        print(cur_weight_losses)
        '''
        return sum(cur_weight_losses)
    elif args.multi_loss_strategy == "auto_weight":
        cur_loss_list = []
        if len(cur_losses) == 1:
            for item1 in cur_losses[0].items():
                for item2 in item1[1].items():
                    if item2[1] is not None and not isinstance(item2[1], float):
                        cur_loss_list.append(item2[1])
        else:
            for idx in range(len(cur_losses)):
                for item1 in cur_losses[idx].items():
                    for item2 in item1[1].items():
                        if item2[1] is not None and not isinstance(item2[1], float):
                            cur_loss_list.append(item2[1])
        return args.awl(*cur_loss_list)
    elif args.multi_loss_strategy == "dynamic_weight_average":
        cur_losses = []
        last_losses = []
        last_last_losses = []
        cur_loss_list = []
        if len(cur_losses) == 1:
            for item1 in cur_losses[0].items():
                for item2 in item1[1].items():
                    if item2[1] is not None and not isinstance(item2[1], float):
                        cur_loss_list.append(item2[1])
        else:
            for idx in range(len(cur_losses)):
                for item1 in cur_losses[idx].items():
                    for item2 in item1[1].items():
                        if item2[1] is not None and not isinstance(item2[1], float):
                            cur_loss_list.append(item2[1])
        for idx in range(len(cur_loss_list)):
            cur_losses.append(cur_loss_list[idx])
            last_losses.append(last_loss_list[idx])
            last_last_losses.append(last_last_loss_list[idx])
        weights = dynamic_weight_average(last_last_losses, last_losses, args.task_num)
        return sum([weights[idx] * cur_losses[idx] for idx in len(weights)])
    elif args.multi_loss_strategy in ["none", "default"]:
        cur_loss_list = []
        if len(cur_losses) == 1:
            for item1 in cur_losses[0].items():
                for item2 in item1[1].items():
                    if item2[1] is not None and not isinstance(item2[1], float):
                        cur_loss_list.append(item2[1])
        else:
            for idx in range(len(cur_losses)):
                for item1 in cur_losses[idx].items():
                    for item2 in item1[1].items():
                        if item2[1] is not None and not isinstance(item2[1], float):
                            cur_loss_list.append(item2[1])
        return sum(cur_loss_list)
    else:
        raise Exception("not support this multi loss strategy: %s" % args.multi_loss_strategy)


def writer_info_tb(tb_writer, logs, global_step, prefix=None):
    '''
    write info to tensorboard
    :param tb_writer:
    :param logs:
    :param global_step:
    :param prefix:
    :return:
    '''
    for key, value in logs.items():
        if isinstance(value, dict):
            '''
            for key1, value1 in value.items():
                tb_writer.add_scalar(key + "_" + key1, value1, global_step)
            '''
            writer_info_tb(tb_writer, value, global_step, prefix=key)
        else:
            tb_writer.add_scalar(prefix + "_" + key if prefix else key, value, global_step)


def calc_avg_loss(total_losses, nb_steps):
    '''
    计算多种loss得平均loss与总loss
    :param total_losses:
    :param nb_steps:
    :return:
    '''
    loss_detail = {}
    loss = 0
    for item1 in total_losses.items():
        key1 = item1[0]
        if key1 not in loss_detail:
            loss_detail[key1] = {}
        for item2 in item1[1].items():
            key2 = item2[0]
            v = item2[1] / nb_steps
            if key2 not in loss_detail[key1]:
                loss_detail[key1][key2] = float(v)
            else:

                loss_detail[key1][key2] += float(v)
            loss += float(v)
    all_result = {
        "loss_detail": loss_detail,
        "loss": loss
    }
    return all_result, loss, loss_detail


aa_d3to1 = {
    'CYS': 'C',
    'ASP': 'D',
    'SER': 'S',
    'GLN': 'Q',
    'LYS': 'K',
    'ILE': 'I',
    'PRO': 'P',
    'THR': 'T',
    'PHE': 'F',
    'ASN': 'N',
    'GLY': 'G',
    'HIS': 'H',
    'LEU': 'L',
    'ARG': 'R',
    'TRP': 'W',
    'ALA': 'A',
    'VAL': 'V',
    'GLU': 'E',
    'TYR': 'Y',
    'MET': 'M',
    'SEC': 'U',
    'PYL': 'O'
}


def seq_type_is_match_seq(seq_type, seq):
    """
    判断序列内容与序列类型是否匹配
    :param seq_type:
    :param seq:
    :return:
    """
    if seq_type is None or seq is None:
        return False
    seq = seq.strip().upper()
    atcgu_num = 0
    total_num = 0
    for ch in seq:
        if ch < 'A' or ch > 'Z':
            continue
        total_num += 1
        if ch in {"A", "T", "C", "G", "U", "N"}:
            atcgu_num += 1

    is_gene = False
    if total_num == atcgu_num or atcgu_num >= 0.8 * total_num:
        is_gene = True

    if is_gene and seq_type == "gene":
        return True
    if not is_gene and seq_type == "prot":
        return True
    return False


def gene_seq_replace_re(seq):
    '''
    Nucleic acid 还原
    :param seq:
    :return:
    '''
    new_seq = ""
    for ch in seq:
        if ch == '1':
            new_seq += "A"
        elif ch == '2':
            new_seq += "T"
        elif ch == '3':
            new_seq += "C"
        elif ch == '4':
            new_seq += "G"
        else: # unknown
            new_seq += "N"
    return new_seq


def gene_seq_replace(seq):
    '''
    Nucleic acid （gene replace: A->1, U/T->2, C->3, G->4, N->5
    :param seq:
    :return:
    '''
    new_seq = ""
    for ch in seq:
        if ch in ["A", "a"]:
            new_seq += "1"
        elif ch in ["T", "U", "t", "u"]:
            new_seq += "2"
        elif ch in ["C", "c"]:
            new_seq += "3"
        elif ch in ["G", "g"]:
            new_seq += "4"
        else: # unknown
            new_seq += "5"
    return new_seq


def span_merge(spans, start_index=0, end_index=1, value_index=2, merge_type="intersection"):
    '''
    区间合并，删除子区间，合并连在一起的
    :param spans:
    :param start_index:
    :param end_index:
    :param value_index:
    :param merge_type: 合并类型，intersection：只要有交集就合并， sub: 要是子集才合并； join: 包括首尾相接的， sub-join: 子集或者首尾相接的情况
    :return:
    '''
    sorted_spans = sorted(spans, key=lambda x:(x[start_index], -x[end_index]))
    result = []
    for span in sorted_spans:
        if result:
            if merge_type == "intersection" and result[-1][end_index] > span[start_index] and result[-1][value_index] == span[value_index]:
                # result中最后一个区间的右值>新区间的左值，说明两个区间有重叠，这种有交集，但是交集不是首尾相接
                # 将result中最后一个区间更新为合并之后的新区间
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            elif merge_type == "sub" and result[-1][end_index] >= span[end_index] and result[-1][value_index] == span[value_index]:
                # 要是子集包含
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            elif merge_type == "join" and result[-1][end_index] >= span[start_index] and result[-1][value_index] == span[value_index]:
                # 有交集或者首尾相接的情况
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            elif merge_type == "sub-join" and (result[-1][end_index] == span[start_index] or result[-1][end_index] >= span[end_index]) and result[-1][value_index] == span[value_index]:
                # 子集或者首尾相接的情况
                result[-1][end_index] = max(result[-1][end_index], span[end_index])
            else:
                result.append(span)
        else:
            result.append(span)

    return result


def re_positional(seq, tokens, tokenizer, special_tokens, labels, ignore_index=-100):
    '''
    {
        'span_level': {
            'gene_type': [[0, 4, 3], [7, 8, 1], [10, 13, 3]]
        },
        'seq_level': {
            'gene_taxonomy': 1
        }
    }
    {
        'span_level': {
            'prot_homo': [[0, 3, 16], [6, 8, 2820], [12, 14, 2141]],
            'prot_site': [[0, 5, 342]]
        },
        'seq_level': {
            'prot_taxonomy': 947,
            'prot_keyword': [7, 558, 669, 709, 715, 750, 751, 760, 768, 838, 982, 1031, 1047, 1094]
        },
        'structure_level': {
            'prot_structure': [[0.571536098134849, 0.3495862586035716, 0.25961893180173456], [0.1713861923925235, 0.017310542411784646, 0.620125363591797]]
        }
    }

    {
        'span_level': {
            'gene_type': [[0, 4, 3], [7, 8, 1], [10, 13, 3]],
            'prot_homo': [[0, 3, 16], [6, 8, 2820], [12, 14, 2141]],
            'prot_site': [[0, 5, 342]]
        },
        'seq_level': {
            'gene_taxonomy': 1,
            'prot_taxonomy': 947,
            'prot_keyword': [7, 558, 669, 709, 715, 750, 751, 760, 768, 838, 982, 1031, 1047, 1094]
        },
        'structure_level': {
            'prot_structure': [[0.571536098134849, 0.3495862586035716, 0.25961893180173456], [0.1713861923925235, 0.017310542411784646, 0.620125363591797]]
        }
    }
    {
        'pair_level': {
            'trans': 0
        }
    }
    '''
    # token_level 没有位置
    # span_level 有位置，需要重定义
    # seq_level 没有位置
    # structure_level 每个氨基酸的坐标，转变为每个token（多个氨基酸的位置平均）[[x1, y1, z1], [x2, y2, x2], [x3, y3, z3]]
    # pair_level 没有位置
    '''
    :param seq:
    :param tokenizer:
    :param label_dict:
    :return:
    '''
    gene_type = None
    prot_site = None
    prot_homo = None
    prot_structure = None
    if "span_level" in labels and "gene_type" in labels["span_level"]:
        gene_type = labels["span_level"]["gene_type"]
    if "span_level" in labels and "prot_site" in labels["span_level"]:
        prot_site = labels["span_level"]["prot_site"]
    if "span_level" in labels and "prot_homo" in labels["span_level"]:
        prot_homo = labels["span_level"]["prot_homo"]
    if "structure_level" in labels and "prot_structure" in labels["structure_level"]:
        prot_structure = labels["structure_level"]["prot_structure"]
    if tokens is None:
        tokens = tokenizer.tokenize(seq)

        '''
        token_ids = tokenizer.encode_plus(text=seq,
                                  text_pair=None,
                                  add_special_tokens=True,
                                  padding="max_length",
                                  max_length=3,
                                  return_attention_mask=True,
                                  return_token_type_ids=False,
                                  return_length=False,
                                  truncation=True
                                  )
        print("tokenizer:")
        print(tokenizer)
        print("seq:")
        print(seq)
        print("tokens:")
        print(tokens)
        print("token_ids:")
        print(token_ids)
        print("labels:")
        print(labels)
        '''
    '''
    print("seq:")
    print(seq)
    print(tokens)
    print(labels)
    '''
    if gene_type or prot_site or prot_homo or prot_structure:
        new_gene_type = []
        new_prot_site = []
        new_prot_homo = []
        new_prot_structure = []
        # 原始位置到token位置的映射
        ori_pos = 0
        ori_pos_2_token_idx = {}
        for token_idx, token in enumerate(tokens):
            if token in special_tokens:
                ori_pos_2_token_idx[ori_pos] = token_idx
                ori_pos += 1
                if prot_structure:
                    new_prot_structure.append(prot_structure[ori_pos])
            else:
                token_len = len(token)
                begin = ori_pos
                end = ori_pos + token_len
                if prot_structure:
                    pos_num = 0
                    xyz = [0, 0, 0]
                for idx in range(begin, end):
                    ori_pos_2_token_idx[idx] = token_idx
                    if prot_structure:
                        if len(prot_structure[idx]) != 3 or \
                                prot_structure[idx][0] == ignore_index \
                                and prot_structure[idx][1] == ignore_index \
                                and prot_structure[idx][2] == ignore_index:
                            pass
                        else:
                            pos_num += 1
                            xyz[0] += prot_structure[idx][0]
                            xyz[1] += prot_structure[idx][1]
                            xyz[2] += prot_structure[idx][2]
                if prot_structure:
                    if pos_num == 0:
                        new_prot_structure.append([ignore_index, ignore_index, ignore_index])
                    else:
                        new_prot_structure.append([xyz[0]/pos_num, xyz[1]/pos_num, xyz[2]/pos_num])
                ori_pos += len(token)
        if prot_structure:
            labels["structure_level"]["prot_structure"] = new_prot_structure
        if gene_type:
            for item in gene_type:
                start = item[0]
                end = item[1]
                type_v = item[2]
                new_start = ori_pos_2_token_idx[start]
                if end not in ori_pos_2_token_idx:
                    print("gene_type:")
                    print("seq:")
                    print(seq)
                    print(tokens)
                    print(labels)

                new_end = ori_pos_2_token_idx[end]
                new_gene_type.append([new_start, new_end, type_v])
            # 可能存在区间需要合并（有overleap或者首位拼接的，并且类型相同的才合并）
            new_gene_type = span_merge(new_gene_type, start_index=0, end_index=1, value_index=2, merge_type="join")
            labels["span_level"]["gene_type"] = new_gene_type
        if prot_site:
            for item in prot_site:
                start = item[0]
                end = item[1]
                type_v = item[2]
                new_start = ori_pos_2_token_idx[start]
                if end not in ori_pos_2_token_idx:
                    print("prot_site:")
                    print("seq:")
                    print(seq)
                    print(tokens)
                    print(labels)
                new_end = ori_pos_2_token_idx[end]
                new_prot_site.append([new_start, new_end, type_v])
            # 可能存在区间需要合并（有overleap或者首位拼接的，并且类型相同的才合并）
            new_prot_site = span_merge(new_prot_site, start_index=0, end_index=1, value_index=2, merge_type="join")
            labels["span_level"]["prot_site"] = new_prot_site
        if prot_homo:
            for item in prot_homo:
                start = item[0]
                end = item[1]
                type_v = item[2]
                new_start = ori_pos_2_token_idx[start]
                if end not in ori_pos_2_token_idx:
                    print("prot_site:")
                    print("seq:")
                    print(seq)
                    print(tokens)
                    print(labels)
                new_end = ori_pos_2_token_idx[end]
                new_prot_homo.append([new_start, new_end, type_v])
            # 可能存在区间需要合并（有overleap或者首位拼接的，并且类型相同的才合并）
            new_prot_homo = span_merge(new_prot_homo, start_index=0, end_index=1, value_index=2, merge_type="join")
            labels["span_level"]["prot_homo"] = new_prot_homo
        return tokens, labels
    return tokens, labels


def calc_eval_test_loss(losses, total_losses, total_loss):
    cur_loss = 0.0
    current_losses = {}
    for cur_losses in losses:
        for item1 in cur_losses.items():
            key1 = item1[0]
            if key1 not in total_losses:
                total_losses[key1] = {}
            if key1 not in current_losses:
                current_losses[key1] = {}
            for item2 in item1[1].items():
                key2 = item2[0]
                if item2[1] is not None:
                    if torch.is_tensor(item2[1]):
                        v = item2[1].item()
                    else:
                        v = item2[1]
                    if key2 not in total_losses[key1]:
                        total_losses[key1][key2] = v
                    else:
                        total_losses[key1][key2] += v
                    if key2 not in current_losses[key1]:
                        current_losses[key1][key2] = v
                    else:
                        current_losses[key1][key2] += v
                    total_loss += v
                    cur_loss += v
    return current_losses, total_losses, total_loss, cur_loss


def print_shape(item):
    '''
    print shape
    :param item:
    :return:
    '''
    if isinstance(item, dict):
        for item1 in item.items():
            print(item1[0] + ":")
            print_shape(item1[1])
    elif isinstance(item, list):
        for idx, item1 in enumerate(item):
            print("idx: %d" % idx)
            print_shape(item1)
    else:
        print("shape:", item.shape)


def print_batch(value, key=None, debug_path=None, wfp=None, local_rank=-1):
    '''
    print a batch
    :param value:
    :param key:
    :param debug_path:
    :param wfp:
    :param local_rank:
    :return:
    '''
    if isinstance(value, list):
        for idx, v in enumerate(value):
            if wfp is not None:
                if v is not None:
                    wfp.write(str([torch.min(v), torch.min(torch.where(v == -100, 10000, v)), torch.max(v)]) + "\n")
                    wfp.write(str(v.shape) + "\n")
                else:
                    wfp.write("None\n")
                wfp.write("-" * 10 + "\n")
            else:
                if v is not None:
                    print([torch.min(v), torch.min(torch.where(v == -100, 10000, v)), torch.max(v)])
                    print(v.shape)
                else:
                    print("None")
                print("-" * 50)
            if v is not None:
                try:
                    value = v.detach().cpu().numpy().astype(int)
                    if debug_path is not None:
                        if value.ndim == 3:
                            for dim_1_idx in range(value.shape[0]):
                                np.savetxt(os.path.join(debug_path, "%s_batch_%d.txt" % (key, dim_1_idx)), value[dim_1_idx, :, :], fmt='%i', delimiter=",")
                        else:
                            np.savetxt(os.path.join(debug_path, "%d.txt" % idx), value, fmt='%i', delimiter=",")
                    else:
                        if value.ndim == 3:
                            for dim_1_idx in range(value.shape[0]):
                                np.savetxt(os.path.join(debug_path, "%s_batch_%d.txt" % (key, dim_1_idx)), value[dim_1_idx, :, :], fmt='%i', delimiter=",")
                        else:
                            np.savetxt("%d.txt" % idx, value, fmt='%i', delimiter=",")
                except Exception as e:
                    print(e)
    elif isinstance(value, dict):
        for item in value.items():
            if wfp is not None:
                wfp.write(str(item[0]) + ":\n")
            else:
                print(str(item[0]) + ':')
            print_batch(item[1], item[0], debug_path, wfp, local_rank)
    else:
        if wfp is not None:
            if value is not None:
                wfp.write(str([torch.min(value), torch.min(torch.where(value == -100, 10000, value)), torch.max(value)]) + "\n")
                wfp.write(str(value.shape) + "\n")
            else:
                wfp.write("None\n")
            wfp.write("-" * 10 + "\n")
        else:
            if value is not None:
                print([torch.min(value), torch.min(torch.where(value == -100, 10000, value)), torch.max(value)])
                print(value)
                print(value.shape)
            else:
                print("None")
            print("-" * 10)
        if value is not None:
            if key != "prot_structure":
                fmt = '%i'
                d_type = int
            else:
                fmt = '%0.4f'
                d_type = float
            try:
                value = value.detach().cpu().numpy().astype(d_type)
                if debug_path is not None:
                    if value.ndim == 3:
                        for dim_1_idx in range(value.shape[0]):
                            np.savetxt(os.path.join(debug_path, "%s_batch_%d.txt" % (key, dim_1_idx)), value[dim_1_idx, :, :], fmt=fmt, delimiter=",")
                    else:
                        np.savetxt(os.path.join(debug_path, "%s.txt" % key), value, fmt=fmt, delimiter=",")
                else:
                    if value.ndim == 3:
                        for dim_1_idx in range(value.shape[0]):
                            np.savetxt("%s_batch_%d.txt" % (key, dim_1_idx), value[dim_1_idx, :, :], fmt=fmt, delimiter=",")
                    else:
                        np.savetxt("%s.txt" % key, value, fmt=fmt, delimiter=",")
            except Exception as e:
                print(e)


def save_model_parameters(model, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    for name, param in model.named_parameters():
        weights = param.data.numpy()
        if param.requires_grad:
            np.savetxt(os.path.join(save_path, "%s_grad.txt" % name), weights, fmt="%.6f", delimiter="\n")
        else:
            np.savetxt(os.path.join(save_path, "%s_no_grad.txt" % name), weights, fmt="%.6f", delimiter="\n")


def load_trained_model(model_config, args, model_class, model_dirpath):
    # load exists checkpoint
    print("load pretrained model: %s" % model_dirpath)
    try:
        model = model_class.from_pretrained(model_dirpath, args=args)
    except Exception as e:
        model = model_class(model_config, args=args)
        pretrained_net_dict = torch.load(os.path.join(args.model_dirpath, "pytorch.pth"),
                                         map_location=torch.device("cpu"))
        model_state_dict_keys = set()
        for key in model.state_dict():
            model_state_dict_keys.add(key)
        new_state_dict = OrderedDict()
        for k, v in pretrained_net_dict.items():
            if k.startswith("module."):
                # remove `module.`
                name = k[7:]
            else:
                name = k
            if name in model_state_dict_keys:
                new_state_dict[name] = v
        print("diff:")
        print(model_state_dict_keys.difference(new_state_dict.keys()))
        model.load_state_dict(new_state_dict)
    return model


def clean_seq(protein_id, seq, return_rm_index=False):
    seq = seq.upper()
    new_seq = ""
    has_invalid_char = False
    invalid_char_set = set()
    return_rm_index_set = set()
    for idx, ch in enumerate(seq):
        if 'A' <= ch <= 'Z' and ch not in ['J']:
            new_seq += ch
        else:
            invalid_char_set.add(ch)
            return_rm_index_set.add(idx)
            has_invalid_char = True
    if has_invalid_char:
        print("id: %s. Seq: %s" % (protein_id, seq))
        print("invalid char set:", invalid_char_set)
        print("return_rm_index:", return_rm_index_set)
    if return_rm_index:
        return new_seq, return_rm_index_set
    return new_seq


def gcd(x, y):
    '''
    最大公约数
    :param x:
    :param y:
    :return:
    '''
    m = max(x, y)
    n = min(x, y)
    while m % n:
        m, n = n, m % n
    return n


def lcm(x, y):
    '''
    最小公倍数
    :param x:
    :param y:
    :return:
    '''
    m = max(x, y)
    n = min(x, y)
    while m % n:
        m, n = n, m % n
    return x*y//n


def dict_update(raw, new):
    dict_update_iter(raw, new)
    dict_add(raw, new)


def dict_update_iter(raw, new):
    for key in raw:
        if key not in new.keys():
            continue
        if isinstance(raw[key], dict) and isinstance(new[key], dict):
            dict_update(raw[key], new[key])
        else:
            raw[key] = new[key]


def dict_add(raw, new):
    update_dict = {}
    for key in new:
        if key not in raw.keys():
            update_dict[key] = new[key]

    raw.update(update_dict)


def write_processed_sample_ids(dataset_type, time_str, sample_ids, epoch, local_rank):
    """
    将已处理的样本id写入，便于恢复现场
    :param dataset_type: 数据集类型，train，validation，
    :param sample_ids: 要写入的sample_ids
    :param time_str: modeling time str
    :param epoch: 当前第几个epoch
    :param local_rank: cuda的id
    :return:
    """
    if local_rank > -1:
        dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "processed_samples",
            time_str,
            dataset_type,
            "rank-%d" % local_rank,
            "epoch-%d" % epoch
        )
    else:
        dir_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "processed_samples",
            time_str,
            dataset_type,
            "epoch-%d" % epoch
        )
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    size = len(sample_ids)
    if size > 0:
        if local_rank > -1:
            file_path = os.path.join(
                dir_path,
                "%sed_sample_ids_epoch_%d_rank_%d.txt" % (dataset_type, epoch, local_rank)
            )
        else:
            file_path = os.path.join(
                dir_path,
                "%sed_sample_ids_epoch_%d.txt" % (dataset_type, epoch)
            )
        with open(file_path, "a+") as afp:
            for sample_id in sample_ids:
                afp.write("%s\n" % str(sample_id))
            print("Wrote %d into %s." % (size, file_path))


def calc_emb_filename_by_seq_id(seq_id, embedding_type):
    """
    根据seq_id得到emb_filename
    :param seq_id:
    :param embedding_type:
    :return:
    """
    seq_id = str(seq_id)
    if seq_id[0] == ">":
        seq_id = seq_id[1:]
    if "|" in seq_id:
        strs = seq_id.split("|")
        if len(strs) > 1:
            emb_filename = embedding_type + "_" + strs[1].strip() + ".pt"
        else:
            emb_filename = embedding_type + "_" + seq_id.replace(" ", "").replace("/", "_") + ".pt"
    else:
        emb_filename = embedding_type + "_" + seq_id.replace(" ", "").replace("/", "_") + ".pt"
    return emb_filename


def download_file(url, local_filename):
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        dir_name = os.path.dirname(local_filename)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        with open(local_filename, 'wb') as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)
    return local_filename


def download_folder(base_url, file_names, local_dir):
    if not os.path.exists(local_dir):
        os.makedirs(local_dir)

    for file_name in file_names:
        print(f"Downloading {file_name}...")
        file_url = f"{base_url}/{file_name}"
        local_filename = os.path.join(local_dir, file_name)
        download_file(file_url, local_filename)
        print(f"Downloaded {file_name}")


def download_trained_checkpoint_lucaone(
        llm_dir,
        llm_type="lucaone_gplm",
        llm_version="v2.0",
        llm_task_level="token_level,span_level,seq_level,structure_level",
        llm_time_str="20231125113045",
        llm_step="17600000",
        base_url="http://47.93.21.181/lucaone/TrainedCheckPoint"
):
    try:
        logs_file_names = ["logs.txt"]
        models_file_names = ["config.json", "pytorch.pth", "training_args.bin", "tokenizer/alphabet.pkl"]
        logs_path = "logs/lucagplm/%s/%s/%s/%s" % (llm_version, llm_task_level, llm_type, llm_time_str)
        models_path = "models/lucagplm/%s/%s/%s/%s/checkpoint-step%s" % (llm_version, llm_task_level, llm_type, llm_time_str, llm_step)
        logs_local_dir = os.path.join(llm_dir, logs_path)
        print("llm_dir: %s" % llm_dir)
        print("logs_local_dir: %s" % logs_local_dir)

        exists = True
        for logs_file_name in logs_file_names:
            filepath = os.path.join(logs_local_dir, logs_file_name)
            if not os.path.exists(filepath):
                exists = False
                break
            else:
                print("file: %s exists: %s." % (logs_file_name, filepath))
        models_local_dir = os.path.join(llm_dir, models_path)
        print("models_local_dir: %s" % models_local_dir)

        if exists:
            for models_file_name in models_file_names:
                filepath = os.path.join(models_local_dir, models_file_name)
                if not os.path.exists(filepath):
                    exists = False
                    break
                else:
                    print("file: %s exists: %s." % (models_file_name, filepath))
        if not exists:
            print("*" * 20 + "Downloading" + "*" * 20)
            print("Downloading LucaOne TrainedCheckPoint: LucaOne-%s-%s-%s ..." % (llm_version, llm_time_str, llm_step))
            print("Wait a moment(total 8GB), please.")
            # download logs
            if not os.path.exists(logs_local_dir):
                os.makedirs(logs_local_dir)
            logs_base_url = os.path.join(base_url, logs_path)
            download_folder(logs_base_url, logs_file_names, logs_local_dir)
            # download models
            if not os.path.exists(models_local_dir):
                os.makedirs(models_local_dir)
            models_base_url = os.path.join(base_url, models_path)
            download_folder(models_base_url, models_file_names, models_local_dir)
            print("LucaOne Downloaded.")
            print("*" * 50)
    except Exception as e:
        print(e)
        print("Download automatically LucaOne Trained CheckPoint failed!")
        print("You can manually download 'logs/' and 'models/' into local directory: %s/ from %s" % (os.path.abspath(llm_dir), os.path.join(base_url, "TrainedCheckPoint/")))
        raise Exception(e)