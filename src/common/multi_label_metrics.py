#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.**@**.com
@tel: 137****6540
@datetime: 2022/11/26 21:05
@project: LucaOne
@file: multi_label_metrics
@desc: metrics for multi-label classification
'''
import csv
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, average_precision_score


def multi_label_acc(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    acc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            acc_list.append(1.0)
        else:
            # acc
            acc = 1.0 - (union_len - intersection_len) / targets.shape[1]
            acc_list.append(acc)
    return round(sum(acc_list)/len(acc_list), 6) if len(acc_list) > 0 else 0


def multi_label_precision(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    prec_list = []

    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            prec_list.append(1.0)
        else:
            # precision
            prec = 0.0
            if predict_len > 0:
                prec = intersection_len / predict_len
            prec_list.append(prec)

    round(sum(prec_list)/len(prec_list), 6) if len(prec_list) > 0 else 0


def multi_label_recall(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    recall_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            recall_list.append(1.0)
        else:
            # recall
            if target_len > 0:
                recall = intersection_len / target_len
            else:
                recall = 1.0
            recall_list.append(recall)
    return round(sum(recall_list)/len(recall_list), 6) if len(recall_list) > 0 else 0


def multi_label_jaccard(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    jaccard_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            jaccard_list.append(1.0)
        else:
            # jaccard sim
            jac = intersection_len / union_len
            jaccard_list.append(jac)
    return round(sum(jaccard_list)/len(jaccard_list), 6) if len(jaccard_list) > 0 else 0


def multi_label_f1(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    f1_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            f1_list.append(1.0)
        else:
            # precision
            prec = 0.0

            # recall
            if target_len > 0:
                recall = intersection_len / target_len
            else:
                recall = 1.0
            # f1
            if prec + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * prec * recall / (prec + recall)
            f1_list.append(f1)
    return round(sum(f1_list)/len(f1_list), 6) if len(f1_list) > 0 else 0


def multi_label_roc_auc(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    roc_auc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        if union_len == 0:
            roc_auc_list.append(1.0)
        else:
            # roc_auc
            if len(np.unique(targets[idx, :])) > 1:
                roc_auc = roc_auc_macro(targets[idx, :], probs[idx, :])
                roc_auc_list.append(roc_auc)
    return round(sum(roc_auc_list)/len(roc_auc_list), 6) if len(roc_auc_list) > 0 else 0


def multi_label_pr_auc(targets, probs, threshold=0.5):
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    pr_auc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        if union_len == 0:
            pr_auc_list.append(1.0)
        else:
            # roc_auc
            if len(np.unique(targets[idx, :])) > 1:

                pr_auc = pr_auc_macro(targets[idx, :], probs[idx, :])
                pr_auc_list.append(pr_auc)

    return round(sum(pr_auc_list)/len(pr_auc_list), 6) if len(pr_auc_list) > 0 else 0


def metrics_multi_label(targets,  probs, threshold=0.5):
    '''
    metrics of multi-label classification
    cal metrics for true matrix to predict probability matrix
    :param targets: true 0-1 indicator matrix (n_samples, n_labels)
    :param probs: probs 0~1 probability matrix (n_samples, n_labels)
    :param thresold: negative-positive threshold
    :return: some metrics
    '''
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes((probs >= threshold).astype(int))
    acc_list = []
    prec_list = []
    recall_list = []
    jaccard_list = []
    f1_list = []
    roc_auc_list = []
    pr_auc_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]
        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        if union_len == 0:
            acc_list.append(1.0)
            prec_list.append(1.0)
            recall_list.append(1.0)
            roc_auc_list.append(1.0)
            jaccard_list.append(1.0)
            f1_list.append(1.0)
            pr_auc_list.append(1.0)
        else:
            # acc
            acc = 1.0 - (union_len - intersection_len) / targets.shape[1]
            acc_list.append(acc)

            # precision
            prec = 0.0
            if predict_len > 0:
                prec = intersection_len / predict_len
            prec_list.append(prec)

            # recall
            if target_len > 0:
                recall = intersection_len / target_len
            else:
                recall = 1.0
            recall_list.append(recall)

            # jaccard sim
            jac = intersection_len / union_len
            jaccard_list.append(jac)

            # f1
            if prec + recall == 0:
                f1 = 0.0
            else:
                f1 = 2.0 * prec * recall / (prec + recall)
            f1_list.append(f1)

            # roc_auc
            if len(np.unique(targets[idx, :])) > 1:
                roc_auc = roc_auc_macro(targets[idx, :], probs[idx, :])
                roc_auc_list.append(roc_auc)
                pr_auc = pr_auc_macro(targets[idx, :], probs[idx, :])
                pr_auc_list.append(pr_auc)

    f_max_value, p_max_value, r_max_value, t_max_value, preds_max_value = f_max(targets, probs)
    return {
        "acc": round(float(sum(acc_list)/len(acc_list)), 6) if len(acc_list) > 0 else 0,
        "jaccard": round(float(sum(jaccard_list)/len(jaccard_list)), 6) if len(jaccard_list) > 0 else 0,
        "prec": round(float(sum(prec_list)/len(prec_list)), 6) if len(prec_list) > 0 else 0,
        "recall": round(float(sum(recall_list)/len(recall_list)), 6) if len(recall_list) > 0 else 0,
        "f1": round(float(sum(f1_list)/len(f1_list)), 6) if len(f1_list) > 0 else 0,
        "pr_auc": round(float(sum(pr_auc_list)/len(pr_auc_list)), 6) if len(pr_auc_list) > 0 else 0,
        "roc_auc": round(float(sum(roc_auc_list)/len(roc_auc_list)), 6) if len(roc_auc_list) > 0 else 0,
        "fmax": round(float(f_max_value), 6),
        "pmax": round(float(p_max_value), 6) ,
        "rmax": round(float(r_max_value), 6),
        "tmax": round(float(t_max_value), 6)
    }


def f_max(targets, probs, gos=None):
    '''
    f-max for multi-label classification
    :param targets: true 0-1 indicator matrix (n_samples, n_labels)
    :param probs: probs 0~1 probability matrix (n_samples, n_labels)
    :param gos:
    :return: fmax, p_max(precision max）, r_max（recall max）, t_max（classificaton threshold）, preds_max（0-1 indicator matrix)
    '''
    preds_max = None
    f_max = 0
    p_max = 0
    r_max = 0
    t_max = 0
    # from 0.01 to 1 (100 thresholds)
    for tt in range(1, 101):
        threshold = tt / 100.0
        preds = (probs > threshold).astype(np.int32)
        p = 0.0
        r = 0.0
        total = 0
        p_total = 0
        for i in range(preds.shape[0]):
            tp = np.sum(preds[i, :] * targets[i, :])
            fp = np.sum(preds[i, :]) - tp
            fn = np.sum(targets[i, :]) - tp
            if gos:
                fn += gos[i]

            if tp == 0 and fp == 0 and fn == 0:
                continue
            total += 1
            if tp != 0:
                p_total += 1
                precision = tp / (1.0 * (tp + fp))
                recall = tp / (1.0 * (tp + fn))
                p += precision
                r += recall

        if total > 0 and p_total > 0:
            r /= total
            p /= p_total
            if p + r > 0:
                f = 2 * p * r / (p + r)
                if f_max < f:
                    f_max = f
                    p_max = p
                    r_max = r
                    t_max = threshold
                    preds_max = preds

    return f_max, p_max, r_max, t_max, preds_max


def metrics_multi_label_for_pred(targets,  preds, savepath=None):
    '''
    metrics for multi-label classification
    cal metrics for true matrix to predict
    :param targets: true 0-1 indicator matrix (n_samples, n_labels)
    :param preds: preds 0~1 indicator matrix  (n_samples, n_labels)
    :return: some metrics
    '''
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes(preds)
    acc_list = []
    prec_list = []
    recall_list = []
    jaccard_list = []
    f1_list = []
    for idx in range(targets.shape[0]):
        target_relevant = targets_relevant[idx]
        pred_relevant = preds_relevant[idx]

        target_len = len(target_relevant)
        predict_len = len(pred_relevant)
        union_len = len(set(target_relevant).union(set(pred_relevant)))
        intersection_len = len(set(target_relevant).intersection(set(pred_relevant)))
        acc = 1.0 - (union_len - intersection_len) / targets.shape[1]
        prec = 0.0
        if predict_len > 0:
            prec = intersection_len / predict_len
        recall = 0
        if target_len > 0:
            recall = intersection_len / target_len
        else:
            print(targets[idx])
        jac = intersection_len / union_len
        if prec + recall == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * recall / (prec + recall)

        acc_list.append(acc)
        prec_list.append(prec)
        recall_list.append(recall)
        jaccard_list.append(jac)
        f1_list.append(f1)

    return {
        "acc": round(sum(acc_list)/targets.shape[0], 6),
        "jaccard": round(sum(jaccard_list)/targets.shape[0], 6),
        "prec": round(sum(prec_list)/targets.shape[0], 6),
        "recall": round(sum(recall_list)/targets.shape[0], 6),
        "f1": round(sum(f1_list)/targets.shape[0], 6)
    }


def label_id_2_array(label_ids, label_size):
    '''
    building 0-1 indicator array for multi-label classification
    :param label_ids:
    :param label_size:
    :return:
    '''
    arr = np.zeros(label_size)
    arr[label_ids] = 1
    return arr


def relevant_indexes(matrix):
    '''
    Which positions in the multi-label are labeled as 1
    :param matrix:
    :return:
    '''
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()
    relevants = []
    shape = matrix.shape
    if matrix.ndim == 3:

        for x in range(shape[0]):
            relevant_x = []
            for y in range(shape[1]):
                relevant_y = []
                for z in range(shape[2]):
                    if matrix[x, y, z] == 1:
                        relevant_y.append(int(z))
                relevant_x.append(relevant_y)
            relevants.append(relevant_x)
    elif matrix.ndim == 2:
        for row in range(shape[0]):
            relevant = []
            for col in range(shape[1]):
                if matrix[row, col] == 1:
                    relevant.append(int(col))
            relevants.append(relevant)
    else:
        for idx in range(matrix.shape[0]):
            if matrix[idx] == 1:
                relevants.append(int(idx))
    return relevants


def irrelevant_indexes(matrix):
    '''
    Which positions in the multi-label label are 0
    :param matrix:
    :return:
    '''
    if torch.is_tensor(matrix):
        matrix = matrix.detach().cpu().numpy()

    irrelevants = []
    if matrix.ndim == 3:
        for x in range(matrix.shape[0]):
            irrelevant_x = []
            for y in range(matrix.shape[1]):
                irrelevant_y = []
                for z in range(matrix.shape[2]):
                    if matrix[x, y, z] == 0:
                        irrelevant_y.append(int(z))
                irrelevant_x.append(irrelevant_y)
            irrelevants.append(irrelevant_x)
    elif matrix.ndim == 2:
        for row in range(matrix.shape[0]):
            irrelevant = []
            for col in range(matrix.shape[1]):
                if matrix[row, col] == 1:
                    irrelevant.append(int(col))
            irrelevants.append(irrelevant)
    else:
        for idx in range(matrix.shape[0]):
            if matrix[idx] == 1:
                irrelevants.append(int(idx))

    return irrelevants


def prob_2_pred(prob, threshold):
    '''
    Probabilities converted to 0-1 predicted labels
    :param prob:
    :param threshold:
    :return:
    '''
    if torch.is_tensor(prob):
        prob = prob.detach().cpu().numpy()

    if isinstance(prob, (np.ndarray, np.generic)):
        return (prob >= threshold).astype(int)


def roc_auc_macro(target, prob):
    '''
    macro roc auc
    :param target:
    :param prob:
    :return:
    '''
    return roc_auc_score(target, prob, average="macro")


def pr_auc_macro(target, prob):
    '''
    macro pr-auc
    :param target:
    :param prob:
    :return:
    '''
    return average_precision_score(target, prob, average="macro")


def write_error_samples_multi_label(filepath, samples, input_indexs, input_id_2_names, output_id_2_name, targets,
                                    probs, threshold=0.5,
                                    use_other_diags=False, use_other_operas=False, use_checkin_department=False):
    '''
    writer bad cases for multi-label classification
    :param filepath:
    :param samples:
    :param input_indexs:
    :param input_id_2_names:
    :param output_id_2_name:
    :param targets:
    :param probs:
    :param threshold:
    :param use_other_diags:
    :param use_other_operas:
    :param use_checkin_department:
    :return:
    '''
    preds = prob_2_pred(probs, threshold=threshold)
    targets_relevant = relevant_indexes(targets)
    preds_relevant = relevant_indexes(preds)
    with open(filepath, "w") as fp:
        writer = csv.writer(fp)
        writer.writerow(["score", "y_true", "y_pred", "inputs"])
        for i in range(len(targets_relevant)):
            target = set(targets_relevant[i])
            pred = set(preds_relevant[i])
            jacc = len(target.intersection(pred))/(len(target.union(pred)))
            if output_id_2_name:
                target_labels = [output_id_2_name[v] for v in target]
                pred_labels = [output_id_2_name[v] for v in pred]
            else:
                target_labels = target
                pred_labels = pred
            sample = samples[i]
            if input_id_2_names:
                new_sample = []
                for idx, input_index in enumerate(input_indexs):
                    if input_index == 3 and not use_checkin_department:
                        input_index = 12
                    new_sample.append([input_id_2_names[idx][v] for v in sample[input_index]])
                    if input_index == 6 and use_other_diags or input_index == 8 and use_other_operas or input_index == 10 and use_other_diags:
                        new_sample.append([input_id_2_names[idx][v] for v in sample[input_index + 1]])
            else:
                new_sample = sample
            row = [jacc, target_labels, pred_labels, new_sample]
            writer.writerow(row)



