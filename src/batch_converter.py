#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/24 15:14
@project: LucaOne
@file: batch_converter
@desc: batch converter
'''
import random
import torch
from typing import Sequence


class BatchConverter(object):
    def __init__(
            self,
            alphabet,
            no_position_embeddings,
            no_token_type_embeddings,
            truncation_seq_length: int = None,
            ignore_index: int = -100,
            mlm_probability=0.15
    ):
        self.alphabet = alphabet
        self.ignore_index = ignore_index
        self.mlm_probability = mlm_probability
        self.no_position_embeddings = no_position_embeddings
        self.truncation_seq_length = truncation_seq_length
        append_len = int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
        self.truncation_seq_length -= append_len
        self.no_token_type_embeddings = no_token_type_embeddings

    def __process_label__(self, max_length, labels):
        encoded_labels = {}
        for label in labels:
            for item1 in label.items():
                level1_name = item1[0]
                if level1_name not in encoded_labels:
                    encoded_labels[level1_name] = {}
                for item2 in item1[1].items():
                    level2_name = item2[0]
                    if level2_name not in encoded_labels[level1_name]:
                        encoded_labels[level1_name][level2_name] = []
                    values = item2[1]
                    encoded_labels[level1_name][level2_name].append(values)
        new_encoded_labels = {}
        for item1 in encoded_labels.items():
            level1_name = item1[0]
            if level1_name not in new_encoded_labels:
                new_encoded_labels[level1_name] = {}
            for item2 in item1[1].items():
                level2_name = item2[0]
                if level2_name not in new_encoded_labels[level1_name]:
                    # padding
                    if level1_name == "structure_level" and level2_name == "prot_structure":
                        new_items = []
                        for v in item2[1]:
                            cur_len = len(v)
                            if cur_len < max_length:
                                new_items.append(v + [[self.ignore_index, self.ignore_index, self.ignore_index]] * (max_length - cur_len))
                            else:
                                new_items.append(v)
                        new_encoded_labels[level1_name][level2_name] = torch.tensor(new_items, dtype=torch.float32)
                    elif level1_name == "structure_level" and level2_name in ["prot_secondary", "prot_contact"]:
                        # padding
                        new_items = []
                        for v in item2[1]:
                            cur_len = len(v)
                            if cur_len < max_length:
                                new_items.append(v + [self.ignore_index] * (max_length - cur_len))
                            else:
                                new_items.append(v)
                        new_encoded_labels[level1_name][level2_name] = torch.tensor(new_items, dtype=torch.int64)
                    elif level1_name == "token_level" and level2_name in ["gene_mask", "prot_mask"]:
                        new_encoded_labels[level1_name][level2_name] = torch.stack(item2[1], dim=0)
                    elif level1_name == "span_level":
                        # padding
                        new_items = []
                        for v in item2[1]:
                            cur_len = len(v)
                            if cur_len < max_length:
                                new_items.append(v + [self.ignore_index] * (max_length - cur_len))
                            else:
                                new_items.append(v)
                        new_encoded_labels[level1_name][level2_name] = torch.tensor(new_items, dtype=torch.int64)
                    elif level1_name == "seq_level":
                        new_encoded_labels[level1_name][level2_name] = torch.tensor(item2[1], dtype=torch.int64)
                    else:
                        raise Exception("not support task_level=%s" % level1_name)
        return new_encoded_labels

    def __mask_tokens__(self, input_ids, seq_len):
        labels = input_ids.clone()
        probability_matrix = torch.full(labels.shape, self.mlm_probability)

        # 特殊字符处为1
        special_tokens_mask = [
            1 if v in self.alphabet.all_special_token_idx_list else 0 for v in labels.tolist()
        ]
        special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        # 将特殊字符处填充为0.0
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)

        # 非特殊字符的位置
        masked_indices = torch.bernoulli(probability_matrix).bool()
        # 特殊字符处为-100
        labels[~masked_indices] = self.ignore_index  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with alphabet.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.alphabet.mask_idx

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.alphabet), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        if torch.any(labels != self.ignore_index):
            return input_ids, labels
        else:
            # non [MASK]， random one position, convect to [MASK]
            rand_idx = random.randint(int(self.alphabet.prepend_bos), seq_len + int(self.alphabet.prepend_bos) - 1)
            labels[rand_idx] = input_ids[rand_idx]
            input_ids[rand_idx] = self.alphabet.mask_idx
            return input_ids, labels

    def __call_single__(self, batch_size, seq_types, seqs, seq_labels):
        seq_encoded_list = [self.alphabet.encode(seq_type=seq_type, seq=seq_str.upper()) for seq_type, seq_str in zip(seq_types, seqs)]
        batch_size = min(batch_size, len(seq_encoded_list))
        if self.truncation_seq_length:
            seq_encoded_list = [encoded[:self.truncation_seq_length] for encoded in seq_encoded_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        max_len = max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
        # for input
        input_ids = torch.empty(
            (
                batch_size,
                max_len
            ),
            dtype=torch.int64,
        )
        input_ids.fill_(self.alphabet.padding_idx)

        position_ids = None
        if not self.no_position_embeddings:
            position_ids = torch.empty(
                (
                    batch_size,
                    max_len
                ),
                dtype=torch.int64,
            )
            position_ids.fill_(self.alphabet.padding_idx)

        token_type_ids = None
        if not self.no_token_type_embeddings:
            token_type_ids = torch.empty(
                (
                    batch_size,
                    max_len
                ),
                dtype=torch.int64,
            )
            token_type_ids.fill_(self.alphabet.padding_idx)

        strs = []
        labels = []
        for i, (label, seq_type, seq_str, seq_encoded) in enumerate(
                zip(seq_labels, seq_types, seqs, seq_encoded_list)
        ):
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                input_ids[i, 0] = self.alphabet.cls_idx
            seq_len = len(seq_encoded)
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            input_ids[i, int(self.alphabet.prepend_bos): len(seq_encoded) + int(self.alphabet.prepend_bos)] = seq
            if self.alphabet.append_eos:
                input_ids[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx

            if not self.no_position_embeddings:
                cur_len = int(self.alphabet.prepend_bos) + len(seq_encoded) + int(self.alphabet.append_eos)
                for idx in range(0, cur_len):
                    position_ids[i, idx] = idx
            if not self.no_token_type_embeddings:
                if seq_type == "gene":
                    type_value = 0
                else:
                    type_value = 1
                cur_len = int(self.alphabet.prepend_bos) + len(seq_encoded) + int(self.alphabet.append_eos)
                for idx in range(0, cur_len):
                    token_type_ids[i, idx] = type_value
            # labels
            if label is None:
                label = {}
            if "token_level" not in label:
                label["token_level"] = {}

            if seq_type == "gene":
                input_ids[i, :], label["token_level"]["gene_mask"] = self.__mask_tokens__(
                    input_ids[i, :],
                    seq_len=seq_len
                )
            else:
                input_ids[i, :], label["token_level"]["prot_mask"] = self.__mask_tokens__(
                    input_ids[i, :],
                    seq_len=seq_len
                )
            labels.append(label)
        return self.__process_label__(max_len, labels), strs, input_ids, position_ids, token_type_ids, max_len

    def __call_single_for_eval_mask__(
            self,
            batch_size,
            seq_types,
            seqs,
            seq_labels
    ):
        seq_encoded_list = [self.alphabet.encode_for_eval_mask(
            seq_type=seq_type, seq=seq_str.upper()) for seq_type, seq_str in zip(seq_types, seqs)
        ]
        seq_mask_label_list = [
            self.alphabet.encode(seq_type=seq_types[idx], seq=label["token_level"]["gene_mask"].upper())
            if seq_labels[idx] == "gene" else self.alphabet.encode(seq_type=seq_types[idx], seq=label["token_level"]["prot_mask"].upper())
            for idx, label in enumerate(seq_labels)
        ]
        batch_size = min(batch_size, len(seq_encoded_list))
        if self.truncation_seq_length:
            seq_encoded_list = [encoded[:self.truncation_seq_length] for encoded in seq_encoded_list]
            seq_mask_label_list = [encoded[:self.truncation_seq_length] for encoded in seq_mask_label_list]
        max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
        max_len = max_len + int(self.alphabet.prepend_bos) + int(self.alphabet.append_eos)
        # for input
        input_ids = torch.empty(
            (
                batch_size,
                max_len
            ),
            dtype=torch.int64,
        )
        input_ids.fill_(self.alphabet.padding_idx)

        seq_mask_label_input_ids = torch.empty(
            (
                batch_size,
                max_len
            ),
            dtype=torch.int64,
        )
        seq_mask_label_input_ids.fill_(self.alphabet.padding_idx)

        position_ids = None
        if not self.no_position_embeddings:
            position_ids = torch.empty(
                (
                    batch_size,
                    max_len
                ),
                dtype=torch.int64,
            )
            position_ids.fill_(self.alphabet.padding_idx)

        token_type_ids = None
        if not self.no_token_type_embeddings:
            token_type_ids = torch.empty(
                (
                    batch_size,
                    max_len
                ),
                dtype=torch.int64,
            )
            token_type_ids.fill_(self.alphabet.padding_idx)

        strs = []
        labels = []
        for i, (seq_mask_label_encoded, seq_type, seq_str, seq_encoded) in enumerate(
                zip(seq_mask_label_list, seq_types, seqs, seq_encoded_list)
        ):
            label = None
            assert len(seq_mask_label_encoded) == len(seq_encoded)
            strs.append(seq_str)
            if self.alphabet.prepend_bos:
                input_ids[i, 0] = self.alphabet.cls_idx
                seq_mask_label_input_ids[i, 0] = self.ignore_index
            seq = torch.tensor(seq_encoded, dtype=torch.int64)
            input_ids[i, int(self.alphabet.prepend_bos): len(seq_encoded) + int(self.alphabet.prepend_bos)] = seq

            seq_mask_label = torch.tensor(seq_mask_label_encoded, dtype=torch.int64)
            seq_mask_label_input_ids[i, int(self.alphabet.prepend_bos): len(seq_mask_label_encoded) + int(self.alphabet.prepend_bos)] = seq_mask_label

            if self.alphabet.append_eos:
                input_ids[i, len(seq_encoded) + int(self.alphabet.prepend_bos)] = self.alphabet.eos_idx
                seq_mask_label_input_ids[i, len(seq_mask_label_encoded) + int(self.alphabet.prepend_bos)] = self.ignore_index

            if not self.no_position_embeddings:
                cur_len = int(self.alphabet.prepend_bos) + len(seq_encoded) + int(self.alphabet.append_eos)
                for idx in range(0, cur_len):
                    position_ids[i, idx] = idx
            if not self.no_token_type_embeddings:
                if seq_type == "gene":
                    type_value = 0
                else:
                    type_value = 1
                cur_len = int(self.alphabet.prepend_bos) + len(seq_encoded) + int(self.alphabet.append_eos)
                for idx in range(0, cur_len):
                    token_type_ids[i, idx] = type_value
            # labels
            if label is None:
                label = {}
            if "token_level" not in label:
                label["token_level"] = {}

            mask_position = input_ids[i, :] == self.alphabet.mask_idx
            seq_mask_label_input_ids[i, mask_position] = self.ignore_index
            if seq_type == "gene":
                label["token_level"]["gene_mask"] = seq_mask_label_input_ids
            else:
                label["token_level"]["prot_mask"] = seq_mask_label_input_ids
            labels.append(label)
        return self.__process_label__(
            max_len,
            labels
        ), strs, input_ids, position_ids, token_type_ids, max_len

    def __call__(self, raw_batch: Sequence[dict]):
        batch_size = len(raw_batch)
        sample_ids = []
        # gene-prot pair
        if "gene_seq" in raw_batch[0] or "prot_seq" in raw_batch[0]:
            res = {}
            seq_types = []
            seqs = []
            seq_labels = []
            seq_types_b = []
            seqs_b = []
            seq_labels_b = []
            pair_labels = []
            for item in raw_batch:
                if "sample_id" in item:
                    sample_ids.append(item["sample_id"])
                else:
                    sample_ids.append("unknown")
                if "gene_seq" in item:
                    seq_types.append("gene")
                    seqs.append(item["gene_seq"])
                    seq_labels.append(item["gene_labels"])
                if "prot_seq" in item:
                    seq_types_b.append("prot")
                    seqs_b.append(item["prot_seq"])
                    seq_labels_b.append(item["prot_labels"])
                if "pair" in item and "pair_label" in item["pair"]:
                    pair_labels.append(item["pair"]["pair_label"])
            labels, strs, input_ids, position_ids, token_type_ids, max_len = None, None, None, None, None, None
            if seq_types is not None and len(seq_types) > 0:
                labels, strs, input_ids, position_ids, token_type_ids, max_len = self.__call_single__(
                    batch_size,
                    seq_types,
                    seqs,
                    seq_labels
                )
                res.update({
                    "input_ids": input_ids,
                    "position_ids": position_ids,
                    "token_type_ids": token_type_ids,
                    "labels": labels
                })
            labels_b, strs_b, input_ids_b, position_ids_b, token_type_ids_b, max_len_b = None, None, None, None, None, None
            if seq_types_b is not None and len(seq_types_b) > 0:
                labels_b, strs_b, input_ids_b, position_ids_b, token_type_ids_b, max_len_b = self.__call_single__(
                    batch_size,
                    seq_types_b,
                    seqs_b,
                    seq_labels_b
                )
                res.update({
                    "input_ids_b": input_ids_b,
                    "position_ids_b": position_ids_b,
                    "token_type_ids_b": token_type_ids_b,
                    "labels_b": labels_b
                })
            if max_len and max_len_b and pair_labels:
                pair_labels = self.__process_label__(max(max_len, max_len_b), pair_labels)
                res.update({
                    "pair_labels": pair_labels
                })
            res["sample_ids"] = sample_ids
            return res
        else:
            seq_types = []
            seqs = []
            seq_labels = []
            for item in raw_batch:
                if "sample_id" in item:
                    sample_ids.append(item["sample_id"])
                else:
                    sample_ids.append("unknown")
                seq_types.append(item["obj_type"])
                seqs.append(item["obj_seq"])
                seq_labels.append(item["obj_labels"])

            labels, strs, input_ids, position_ids, token_type_ids, max_len = self.__call_single__(
                batch_size,
                seq_types,
                seqs,
                seq_labels
            )
            return {
                "sample_ids": sample_ids,
                "input_ids": input_ids,
                "position_ids": position_ids,
                "token_type_ids": token_type_ids,
                "labels": labels
            }