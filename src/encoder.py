#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/21 16:10
@project: LucaOne
@file: encoder
@desc: encoder
'''
import sys
sys.path.append(".")
sys.path.append("../")
sys.path.append("../src")
try:
    from utils import gene_seq_replace, re_positional, gene_seq_replace_re
except ImportError:
    from src.utils import gene_seq_replace, re_positional, gene_seq_replace_re


class Encoder(object):
    def __init__(self,
                 config,
                 tokenizer,
                 tokenization,
                 no_token_type_embeddings,
                 non_ignore,
                 ignore_index,
                 model_type,
                 special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
                 max_coord_x=None,
                 min_coord_x=None,
                 max_coord_y=None,
                 min_coord_y=None,
                 max_coord_z=None,
                 min_coord_z=None
                 ):
        self.config = config

        self.tokenization = tokenization
        self.tokenizer = tokenizer
        self.ignore_index = ignore_index
        self.non_ignore = non_ignore
        self.model_type = model_type
        self.special_tokens = special_tokens
        self.no_token_type_embeddings = no_token_type_embeddings
        self.max_coord_x = 1307.909 if max_coord_x is None else max_coord_x
        self.min_coord_x = -698.673 if min_coord_x is None else min_coord_x
        self.max_coord_y = 1290.457 if max_coord_y is None else max_coord_y
        self.min_coord_y = -963.585 if min_coord_y is None else min_coord_y
        self.max_coord_z = 3017.753 if max_coord_z is None else max_coord_z
        self.min_coord_z = -911.566 if min_coord_z is None else min_coord_z
        assert "add_special_tokens" in self.config and "max_length" in config and "truncation" in config
        self.add_special_tokens = self.config["add_special_tokens"]
        self.max_length = int(self.config["max_length"])
        self.truncation = self.config["truncation"]

    def __encode__(self, pretrain_task_level_type, seq, label, label_size_dict, output_mode_dict):
        seq = seq.upper()
        if self.tokenization:
            # 重定义位置
            tokens, label = re_positional(seq,
                                          None,
                                          self.tokenizer,
                                          self.special_tokens,
                                          label,
                                          ignore_index=self.ignore_index)
            # seq to seq ids
            encoding = self.tokenizer.encode_plus(text=seq,
                                                  text_pair=None,
                                                  add_special_tokens=self.add_special_tokens,
                                                  padding="max_length",
                                                  max_length=self.max_length,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=not self.no_token_type_embeddings,
                                                  return_length=False,
                                                  truncation=True)
            L = sum(encoding["attention_mask"])
        else:
            cur_max_length = self.max_length
            if self.add_special_tokens:
                cur_max_length = cur_max_length - 2
            if len(seq) > cur_max_length:
                if self.truncation == "right":
                    seq = seq[:cur_max_length]
                elif self.truncation == "left":
                    seq = seq[-cur_max_length:]
                else:
                    raise Exception("truncation = %s" % self.truncation)
                L = cur_max_length + 2 if self.add_special_tokens else cur_max_length
            else:
                L = len(seq) + 2 if self.add_special_tokens else len(seq)
            seq = " ".join(list(seq))
            encoding = self.tokenizer.encode_plus(text=seq,
                                                  text_pair=None,
                                                  add_special_tokens=self.add_special_tokens,
                                                  padding="max_length",
                                                  max_length=self.max_length,
                                                  return_attention_mask=True,
                                                  return_token_type_ids=not self.no_token_type_embeddings,
                                                  return_length=False,
                                                  truncation=True)

        if label:
            encoding["labels"] = {}
            encoding["labels"].update(self.__parse_label__(pretrain_task_level_type, label, L, label_size_dict, output_mode_dict))
        return encoding, L

    def __encode_char__(self, pretrain_task_level_type, seq, label, label_size_dict, output_mode_dict):
        encoding = {}
        seq = seq.upper()
        encoding["ori_seq"] = seq
        if self.tokenization:
            raise Exception("not support tokenization")
        else:
            cur_max_length = self.max_length
            if self.add_special_tokens:
                cur_max_length = cur_max_length - 2
            if len(seq) > cur_max_length:
                if self.truncation == "right":
                    seq = seq[:cur_max_length]
                elif self.truncation == "left":
                    seq = seq[-cur_max_length:]
                else:
                    raise Exception("truncation = %s" % self.truncation)
                L = cur_max_length + 2 if self.add_special_tokens else cur_max_length
            else:
                L = len(seq) + 2 if self.add_special_tokens else len(seq)
            encoding["seq"] = seq
        if label:
            encoding["labels"] = {}
            encoding["labels"].update(self.__parse_label__(pretrain_task_level_type, label, L, label_size_dict, output_mode_dict))
        return encoding, L

    def __parse_label__(self, pretrain_task_level_type, label, seq_length, label_size_dict, output_mode_dict):
        if isinstance(label, str):
            label = eval(label)
        res = {}
        if "all" in pretrain_task_level_type or "span_level" in pretrain_task_level_type:
            if "span_level" in label and label["span_level"]:
                res["span_level"] = {}
                for span_level_item in label["span_level"].items():
                    span_level_key = span_level_item[0]
                    span_level_labels = span_level_item[1]
                    cur_label_size = label_size_dict["span_level"][span_level_key]
                    cur_output_mode = output_mode_dict["span_level"][span_level_key]
                    if cur_output_mode in ["multi_label", "multi-label"]:
                        new_span_level_labels = []
                        if self.non_ignore and span_level_key in self.non_ignore:
                            for _ in range(seq_length):
                                tmp = []
                                for _ in range(cur_label_size):
                                    tmp.append(0)
                                new_span_level_labels.append(tmp)
                            # new_span_level_labels = [[0] * cur_label_size] * seq_length
                        else:
                            for _ in range(seq_length):
                                tmp = []
                                for _ in range(cur_label_size):
                                    tmp.append(self.ignore_index)
                                new_span_level_labels.append(tmp)
                            # new_span_level_labels = [[self.ignore_index] * cur_label_size] * seq_length
                    else:
                        new_span_level_labels = []
                        if self.non_ignore and span_level_key in self.non_ignore:
                            # new_span_level_labels = [0] * seq_length
                            for _ in range(seq_length):
                                new_span_level_labels.append(0)
                        else:
                            for _ in range(seq_length):
                                new_span_level_labels.append(self.ignore_index)
                            # new_span_level_labels = [self.ignore_index] * seq_length
                    if span_level_labels is not None and len(span_level_labels) > 0:
                        begin_idx = 0
                        end_idx = seq_length
                        if self.add_special_tokens:
                            begin_idx = 1
                            end_idx = seq_length - 1
                        for item in span_level_labels:
                            for idx in range(item[0], item[1] + 1, 1):
                                idx += begin_idx
                                if idx >= end_idx:
                                    break
                                if cur_output_mode in ["multi_label", "multi-label"]:
                                    new_span_level_labels[idx][item[2]] = 1
                                else:
                                    new_span_level_labels[idx] = item[2]
                    res["span_level"] .update({span_level_key: new_span_level_labels})

        if "all" in pretrain_task_level_type or "structure_level" in pretrain_task_level_type:
            if "structure_level" in label and label["structure_level"]:
                res["structure_level"] = {}
                for structure_level_item in label["structure_level"].items():
                    structure_level_key = structure_level_item[0]
                    if structure_level_key == "prot_structure":
                        new_structure_level_labels = []
                        for _ in range(seq_length):
                            new_structure_level_labels.append([self.ignore_index, self.ignore_index, self.ignore_index])
                        # new_structure_level_labels = [[self.ignore_index, self.ignore_index, self.ignore_index]] * seq_length
                        begin_idx = 0
                        end_idx = seq_length
                        if self.add_special_tokens:
                            begin_idx = 1
                            end_idx = seq_length - 1
                        '''
                        for coord_idx, coord in enumerate(structure_level_item[1]):
                            coord_idx += begin_idx
                            if coord_idx >= end_idx:
                                break
                            if coord == -1 or (coord[0] == self.ignore_index and coord[1] == self.ignore_index and coord[2] == self.ignore_index):
                                continue
                            new_coord = [coord[0], coord[1], coord[2]]
                            new_coord[0] = (new_coord[0] - self.min_coord_x)/(self.max_coord_x - self.min_coord_x)
                            new_coord[1] = (new_coord[1] - self.min_coord_y)/(self.max_coord_y - self.min_coord_y)
                            new_coord[2] = (new_coord[2] - self.min_coord_z)/(self.max_coord_z - self.min_coord_z)
                            new_structure_level_labels[coord_idx] = new_coord
                        '''
                        local_x = [10000000, -10000000]
                        local_y = [10000000, -10000000]
                        local_z = [10000000, -10000000]
                        # protein-wise 归一
                        for coord_idx, coord in enumerate(structure_level_item[1]):
                            coord_idx += begin_idx
                            if coord_idx >= end_idx:
                                break
                            if coord == -1 or (coord[0] == self.ignore_index and coord[1] == self.ignore_index and coord[2] == self.ignore_index):
                                continue
                            x, y, z = coord[0], coord[1], coord[2]
                            if local_x[0] > x:
                                local_x[0] = x
                            if local_x[1] < x:
                                local_x[1] = x
                            if local_y[0] > y:
                                local_y[0] = y
                            if local_y[1] < y:
                                local_y[1] = y
                            if local_z[0] > z:
                                local_z[0] = z
                            if local_z[1] < z:
                                local_z[1] = z
                        for coord_idx, coord in enumerate(structure_level_item[1]):
                            coord_idx += begin_idx
                            if coord_idx >= end_idx:
                                break
                            if coord == -1 or (coord[0] == self.ignore_index and coord[1] == self.ignore_index and coord[2] == self.ignore_index):
                                continue
                            new_coord = [coord[0], coord[1], coord[2]]
                            if local_x[0] == local_x[1]:
                                new_coord[0] = 1.0
                            else:
                                new_coord[0] = (new_coord[0] - local_x[0])/(local_x[1] - local_x[0])
                            if local_y[0] == local_y[1]:
                                new_coord[1] = 1.0
                            else:
                                new_coord[1] = (new_coord[1] - local_y[0])/(local_y[1] - local_y[0])
                            if local_z[0] == local_z[1]:
                                new_coord[2] = 1.0
                            else:
                                new_coord[2] = (new_coord[2] - local_z[0])/(local_z[1] - local_z[0])
                            new_structure_level_labels[coord_idx] = new_coord
                        res["structure_level"].update({structure_level_key: new_structure_level_labels})
                    elif structure_level_key == "prot_secondary":
                        structure_level_labels = structure_level_item[1]
                        cur_label_size = label_size_dict["structure_level"][structure_level_key]
                        cur_output_mode = output_mode_dict["structure_level"][structure_level_key]
                        if cur_output_mode in ["multi_label", "multi-label"]:
                            new_structure_level_labels = []
                            if self.non_ignore and structure_level_key in self.non_ignore:
                                for _ in range(seq_length):
                                    tmp = []
                                    for _ in range(cur_label_size):
                                        tmp.append(0)
                                    new_structure_level_labels.append(tmp)
                                # new_structure_level_labels = [[0] * cur_label_size] * seq_length
                            else:
                                for _ in range(seq_length):
                                    tmp = []
                                    for _ in range(cur_label_size):
                                        tmp.append(self.ignore_index)
                                    new_structure_level_labels.append(tmp)
                                # new_structure_level_labels = [[self.ignore_index] * cur_label_size] * seq_length
                        else:
                            new_structure_level_labels = []
                            if self.non_ignore and structure_level_key in self.non_ignore:
                                for _ in range(seq_length):
                                    new_structure_level_labels.append(0)
                                # new_structure_level_labels = [0] * seq_length
                            else:
                                for _ in range(seq_length):
                                    new_structure_level_labels.append(self.ignore_index)
                                # new_structure_level_labels = [self.ignore_index] * seq_length
                        if structure_level_labels is not None and len(structure_level_labels) > 0:
                            begin_idx = 0
                            end_idx = seq_length
                            if self.add_special_tokens:
                                begin_idx = 1
                                end_idx = seq_length - 1
                            for idx, item in enumerate(structure_level_labels):
                                idx += begin_idx
                                if idx >= end_idx:
                                    break
                                if cur_output_mode in ["multi_label", "multi-label"]:
                                    new_structure_level_labels[idx][item] = 1
                                else:
                                    new_structure_level_labels[idx] = item
                        res["structure_level"] .update({structure_level_key: new_structure_level_labels})
                    elif structure_level_key == "prot_contact":
                        structure_level_labels = structure_level_item[1]
                        cur_output_mode = output_mode_dict["structure_level"][structure_level_key]
                        new_structure_level_labels = []
                        for _ in range(seq_length):
                            new_structure_level_labels.append(self.ignore_index)
                        # new_structure_level_labels = [self.ignore_index] * seq_length
                        begin_idx = 0
                        end_idx = seq_length
                        if self.add_special_tokens:
                            begin_idx = 1
                            end_idx = seq_length - 1
                        for idx, item in enumerate(structure_level_labels):
                            idx += begin_idx
                            if idx >= end_idx:
                                break
                            if cur_output_mode in ["multi_label", "multi-label"]:
                                new_structure_level_labels[idx][item] = 1
                            else:
                                new_structure_level_labels[idx] = item
                        res["structure_level"].update({structure_level_key: new_structure_level_labels})

        if "all" in pretrain_task_level_type or "seq_level" in pretrain_task_level_type:
            if "seq_level" in label and label["seq_level"]:
                res["seq_level"] = {}
                for seq_level_item in label["seq_level"].items():
                    seq_level_key = seq_level_item[0]
                    if seq_level_key in ["prot_taxid", "gene_taxid"]:
                        continue
                    seq_level_labels = seq_level_item[1]
                    cur_label_size = label_size_dict["seq_level"][seq_level_key]
                    cur_output_mode = output_mode_dict["seq_level"][seq_level_key]
                    if cur_output_mode in ["multi_label", "multi-label"]:
                        new_seq_level_labels = []
                        if self.non_ignore and seq_level_key in self.non_ignore:
                            for _ in range(cur_label_size):
                                new_seq_level_labels.append(0)
                            # new_seq_level_labels = [0] * cur_label_size
                        else:
                            for _ in range(cur_label_size):
                                new_seq_level_labels.append(self.ignore_index)
                            # new_seq_level_labels = [self.ignore_index] * cur_label_size
                        if seq_level_labels is not None and len(seq_level_labels) > 0:
                            for v in seq_level_labels:
                                new_seq_level_labels[v] = 1
                    else:
                        if seq_level_labels is not None and len(str(seq_level_labels)) > 0:
                            new_seq_level_labels = [seq_level_labels]
                        else:
                            new_seq_level_labels = [self.ignore_index]
                    res["seq_level"].update({seq_level_key: new_seq_level_labels})
        if "all" in pretrain_task_level_type or "whole_level" in pretrain_task_level_type:
            if "refs" in label or "ref" in label:
                res["whole_level"] = {}
                refs = label["refs"] if "refs" in label else label["ref"]
                new_refs = None
                if refs and len(refs) > 0:
                    new_refs = []
                    begin_idx = 0
                    end_idx = seq_length
                    if self.add_special_tokens:
                        end_idx = seq_length - 1
                        begin_idx = 0
                    for item in refs:
                        span_idx = item + begin_idx
                        if span_idx >= end_idx:
                            continue
                        new_refs.append(span_idx)
                res["whole_level"] .update({"refs": new_refs})

        return res

    def encode_pair(self,
                    pretrain_task_level_type,
                    gene_id,
                    gene_seq, gene_label, gene_label_size_dict, gene_output_mode_dict,
                    prot_id,
                    prot_seq, prot_label, prot_label_size_dict, prot_output_mode_dict,
                    pair_label, pair_label_size_dict, pair_output_mode_dict):
        res = {}
        sample_id = ""
        if gene_seq:
            gene_seq = gene_seq_replace(gene_seq)
            gene_encode, gene_length = self.__encode__(pretrain_task_level_type, gene_seq, gene_label, gene_label_size_dict, gene_output_mode_dict)
            if gene_encode:
                res.update({
                    "gene_id": gene_id,
                    "gene": gene_encode
                })
                sample_id = gene_id
        if prot_seq:
            prot_encode, prot_length = self.__encode__(pretrain_task_level_type, prot_seq, prot_label, prot_label_size_dict, prot_output_mode_dict)
            if not self.no_token_type_embeddings:
                prot_encode["token_type_ids"] = [1] * len(prot_encode["attention_mask"])
            if prot_encode:
                res.update({
                    "prot_id": prot_id,
                    "prot": prot_encode
                })
                if len(sample_id) == 0:
                    sample_id = prot_id
                else:
                    sample_id = sample_id + "#" + prot_id
        res["sample_id"] = sample_id
        pair = None

        if gene_seq and prot_seq and ("all" in pretrain_task_level_type or "pair_level" in pretrain_task_level_type):
            pair = {"pair_label": {"pair_level": {}}}
            for pair_level_item in pair_label["pair_level"].items():
                pair_level_key = pair_level_item[0]
                pair_level_labels = pair_level_item[1]
                cur_label_size = pair_label_size_dict["pair_level"][pair_level_key]
                cur_output_mode = pair_output_mode_dict["pair_level"][pair_level_key]
                if cur_output_mode in ["multi_label", "multi-label"]:
                    if self.non_ignore and pair_level_key in self.non_ignore:
                        new_pair_level_labels = [0] * cur_label_size
                    else:
                        new_pair_level_labels = [self.ignore_index] * cur_label_size
                    if pair_level_labels:
                        for v in pair_level_labels:
                            new_pair_level_labels[v] = 1
                else:
                    if pair_level_labels is not None:
                        new_pair_level_labels = [pair_level_labels]
                    else:
                        new_pair_level_labels = [self.ignore_index]
                pair["pair_label"]["pair_level"] .update({pair_level_key: new_pair_level_labels})

        if pair and len(pair) > 0:
            res.update({
                "pair": pair
            })
        return res

    def encode_single(self,
                      pretrain_task_level_type,
                      obj_id,
                      obj_type,
                      obj_seq,
                      obj_label,
                      label_size_dict,
                      output_mode_dict
                      ):
        res = {}
        if obj_type == "gene":
            obj_seq = gene_seq_replace(obj_seq)
            gene_encode, gene_length = self.__encode__(pretrain_task_level_type, obj_seq, obj_label, label_size_dict, output_mode_dict)
            if gene_encode:
                res.update({
                    "gene_id": obj_id,
                    "gene": gene_encode
                })
        else:
            prot_encode, prot_length = self.__encode__(pretrain_task_level_type, obj_seq, obj_label, label_size_dict, output_mode_dict)
            if not self.no_token_type_embeddings:
                prot_encode["token_type_ids"] = [1] * len(prot_encode["attention_mask"])
            if prot_encode:
                res.update({
                    "prot_id": obj_id,
                    "prot": prot_encode
                })
        res["sample_id"] = obj_id
        return res

    def encode_char_single(self,
                           pretrain_task_level_type,
                           obj_id,
                           obj_type,
                           obj_seq,
                           obj_label,
                           label_size_dict,
                           output_mode_dict):
        res = {}
        if obj_type == "gene":
            obj_seq = gene_seq_replace(obj_seq)
            gene_encode, gene_length = self.__encode_char__(pretrain_task_level_type, obj_seq, obj_label, label_size_dict, output_mode_dict)
            if gene_encode:
                res.update({
                    "gene_id": obj_id,
                    "gene_ori_seq": gene_encode["ori_seq"],
                    "gene_seq": gene_encode["seq"],
                    "gene_labels": gene_encode["labels"],
                })
        else:
            prot_encode, prot_length = self.__encode_char__(pretrain_task_level_type, obj_seq, obj_label, label_size_dict, output_mode_dict)
            if prot_encode:
                res.update({
                    "prot_id": obj_id,
                    "prot_ori_seq": prot_encode["ori_seq"],
                    "prot_seq": prot_encode["seq"],
                    "prot_labels": prot_encode["labels"],
                })
        res["sample_id"] = obj_id
        return res

    def encode_char_pair(self,
                         pretrain_task_level_type,
                         gene_id,
                         gene_seq, gene_label, gene_label_size_dict, gene_output_mode_dict,
                         prot_id,
                         prot_seq, prot_label, prot_label_size_dict, prot_output_mode_dict,
                         pair_label, pair_label_size_dict, pair_output_mode_dict):
        res = {}
        sample_id = ""
        if gene_seq:
            gene_seq = gene_seq_replace(gene_seq)
            gene_encode, gene_length = self.__encode_char__(pretrain_task_level_type, gene_seq, gene_label, gene_label_size_dict, gene_output_mode_dict)
            if gene_encode:
                res.update({
                    "gene_id": gene_id,
                    "gene_ori_seq": gene_encode["ori_seq"],
                    "gene_seq": gene_encode["seq"],
                    "gene_labels": gene_encode["labels"],
                })
                sample_id = gene_id
        if prot_seq:
            prot_encode, prot_length = self.__encode_char__(pretrain_task_level_type, prot_seq, prot_label, prot_label_size_dict, prot_output_mode_dict)
            if prot_encode:
                res.update({
                    "prot_id": prot_id,
                    "prot_ori_seq": prot_encode["ori_seq"],
                    "prot_seq": prot_encode["seq"],
                    "prot_labels": prot_encode["labels"],
                })
                if len(sample_id) == 0:
                    sample_id = prot_id
                else:
                    sample_id = sample_id + "#" + prot_id
        res["sample_id"] = sample_id
        pair = None
        if gene_seq and prot_seq and ("all" in pretrain_task_level_type or "pair_level" in pretrain_task_level_type):
            pair = {"pair_label": {"pair_level": {}}}
            for pair_level_item in pair_label["pair_level"].items():
                pair_level_key = pair_level_item[0]
                pair_level_labels = pair_level_item[1]
                cur_label_size = pair_label_size_dict["pair_level"][pair_level_key]
                cur_output_mode = pair_output_mode_dict["pair_level"][pair_level_key]
                if cur_output_mode in ["multi_label", "multi-label"]:
                    if self.non_ignore and pair_level_key in self.non_ignore:
                        new_pair_level_labels = [0] * cur_label_size
                    else:
                        new_pair_level_labels = [self.ignore_index] * cur_label_size
                    if pair_level_labels:
                        for v in pair_level_labels:
                            new_pair_level_labels[v] = 1
                else:
                    if pair_level_labels is not None:
                        new_pair_level_labels = [pair_level_labels]
                    else:
                        new_pair_level_labels = [self.ignore_index]
                pair["pair_label"]["pair_level"] .update({pair_level_key: new_pair_level_labels})

        if pair and len(pair) > 0:
            res.update({
                "pair": pair
            })
        return res

