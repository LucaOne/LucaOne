#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/18 15:32
@project: LucaOne
@file: get_embedding.py
@desc: get embedding from pretrained LucaOne
'''
import os
import sys
import json
import torch
import argparse
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from .args import Args
    from .utils import set_seed, to_device, get_labels, get_parameter_number, gene_seq_replace
    from .models.lucaone_gplm import LucaGPLM
    from .models.lucaone_gplm_config import LucaGPLMConfig
    from .models.alphabet import Alphabet
    from .batch_converter import BatchConverter
except ImportError as e:
    from src.args import Args
    from src.utils import set_seed, to_device, get_labels, get_parameter_number, gene_seq_replace
    from src.models.lucaone_gplm import LucaGPLM
    from src.models.lucaone_gplm_config import LucaGPLMConfig
    from src.models.alphabet import Alphabet
    from src.batch_converter import BatchConverter
from transformers import AutoTokenizer, PretrainedConfig, BertTokenizer
from collections import OrderedDict


def load_model(log_filepath, model_dirpath):
    '''
    create tokenizer, model config, model
    :param args:
    :return:
    '''
    with open(log_filepath, "r") as rfp:
        for line_idx, line in enumerate(rfp):
            if line_idx == 0:
                try:
                    args_info = json.loads(line.strip(), encoding="UTF-8")
                except Exception as e:
                    args_info = json.loads(line.strip())
                break
    print("Model dirpath: %s" % model_dirpath)
    assert model_dirpath is not None and os.path.exists(model_dirpath)
    # create tokenizer
    tokenizer_dir = os.path.join(model_dirpath, "tokenizer")
    assert os.path.exists(tokenizer_dir)
    if args_info["tokenization"]:
        print("AutoTokenizer, tokenizer dir: %s" % tokenizer_dir)
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            do_lower_case=args_info["do_lower_case"],
            truncation_side=args_info["truncation"]
        )
    elif args_info["model_type"] in ["lucaone_gplm"]:
        print("Alphabet, vocab path: %s" % tokenizer_dir)
        tokenizer = Alphabet.from_pretrained(tokenizer_dir)
    else:
        print("BertTokenizer, vocab path: %s" % tokenizer_dir)
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_dir,
            do_lower_case=args_info["do_lower_case"],
            truncation_side=args_info["truncation"])
    # four type of models
    if args_info["model_type"] in ["lucaone_gplm"]:
        config_class, model_class = LucaGPLMConfig, LucaGPLM
    else:
        raise Exception("Not support model_type=%s" % args_info["model_type"])

    # model config
    model_config: PretrainedConfig = config_class.from_json_file(os.path.join(model_dirpath, "config.json"))

    # load the pretrained model or create the model
    print("Load pretrained model: %s" % model_dirpath)
    args = Args()
    args.pretrain_tasks = args_info["pretrain_tasks"]
    args.ignore_index = args_info["ignore_index"]
    args.label_size = args_info["label_size"]
    args.loss_type = args_info["loss_type"]
    args.output_mode = args_info["output_mode"]
    args.max_length = args_info["max_length"]
    args.classifier_size = args_info["classifier_size"]
    try:
        model = model_class.from_pretrained(model_dirpath, args=args)
    except Exception as e:
        model = None
    if model is None:
        try:
            model = torch.load(os.path.join(model_dirpath, "pytorch.pt"), map_location=torch.device("cpu"))
        except Exception as e:
            model = model_class(model_config, args=args)
            pretrained_net_dict = torch.load(os.path.join(model_dirpath, "pytorch.pth"),
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
            model.load_state_dict(new_state_dict)
    # print(model)
    return args_info, model_config, model, tokenizer


def encoder(args_info, model_config, seq, seq_type, tokenizer):
    seqs = [seq]
    seq_types = [seq_type]
    seq_encoded_list = [tokenizer.encode(seq)]
    if args_info["max_length"]:
        seq_encoded_list = [encoded[:args_info["max_length"]] for encoded in seq_encoded_list]
    max_len = max(len(seq_encoded) for seq_encoded in seq_encoded_list)
    processed_seq_len = max_len + int(tokenizer.prepend_bos) + int(tokenizer.append_eos)
    # for input
    input_ids = torch.empty(
        (
            1,
            processed_seq_len,
        ),
        dtype=torch.int64,
    )
    input_ids.fill_(tokenizer.padding_idx)

    position_ids = None
    if not model_config.no_position_embeddings:
        position_ids = torch.empty(
            (
                1,
                processed_seq_len,
            ),
            dtype=torch.int64,
        )
        position_ids.fill_(tokenizer.padding_idx)

    token_type_ids = None
    if not model_config.no_token_type_embeddings:
        token_type_ids = torch.empty(
            (
                1,
                processed_seq_len,
            ),
            dtype=torch.int64,
        )
        token_type_ids.fill_(tokenizer.padding_idx)

    for i, (seq_type, seq_str, seq_encoded) in enumerate(
            zip(seq_types, seqs, seq_encoded_list)
    ):
        if tokenizer.prepend_bos:
            input_ids[i, 0] = tokenizer.cls_idx
        seq = torch.tensor(seq_encoded, dtype=torch.int64)
        input_ids[i, int(tokenizer.prepend_bos): len(seq_encoded) + int(tokenizer.prepend_bos)] = seq
        if tokenizer.append_eos:
            input_ids[i, len(seq_encoded) + int(tokenizer.prepend_bos)] = tokenizer.eos_idx

        if not model_config.no_position_embeddings:
            cur_len = int(tokenizer.prepend_bos) + len(seq_encoded) + int(tokenizer.append_eos)
            for idx in range(0, cur_len):
                position_ids[i, idx] = idx
        if not model_config.no_token_type_embeddings:
            if seq_type == "gene":
                type_value = 0
            else:
                type_value = 1
            cur_len = int(tokenizer.prepend_bos) + len(seq_encoded) + int(tokenizer.append_eos)
            for idx in range(0, cur_len):
                token_type_ids[i, idx] = type_value

    encoding = {
        "input_ids": input_ids, 
        "token_type_ids": token_type_ids, 
        "position_ids": position_ids
    }

    if seq_type == "prot":
        new_encoding = {}
        for item in encoding.items():
            new_encoding[item[0] + "_b"] = item[1]
        encoding = new_encoding
    return encoding, processed_seq_len


def get_embedding(args_info, model_config, tokenizer, model, seq, seq_type, device):
    if args_info["model_type"] in ["lucaone_gplm"]:
        if seq_type == "gene":
            seq = gene_seq_replace(seq)
            batch, processed_seq_len = encoder(args_info, model_config, seq, seq_type, tokenizer)
        else:
            batch, processed_seq_len = encoder(args_info, model_config, seq, seq_type, tokenizer)
        new_batch = {}
        for item in batch.items():
            if torch.is_tensor(item[1]):
                new_batch[item[0]] = item[1].to(device)
        new_batch["return_contacts"] = True
        new_batch["return_dict"] = True
        new_batch["repr_layers"] = list(range(args_info["num_hidden_layers"] + 1))
        batch = new_batch
        print("batch:")
        print(batch)
    else:
        raise Exception("Not support model_type=%s" % args_info["model_type"])
    print("llm embedding device: ", device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model(**batch)
        return output, processed_seq_len


def get_args():
    parser = argparse.ArgumentParser(description='LucaOne/LucaGPLM')
    # for logging
    parser.add_argument("--dataset_type", type=str, default="v2.0", help="dataset type")
    parser.add_argument("--model_type", type=str, default="lucaone_gplm", help="model type")
    parser.add_argument("--task_level", type=str, default="token_level,span_level,seq_level,structure_level", help="task type")
    parser.add_argument("--time_str", type=str, default="20231125113045", help="time str")
    parser.add_argument("--step", type=int, default=5600000, help="step.")
    parser.add_argument('--no_cuda', action='store_true', help='whether not to use GPU')
    parser.add_argument("--seq", type=str, default=None, required=True, help="seq")
    parser.add_argument("--seq_type", type=str, default=None, required=True, choices=["gene", "prot"], help="seq_type")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    log_filepath = "../logs/lucagplm/%s/%s/%s/%s/logs.txt" % (
        args.dataset_type, args.task_level, args.model_type, args.time_str
    )
    model_dirpath = "../models/lucagplm/%s/%s/%s/%s/checkpoint-step%d" % (
        args.dataset_type, args.task_level, args.model_type,
        args.time_str, args.step
    )
    if not os.path.exists(model_dirpath):
        model_dirpath = "../models/lucagplm/%s/%s/%s/%s/checkpoint-step%d" % (
            args.dataset_type, args.task_level, args.model_type, args.time_str, args.step
    )
    args_info, model_config, model, tokenizer = load_model(
        log_filepath, model_dirpath
    )

    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")
    print("input seq type: %s" % args.seq_type)
    print("input seq length: %d" % len(args.seq))
    print("device: %s" % device)

    emb, processed_seq_len = get_embedding(args_info, model_config, tokenizer, model, args.seq, args.seq_type, device)
    print("done seq length: %d" % processed_seq_len)

    # losses, outputs, hidden_states, attentions, cross_attentions, global_attentions,
    if isinstance(emb, list):
        pass
    else:
        info_type = input("type(l=loss, o=output, h=hidden_states, a=attentions, c=contacts)")
        if info_type == "l":
            if emb.losses is not None:
                print("losses:")
                print(emb.losses)
            if emb.losses_b is not None:
                print("losses_b:")
                print(emb.losses_b)
        elif info_type == "o":
            if emb.outputs is not None:
                print("outputs:")
                print(emb.outputs)
            if emb.outputs_b is not None:
                print("outputs_b:")
                print(emb.outputs_b)
        elif info_type == "h":
            if emb.hidden_states is not None:
                print("hidden_states:")
                print(emb.hidden_states.shape)
                print(emb.hidden_states)
                print(torch.sum(emb.hidden_states, dim=-1))
            if emb.hidden_states_b is not None:
                print("hidden_states_b:")
                print(emb.hidden_states_b.shape)
                print(emb.hidden_states_b)
                print(torch.sum(emb.hidden_states_b, dim=-1))
        elif info_type == "a":
            if emb.attentions is not None:
                print("attentions:")
                print(emb.attentions.shape)
            if emb.attentions_b is not None:
                print("attentions_b:")
                print(emb.attentions_b.shape)
        elif info_type == "c":
            if emb.contacts is not None:
                print("contacts:")
                print(emb.contacts)
                print(emb.contacts.shape)
            if emb.contacts_b is not None:
                print("contacts_b:")
                print(emb.contacts_b)
                print(emb.contacts_b.shape)
        if emb.attentions is not None:
            attention = emb.attentions
        else:
            attention = emb.attentions_b
        while True:
            layer_idx = input("layer idx(1~%d):" % args_info["num_hidden_layers"])

            layer_idx = int(layer_idx)
            if layer_idx < 1 or layer_idx > args_info["num_hidden_layers"]:
                break
            head_idx = input("head idx(1~%d):" % args_info["num_attention_heads"])
            head_idx = int(head_idx)
            if head_idx < 1 or head_idx > args_info["num_attention_heads"]:
                break
            print("the attention matrix(layer=%d, head=%d):" % (layer_idx, head_idx))
            cur_attention = attention[0, layer_idx - 1, head_idx - 1, :, :]
            print(cur_attention)
            print(torch.nonzero(cur_attention))


if __name__ == "__main__":
    main()

