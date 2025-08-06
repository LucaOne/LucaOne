#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/7/18 15:32
@project: LucaOne
@file: get_embedding
@desc: get embedding from pretrained llm
'''
import os
import sys
import json
import torch
import argparse
import numpy as np
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from args import Args
    from file_operator import fasta_reader, csv_reader, tsv_reader
    from utils import set_seed, to_device, get_labels, get_parameter_number, gene_seq_replace, \
        download_trained_checkpoint_lucaone_v1, clean_seq_luca, available_gpu_id, matrix_2_vector,\
        calc_emb_filename_by_seq_id, seq_type_is_match_seq
    from models.lucaone_gplm import LucaGPLM
    from models.lucaone_gplm_config import LucaGPLMConfig
    from models.alphabet import Alphabet
    from batch_converter import BatchConverter
except ImportError as e:
    from src.args import Args
    from src.file_operator import fasta_reader, csv_reader, tsv_reader
    from src.utils import set_seed, to_device, get_labels, get_parameter_number, gene_seq_replace, \
        download_trained_checkpoint_lucaone_v1, clean_seq_luca, available_gpu_id, matrix_2_vector, \
        calc_emb_filename_by_seq_id, seq_type_is_match_seq
    from src.models.lucaone_gplm import LucaGPLM
    from src.models.lucaone_gplm_config import LucaGPLMConfig
    from src.models.alphabet import Alphabet
    from src.batch_converter import BatchConverter
from transformers import AutoTokenizer, PretrainedConfig, BertTokenizer
from collections import OrderedDict


global_log_filepath, global_model_dirpath, global_args_info, \
global_model_config, global_model_version, global_model, global_tokenizer = None, None, None, None, None, None, None


def load_model(
        log_filepath,
        model_dirpath,
        embedding_inference=True
):
    """
    create tokenizer, model config, model
    :param log_filepath:
    :param model_dirpath:
    :param embedding_inference:
    :return:
    """
    with open(log_filepath, "r") as rfp:
        for line_idx, line in enumerate(rfp):
            if line_idx == 0:
                try:
                    args_info = json.loads(line.strip(), encoding="UTF-8")
                except Exception as e:
                    args_info = json.loads(line.strip())
                break
    model_dirpath = os.path.abspath(model_dirpath)
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
    elif args_info["model_type"] in ["lucaone_gplm", "lucaone", "lucagplm"]:
        print("Alphabet, vocab path: %s" % tokenizer_dir)
        tokenizer = Alphabet.from_pretrained(tokenizer_dir)
    else:
        print("BertTokenizer, vocab path: %s" % tokenizer_dir)
        tokenizer = BertTokenizer.from_pretrained(
            tokenizer_dir,
            do_lower_case=args_info["do_lower_case"],
            truncation_side=args_info["truncation"])
    # four type of models
    if args_info["model_type"] in ["lucaone_gplm", "lucaone", "lucagplm"]:
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
    args.pretrained_model_name = None
    args.embedding_inference = embedding_inference
    try:
        model = model_class.from_pretrained(model_dirpath, args=args)
    except Exception as e:
        model = None
    if model is None:
        model = model_class(model_config, args=args)
        pretrained_net_dict = torch.load(
            os.path.join(model_dirpath, "pytorch.pth"),
            map_location=torch.device("cpu")
        )
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


def encoder(
        args_info,
        model_config,
        seq,
        seq_type,
        tokenizer
):
    if args_info["tokenization"]:
        # seq to seq ids
        encoding = tokenizer.encode_plus(
            text=seq,
            text_pair=None,
            add_special_tokens=args_info["add_special_tokens"],
            padding="max_length",
            max_length=model_config.max_position_embeddings,
            return_attention_mask=True,
            return_token_type_ids=not model_config.no_token_type_embeddings,
            return_length=False,
            truncation=True
        )
        processed_seq_len = sum(encoding["attention_mask"])
    elif args_info["model_type"] in ["lucaone_gplm", "lucaone", "lucagplm"]:
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

        encoding = {"input_ids": input_ids, "token_type_ids": token_type_ids, "position_ids": position_ids}
    else:
        max_length = model_config.max_position_embeddings
        if args_info["add_special_tokens"]:
            max_length = max_length - 2
        if len(seq) > max_length:
            if args_info["truncation"] == "right":
                seq = seq[:max_length]
            elif args_info["truncation"] == "left":
                seq = seq[-max_length:]
            processed_seq_len = max_length + 2 if args_info["add_special_tokens"] else max_length
        else:
            processed_seq_len = len(seq) + 2 if args_info["add_special_tokens"] else len(seq)
        seq = " ".join(list(seq))
        encoding = tokenizer.encode_plus(
            text=seq,
            text_pair=None,
            add_special_tokens=args_info["add_special_tokens"],
            padding="max_length",
            max_length=model_config.max_position_embeddings,
            return_attention_mask=True,
            return_token_type_ids=not model_config.no_token_type_embeddings,
            return_length=False,
            truncation=True
        )
    if args_info["model_type"] in ["lucaone_longformer"]:
        # [CLS] and [SEP] for global attention
        # start:[CLS]
        encoding["attention_mask"][0] = 2
        # end:[SEP]
        encoding["attention_mask"][processed_seq_len - 1] = 2
    if seq_type == "prot":
        new_encoding = {}
        for item in encoding.items():
            new_encoding[item[0] + "_b"] = item[1]
        encoding = new_encoding
    return encoding, processed_seq_len


def get_embedding(
        args_info,
        model_config,
        tokenizer,
        model,
        seq,
        seq_type,
        device
):
    if args_info["model_type"] in ["lucaone_gplm", "lucaone", "lucagplm"]:
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
        # print("batch:")
        # print(batch)
    else:
        if seq_type == "gene":
            seq = gene_seq_replace(seq)
            encoding, processed_seq_len = encoder(args_info, model_config, seq, seq_type, tokenizer)
        else:
            encoding, processed_seq_len = encoder(args_info, model_config, seq, seq_type, tokenizer)
            if not model_config.no_token_type_embeddings:
                encoding["token_type_ids"] = [1] * len(encoding["attention_mask"])

        batch = {
            "input_ids": torch.tensor([encoding["input_ids"]], dtype=torch.long).to(device),
            "attention_mask": torch.tensor([encoding["attention_mask"]], dtype=torch.long).to(device),
            "token_type_ids": torch.tensor([encoding["token_type_ids"]], dtype=torch.long).to(device),
            "output_attentions": True,
            "output_hidden_states": True,
            "return_dict": True
        }
    print("llm embedding device: ", device)
    model.to(device)
    model.eval()
    try:
        with torch.no_grad():
            output = model(**batch)
            return output, processed_seq_len
    except Exception as e:
        print("get embedding:", e)
        return None, None


def predict_embedding(
        llm_dirpath,
        sample,
        trunc_type,
        embedding_type,
        repr_layers=[-1],
        truncation_seq_length=4094,
        device=None,
        matrix_add_special_token=False,
        save_type="numpy"
):
    """
    use sequence to predict bio seq embedding matrix or vector(bos)
    :param llm_dirpath: llm dir path
    :param sample: [seq_id, seq_type, seq]
    :param trunc_type:
    :param embedding_type: bos or representations
    :param repr_layers: [-1]
    :param truncation_seq_length: [4094, 2046, 1982, 1790, 1534, 1278, 1150, 1022]
    :param device:
    :param matrix_add_special_token:
    :parma save_type:
    :return: embedding, processed_seq_len
    """
    global global_log_filepath, global_model_dirpath, global_args_info, \
        global_model_config, global_model_version, global_model, global_tokenizer

    assert "bos" in embedding_type \
           or "representations" in embedding_type \
           or "matrix" in embedding_type \
           or "vector" in embedding_type \
           or "contacts" in embedding_type

    seq_id, seq_type, seq = sample[0], sample[1], sample[2]

    if isinstance(llm_dirpath, str):
        cur_log_filepath = os.path.join(os.path.dirname(llm_dirpath).replace("models", "logs"), "logs.txt")
        cur_model_dirpath = llm_dirpath
        if global_log_filepath != cur_log_filepath or global_model_dirpath != cur_model_dirpath:
            global_log_filepath = cur_log_filepath
            global_model_dirpath = cur_model_dirpath
            global_args_info, global_model_config, global_model, global_tokenizer = \
                load_model(
                    log_filepath=global_log_filepath,
                    model_dirpath=global_model_dirpath,
                    embedding_inference=True
                )
        global_args_info["max_length"] = truncation_seq_length
    else:
        for item in llm_dirpath.items():
            key = item[0]
            tmp_log_filepath = os.path.join(os.path.dirname(item[1]).replace("models", "logs"), "logs.txt")
            tmp_model_dirpath = item[1]
            if global_log_filepath is None or key not in global_log_filepath or global_log_filepath[key] != tmp_log_filepath:
                print("item:", item)
                if global_log_filepath is None:
                    global_log_filepath = {}
                    global_model_dirpath = {}
                    global_args_info = {}
                    global_model_config = {}
                    global_model = {}
                    global_tokenizer = {}
                global_log_filepath[key] = tmp_log_filepath
                global_model_dirpath[key] = tmp_model_dirpath
                tmp_args_info, tmp_model_config, tmp_model, tmp_tokenizer = \
                    load_model(
                        log_filepath=tmp_log_filepath,
                        model_dirpath=tmp_model_dirpath,
                        embedding_inference=True
                    )
                global_args_info[key] = tmp_args_info
                global_model_config[key] = tmp_model_config
                global_model[key] = tmp_model
                global_tokenizer[key] = tmp_tokenizer
            global_args_info[key]["max_length"] = truncation_seq_length

    processed_seq = clean_seq_luca(seq_id, seq)
    if len(processed_seq) > truncation_seq_length:
        if trunc_type == "left":
            processed_seq = processed_seq[-truncation_seq_length:]
        else:
            processed_seq = processed_seq[:truncation_seq_length]

    if device is None and not torch.cuda.is_available():
        device = torch.device("cpu")
        # print("llm use cpu")
    elif device is None and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        pass
    # print("get embedding device:", device)
    if isinstance(llm_dirpath, str):
        emb, processed_seq_len = get_embedding(
            global_args_info,
            global_model_config,
            global_tokenizer,
            global_model,
            processed_seq,
            seq_type,
            device=device
        )
    else:
        if seq_type == "gene":
            emb, processed_seq_len = get_embedding(
                global_args_info["gene"],
                global_model_config["gene"],
                global_tokenizer["gene"],
                global_model["gene"],
                processed_seq,
                seq_type,
                device=device
            )
        elif seq_type == "prot":
            emb, processed_seq_len = get_embedding(
                global_args_info["protein"],
                global_model_config["protein"],
                global_tokenizer["protein"],
                global_model["protein"],
                processed_seq,
                seq_type,
                device=device
            )
        else:
            raise Exception("Not support seq_type=%s for LucaOne" % seq_type)

    if emb is None:
        return None, None

    embeddings = {}
    if isinstance(llm_dirpath, str):
        prepend_len = int(global_tokenizer.prepend_bos)
        append_len = int(global_tokenizer.append_eos)
    else:
        if seq_type == "gene":
            prepend_len = int(global_tokenizer["gene"].prepend_bos)
            append_len = int(global_tokenizer["gene"].append_eos)
        elif seq_type == "prot":
            prepend_len = int(global_tokenizer["protein"].prepend_bos)
            append_len = int(global_tokenizer["protein"].append_eos)
        else:
            raise Exception("Not support seq_type=%s for LucaOne" % seq_type)

    if "representations" in embedding_type or "matrix" in embedding_type:
        if seq_type == "prot":
            embedding = emb.hidden_states_b
        else:
            embedding = emb.hidden_states
        if matrix_add_special_token:
            embeddings["representations"] = embedding[0, 0: processed_seq_len, :].to(device="cpu").clone()
        else:
            embeddings["representations"] = embedding[0, prepend_len: processed_seq_len - append_len, :].to(device="cpu").clone()
        if save_type == "numpy":
            embeddings["representations"] = embeddings["representations"].numpy()
        # print(embeddings["representations"].shape)
    if "bos" in embedding_type or "vector" in embedding_type:
        if seq_type == "prot":
            embedding = emb.hidden_states_b
        else:
            embedding = emb.hidden_states
        embeddings["bos_representations"] = embedding[0, 0, :].to(device="cpu").clone()
        if save_type == "numpy":
            embeddings["bos_representations"] = embeddings["bos_representations"].numpy()
    if "contacts" in embedding_type:
        if seq_type == "prot":
            embedding = emb.contacts_b
        else:
            embedding = emb.contacts
        embeddings["contacts"] = embedding.to(device="cpu")[0, :, :].clone()
        if save_type == "numpy":
            embeddings["contacts"] = embeddings["contacts"].numpy()
    if len(embeddings) > 1:
        return embeddings, processed_seq_len
    elif len(embeddings) == 1:
        return list(embeddings.items())[0][1], processed_seq_len
    else:
        return None, None


def complete_embedding_matrix(
        seq_id,
        seq_type,
        seq,
        truncation_seq_length,
        init_emb,
        model_args,
        embedding_type,
        matrix_add_special_token,
        save_type="numpy",
        use_cpu=False
):
    if init_emb is not None and model_args.embedding_complete and ("representations" in embedding_type or "matrix" in embedding_type):
        ori_seq_len = len(seq)
        # 每次能处理这么长度
        cur_segment_len = init_emb.shape[0]
        if matrix_add_special_token:
            complete_emb = init_emb[1:cur_segment_len - 1]
        else:
            complete_emb = init_emb
        if matrix_add_special_token:
            cur_segment_len = cur_segment_len - 2
        segment_num = int((ori_seq_len + cur_segment_len - 1)/cur_segment_len)
        if segment_num <= 1:
            return init_emb
        if model_args.embedding_complete_seg_overlap:
            sliding_window = cur_segment_len // 2
            print("updated window: %d" % sliding_window)
            # 第一个已经处理，滑动窗口
            if model_args.trunc_type == "right":
                last_end = cur_segment_len
                seg_idx = 0
                for pos_idx in range(cur_segment_len, ori_seq_len - sliding_window, sliding_window):
                    seg_idx += 1
                    last_end = min(pos_idx + sliding_window, ori_seq_len)
                    seg_seq = seq[pos_idx - sliding_window:last_end]
                    print("segment idx: %d, seg seq len: %d" % (seg_idx, len(seg_seq)))
                    seg_emb, seg_processed_seq_len = predict_embedding(
                        global_model_dirpath,
                        [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                        model_args.trunc_type,
                        embedding_type,
                        repr_layers=[-1],
                        truncation_seq_length=truncation_seq_length,
                        device=model_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type
                    )

                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[sliding_window:]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((complete_emb, seg_emb[sliding_window:]), axis=0)
                        else:
                            complete_emb = torch.cat((complete_emb, seg_emb[sliding_window:]), dim=0)
                if last_end < ori_seq_len:
                    seg_idx += 1
                    remain = ori_seq_len - last_end
                    seg_seq = seq[ori_seq_len - 2 * sliding_window:ori_seq_len]
                    seg_emb, seg_processed_seq_len = predict_embedding(
                        global_model_dirpath,
                        [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                        model_args.trunc_type,
                        embedding_type,
                        repr_layers=[-1],
                        truncation_seq_length=truncation_seq_length,
                        device=model_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type
                    )
                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[-remain:]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((complete_emb, seg_emb[-remain:]), axis=0)
                        else:
                            complete_emb = torch.cat((complete_emb, seg_emb[-remain:]), dim=0)
            else:
                last_start = -cur_segment_len
                seg_idx = 0
                for pos_idx in range(-cur_segment_len, -ori_seq_len + sliding_window, -sliding_window):
                    seg_idx += 1
                    last_start = min(pos_idx - sliding_window, -ori_seq_len)
                    seg_seq = seq[last_start: pos_idx + sliding_window]
                    seg_emb, seg_processed_seq_len = predict_embedding(
                        global_model_dirpath,
                        [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                        model_args.trunc_type,
                        embedding_type,
                        repr_layers=[-1],
                        truncation_seq_length=truncation_seq_length,
                        device=model_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type
                    )
                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[:sliding_window]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((seg_emb[:sliding_window], complete_emb), axis=0)
                        else:
                            complete_emb = torch.cat((seg_emb[:sliding_window], complete_emb), dim=0)
                if last_start > -ori_seq_len:
                    seg_idx += 1
                    remain = last_start - ori_seq_len
                    seg_seq = seq[-ori_seq_len:-ori_seq_len + 2 * sliding_window]
                    seg_emb, seg_processed_seq_len = predict_embedding(
                        global_model_dirpath,
                        [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                        model_args.trunc_type,
                        embedding_type,
                        repr_layers=[-1],
                        truncation_seq_length=truncation_seq_length,
                        device=model_args.device if not use_cpu else torch.device("cpu"),
                        matrix_add_special_token=False,
                        save_type=save_type
                    )
                    # 有seq overlap 所以要截取
                    if complete_emb is None:
                        complete_emb = seg_emb[:remain]
                    else:
                        if save_type == "numpy":
                            complete_emb = np.concatenate((seg_emb[:remain], complete_emb), axis=0)
                        else:
                            complete_emb = torch.cat((seg_emb[:remain], complete_emb), dim=0)
        else:
            # 第一个已经处理，最后一个单独处理（需要向左/向右扩充至cur_segment_len长度）
            if model_args.trunc_type == "right":
                begin_seq_idx = 0
            else:
                begin_seq_idx = ori_seq_len - (segment_num - 1) * cur_segment_len
            for seg_idx in range(1, segment_num - 1):
                seg_seq = seq[begin_seq_idx + seg_idx * cur_segment_len: begin_seq_idx + (seg_idx + 1) * cur_segment_len]
                seg_emb, seg_processed_seq_len = predict_embedding(
                    global_model_dirpath,
                    [seq_id + "_seg_%d" % seg_idx, seq_type, seg_seq],
                    model_args.trunc_type,
                    embedding_type,
                    repr_layers=[-1],
                    truncation_seq_length=truncation_seq_length,
                    device=model_args.device if not use_cpu else torch.device("cpu"),
                    matrix_add_special_token=False,
                    save_type=save_type
                )
                if model_args.trunc_type == "right":
                    if save_type == "numpy":
                        complete_emb = np.concatenate((complete_emb, seg_emb), axis=0)
                    else:
                        complete_emb = torch.cat((complete_emb, seg_emb), dim=0)
                else:
                    if save_type == "numpy":
                        complete_emb = np.concatenate((seg_emb, complete_emb), axis=0)
                    else:
                        complete_emb = torch.cat((seg_emb, complete_emb), dim=0)
            if model_args.trunc_type == "right": # 处理最后一个
                last_seg_seq = seq[-cur_segment_len:]
                really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                last_seg_emb, last_seg_processed_seq_len = predict_embedding(
                    global_model_dirpath,
                    [seq_id + "_seg_%d" % (segment_num - 1), seq_type, last_seg_seq],
                    model_args.trunc_type,
                    embedding_type,
                    repr_layers=[-1],
                    truncation_seq_length=truncation_seq_length,
                    device=model_args.device if not use_cpu else torch.device("cpu"),
                    matrix_add_special_token=False,
                    save_type=save_type
                )
                last_seg_emb = last_seg_emb[-really_len:, :]
                if save_type == "numpy":
                    complete_emb = np.concatenate((complete_emb, last_seg_emb), axis=0)
                else:
                    complete_emb = torch.cat((complete_emb, last_seg_emb), dim=0)
            else:
                # 处理第一个
                first_seg_seq = seq[:cur_segment_len]
                really_len = (ori_seq_len - (segment_num - 1) * cur_segment_len)
                first_seg_emb, first_seg_processed_seq_len = predict_embedding(
                    global_model_dirpath,
                    [seq_id + "_seg_0", seq_type, first_seg_seq],
                    model_args.trunc_type,
                    embedding_type,
                    repr_layers=[-1],
                    truncation_seq_length=truncation_seq_length,
                    device=model_args.device if not use_cpu else torch.device("cpu"),
                    matrix_add_special_token=False,
                    save_type=save_type
                )
                first_seg_emb = first_seg_emb[:really_len, :]
                if save_type == "numpy":
                    complete_emb = np.concatenate((first_seg_emb, complete_emb), axis=0)
                else:
                    complete_emb = torch.cat((first_seg_emb, complete_emb), dim=0)
        print("seq_len: %d, seq_embedding matrix len: %d" % (ori_seq_len, complete_emb.shape[0] + (2 if matrix_add_special_token else 0)))
        assert complete_emb.shape[0] == ori_seq_len
        if matrix_add_special_token:
            if save_type == "numpy":
                complete_emb = np.concatenate((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), axis=0)
            else:
                complete_emb = torch.cat((init_emb[0:1, :], complete_emb, init_emb[-1:, :]), dim=0)
        init_emb = complete_emb
    return init_emb


def get_args():
    parser = argparse.ArgumentParser(description='LucaOne/LucaOne-Gene/LucaOne-Prot Embedding')
    # for one seq
    parser.add_argument("--seq_id", type=str, default=None,
                        help="the seq id")
    parser.add_argument("--seq", type=str, default=None,
                        help="when to input a seq")
    parser.add_argument("--seq_type", type=str, default=None, required=True, choices=["gene", "prot"],
                        help="the input seq type")

    # for many
    parser.add_argument("--input_file", type=str, default=None,
                        help="the input file（format: fasta or csv or tsv)")
    # for input csv
    parser.add_argument("--id_idx", type=int, default=None,
                        help="id col idx(0 start)")
    parser.add_argument("--seq_idx", type=int, default=None,
                        help="seq col idx(0 start)")

    # for saved path
    parser.add_argument("--save_path", type=str, default=None,
                        help="embedding file save dir path")

    # for trained llm checkpoint
    parser.add_argument("--llm_dir", type=str, default="../models/",
                        help="the llm model dir")
    parser.add_argument("--llm_type", type=str, default="lucaone_gplm", choices=["esm", "lucaone_gplm"],
                        help="the llm type")
    parser.add_argument("--llm_version", type=str, default="v2.0", choices=["v2.0"],
                        help="the llm version")
    parser.add_argument("--llm_task_level", type=str, default="token_level,span_level,seq_level,structure_level",
                        choices=["token_level", "token_level,span_level,seq_level,structure_level"],
                        help="the llm task level")
    parser.add_argument("--llm_time_str", type=str, default=None,
                        help="the llm running time str")
    parser.add_argument("--llm_step", type=int, default=None,
                        help="the llm checkpoint step.")

    # for embedding
    parser.add_argument(
        "--embedding_type",
        type=str,
        default="matrix",
        choices=["matrix", "vector"],
        help="the llm embedding type."
    )
    parser.add_argument(
        "--save_type",
        type=str,
        default="numpy",
        choices=["tensor", "numpy"],
        help="the embedding save type(tensor or numpy)."
    )
    parser.add_argument(
        "--vector_type",
        type=str,
        default="mean",
        choices=["mean", "max", "cls"],
        help="the llm vector embedding type."
    )
    parser.add_argument("--trunc_type", type=str, default="right", choices=["left", "right"],
                        help="llm trunc type.")
    parser.add_argument("--truncation_seq_length", type=int, default=4094,
                        help="the llm truncation seq length(not contain [CLS] and [SEP].")
    parser.add_argument("--matrix_add_special_token", action="store_true",
                        help="whether to add special token embedding vector in seq representation matrix")
    parser.add_argument("--embedding_complete",  action="store_true",
                        help="when the seq len > inference_max_len, then the embedding matrix is completed by segment")
    parser.add_argument("--embedding_complete_seg_overlap",  action="store_true",
                        help="segment overlap")
    parser.add_argument("--embedding_fixed_len_a_time", type=int, default=None,
                        help="the embedding fixed length of once inference for longer sequence")

    # for running
    parser.add_argument('--gpu_id', type=int, default=-1,
                        help="the gpu id to use.")
    input_args = parser.parse_args()
    return input_args


def main(model_args):
    global global_log_filepath, global_model_dirpath, global_args_info, \
        global_model_config, global_model, global_tokenizer
    # model_args = get_args(), python39
    print("*" * 20 + "Model Args:" + "*" * 20)
    print(model_args)
    print("*" * 50)

    if model_args.llm_dir is None:
        model_args.llm_dir = ".."
    download_trained_checkpoint_lucaone_v1(
        llm_dir=model_args.llm_dir,
        llm_time_str=model_args.llm_time_str,
        llm_type=model_args.llm_type,
        llm_task_level=model_args.llm_task_level,
        llm_version=model_args.llm_version,
        llm_step=model_args.llm_step
    )

    cur_log_filepath = "%s/logs/lucagplm/%s/%s/%s/%s/logs.txt" % (
        model_args.llm_dir if model_args.llm_dir else "..", model_args.llm_version, model_args.llm_task_level,
        model_args.llm_type, model_args.llm_time_str
    )
    print("log_filepath: %s" % os.path.abspath(cur_log_filepath))

    cur_model_dirpath = "%s/models/lucagplm/%s/%s/%s/%s/checkpoint-%d" % (
        model_args.llm_dir if model_args.llm_dir else "..", model_args.llm_version, model_args.llm_task_level,
        model_args.llm_type, model_args.llm_time_str, model_args.llm_step
    )
    if not os.path.exists(cur_model_dirpath):
        cur_model_dirpath = "%s/models/lucagplm/%s/%s/%s/%s/checkpoint-step%d" % (
            model_args.llm_dir if model_args.llm_dir else "..", model_args.llm_version, model_args.llm_task_level,
            model_args.llm_type, model_args.llm_time_str, model_args.llm_step
        )
    print("model_dirpath: %s" % os.path.abspath(cur_model_dirpath))

    if not os.path.exists(cur_model_dirpath):
        cur_model_dirpath = "%s/models/lucagplm/%s/%s/%s/%s/checkpoint-step%d" % (
            model_args.llm_dir if model_args.llm_dir else "..", model_args.llm_version, model_args.llm_task_level,
            model_args.llm_type, model_args.llm_time_str, model_args.llm_step
        )
        cur_log_filepath = "%s/logs/lucagplm/%s/%s/%s/%s/logs.txt" % (
            model_args.llm_dir if model_args.llm_dir else "..", model_args.llm_version, model_args.llm_task_level,
            model_args.llm_type, model_args.llm_time_str
        )
    if global_log_filepath != cur_log_filepath or global_model_dirpath != cur_model_dirpath:
        global_log_filepath = cur_log_filepath
        global_model_dirpath = cur_model_dirpath
        global_args_info, global_model_config, global_model, global_tokenizer = load_model(
            log_filepath=global_log_filepath,
            model_dirpath=global_model_dirpath,
            embedding_inference=True
        )
    if model_args.gpu_id >= 0:
        gpu_id = model_args.gpu_id
    else:
        # gpu_id = available_gpu_id()
        gpu_id = -1
        print("gpu_id: ", gpu_id)

    model_args.device = torch.device("cuda:%d" % gpu_id if gpu_id > -1 else "cpu")
    # model.to(model_args.device)

    assert (model_args.input_file is not None and os.path.exists(model_args.input_file)) or model_args.seq is not None
    print("input seq type: %s" % model_args.seq_type)
    print("model_args device: %s" % model_args.device)
    embedding_type = model_args.embedding_type
    vector_type = model_args.vector_type
    if embedding_type == "vector" and vector_type == "cls":
        matrix_add_special_token = True
    else:
        matrix_add_special_token = model_args.matrix_add_special_token
    seq_type = model_args.seq_type
    emb_save_path = model_args.save_path
    print("emb save dir: %s" % os.path.abspath(emb_save_path))
    if seq_type not in ["gene", "prot"]:
        print("Error! arg: --seq_type=%s is not gene or prot" % seq_type)
        sys.exit(-1)
    if not os.path.exists(emb_save_path):
        os.makedirs(emb_save_path)

    if model_args.input_file and os.path.exists(model_args.input_file):
        done = 0
        file_reader = fasta_reader
        if model_args.input_file.endswith(".csv"):
            file_reader = csv_reader
        elif model_args.input_file.endswith(".tsv"):
            file_reader = tsv_reader

        for row in file_reader(model_args.input_file):
            if model_args.id_idx is None or model_args.seq_idx is None:
                if len(row) > 2:
                    seq_id, seq = row[0].strip(), row[2].upper()
                else:
                    seq_id, seq = row[0].strip(), row[1].upper()
            else:
                seq_id, seq = row[model_args.id_idx].strip(), row[model_args.seq_idx].upper()
            if not seq_type_is_match_seq(seq_type, seq):
                print("Error! the input seq(seq_id=%s) not match the arg: --seq_type=%s: %s" % (seq_id, seq_type, seq))
                sys.exit(-1)
            emb_filename = calc_emb_filename_by_seq_id(seq_id=seq_id, embedding_type=embedding_type)
            embedding_filepath = os.path.join(emb_save_path, emb_filename)
            if not os.path.exists(embedding_filepath):
                input_seq_len = len(seq)
                if model_args.embedding_complete:
                    truncation_seq_length = input_seq_len
                else:
                    truncation_seq_length = min(input_seq_len, model_args.truncation_seq_length)
                while True:
                    # 设置了一次性推理长度
                    if model_args.embedding_fixed_len_a_time and model_args.embedding_fixed_len_a_time > 0:
                        emb, processed_seq_len = predict_embedding(
                            global_model_dirpath,
                            [seq_id, seq_type, seq],
                            model_args.trunc_type,
                            embedding_type="matrix",
                            repr_layers=[-1],
                            truncation_seq_length=model_args.embedding_fixed_len_a_time,
                            device=model_args.device,
                            matrix_add_special_token=matrix_add_special_token,
                            save_type=model_args.save_type
                        )
                        # 如果指定的设备运行失败，则使用CPU
                        use_cpu = False
                        if emb is None:
                            emb, processed_seq_len = predict_embedding(
                                global_model_dirpath,
                                [seq_id, seq_type, seq],
                                model_args.trunc_type,
                                embedding_type="matrix",
                                repr_layers=[-1],
                                truncation_seq_length=model_args.embedding_fixed_len_a_time,
                                device=torch.device("cpu"),
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=model_args.save_type
                            )
                            use_cpu = True
                        if emb is not None and input_seq_len > model_args.embedding_fixed_len_a_time:
                            emb = complete_embedding_matrix(
                                seq_id,
                                seq_type,
                                seq,
                                model_args.embedding_fixed_len_a_time,
                                emb,
                                model_args,
                                embedding_type="matrix",
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=model_args.save_type,
                                use_cpu=use_cpu
                            )
                        if use_cpu:
                            print("use_cpu: %r" % use_cpu)
                    else:
                        emb, processed_seq_len = predict_embedding(
                            global_model_dirpath,
                            [seq_id, seq_type, seq],
                            model_args.trunc_type,
                            embedding_type="matrix",
                            repr_layers=[-1],
                            truncation_seq_length=truncation_seq_length,
                            device=model_args.device,
                            matrix_add_special_token=matrix_add_special_token,
                            save_type=model_args.save_type
                        )
                        use_cpu = False
                        if emb is None:
                            emb, processed_seq_len = predict_embedding(
                                global_model_dirpath,
                                [seq_id, seq_type, seq],
                                model_args.trunc_type,
                                embedding_type="matrix",
                                repr_layers=[-1],
                                truncation_seq_length=truncation_seq_length,
                                device=torch.device("cpu"),
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=model_args.save_type
                            )
                            use_cpu = True
                        # embedding全
                        if emb is not None and input_seq_len > truncation_seq_length:
                            emb = complete_embedding_matrix(
                                seq_id,
                                seq_type,
                                seq,
                                truncation_seq_length,
                                emb,
                                model_args,
                                embedding_type="matrix",
                                matrix_add_special_token=matrix_add_special_token,
                                save_type=model_args.save_type,
                                use_cpu=use_cpu
                            )
                        if use_cpu:
                            print("use_cpu: %r" % use_cpu)
                    if emb is not None:
                        # print("seq_len: %d" % len(seq))
                        # print("emb shape:", embedding_info.shape)
                        if embedding_type == "vector":
                            emb = matrix_2_vector(emb, matrix_add_special_token, vector_type, model_args.save_type)
                        torch.save(emb, embedding_filepath)
                        break
                    print("%s embedding error, max_len from %d truncate to %d" % (
                        seq_id,
                        truncation_seq_length,
                        int(truncation_seq_length * 0.95)
                    ))
                    truncation_seq_length = int(truncation_seq_length * 0.95)
                torch.cuda.empty_cache()
            else:
                print("%s exists." % embedding_filepath)
            done += 1
            if done % 1000 == 0:
                print("embedding done: %d" % done)
        print("embedding over, done: %d" % done)
    elif model_args.seq and model_args.seq_type:
        print("input seq length: %d" % len(model_args.seq))
        if model_args.seq_id is None:
            model_args.seq_id = "Unknown"
        if not seq_type_is_match_seq(model_args.seq_type, model_args.seq):
            print("Error! the input seq(seq_id=%s) not match the arg: --seq_type=%s: %s" % (model_args.seq_id, seq_type, model_args.seq))
            sys.exit(-1)
        input_seq_len = len(model_args.seq)
        if model_args.embedding_complete:
            truncation_seq_length = input_seq_len
        else:
            truncation_seq_length = min(input_seq_len, model_args.truncation_seq_length)
        while True:
            emb, processed_seq_len = predict_embedding(
                global_model_dirpath,
                [model_args.seq_id, model_args.seq_type, model_args.seq],
                model_args.trunc_type,
                embedding_type="matrix",
                repr_layers=[-1],
                truncation_seq_length=truncation_seq_length,
                device=model_args.device,
                matrix_add_special_token=matrix_add_special_token,
                save_type=model_args.save_type
            )
            use_cpu = False
            if emb is None:
                emb, processed_seq_len = predict_embedding(
                    global_model_dirpath,
                    [model_args.seq_id, model_args.seq_type, model_args.seq],
                    model_args.trunc_type,
                    embedding_type="matrix",
                    repr_layers=[-1],
                    truncation_seq_length=truncation_seq_length,
                    device=torch.device("cpu"),
                    matrix_add_special_token=matrix_add_special_token,
                    save_type=model_args.save_type
                )
                use_cpu = True
            # embedding全
            if emb is not None and input_seq_len > truncation_seq_length:
                emb = complete_embedding_matrix(
                    seq_id=model_args.seq_id,
                    seq_type=model_args.seq_type,
                    seq=model_args.seq,
                    truncation_seq_length=truncation_seq_length,
                    init_emb=emb,
                    model_args=model_args,
                    embedding_type="matrix",
                    matrix_add_special_token=matrix_add_special_token,
                    save_type=model_args.save_type,
                    use_cpu=use_cpu
                )
            if use_cpu:
                print("use_cpu: %r" % use_cpu)
            if emb is not None:
                print("processed seq length: %d, truncation seq length: %d" % (emb.shape[0], truncation_seq_length))
                break
            print("%s embedding error, max_len from %d truncate to %d" % (
                model_args.seq_id,
                truncation_seq_length,
                int(truncation_seq_length * 0.95)
            ))
            truncation_seq_length = int(truncation_seq_length * 0.95)
        print("seq: %s" % model_args.seq_id)
        print("seq: %s" % model_args.seq)
        print("embedding: ")
        if embedding_type == "vector":
            emb = matrix_2_vector(emb, matrix_add_special_token, vector_type, model_args.save_type)
        print(emb)
    else:
        raise Exception("input error, please --help.")


if __name__ == "__main__":
    run_args = get_args()
    main(run_args)