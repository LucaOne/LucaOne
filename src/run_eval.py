#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/26 14:48
@project: LucaOne
@file: run.py
@desc: classifier_dropout -> classifier_dropout_prob
'''
import os
import sys, json
import datetime
import argparse
from collections import OrderedDict
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, PretrainedConfig
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from data_collator import *
    from encoder import Encoder
    from utils import set_seed, to_device, get_labels, get_parameter_number, save_model_parameters
    from multi_files_stream_dataloader import MultiFilesStreamLoader
    from models.lucaone_gplm import LucaGPLM
    from models.alphabet import Alphabet
    from models.lucaone_gplm_config import LucaGPLMConfig
    from batch_converter import BatchConverter
    from do_eval_all_checkpoints import do_eval_all_checkpoints
except ImportError as e:
    from src.data_collator import *
    from src.encoder import Encoder
    from src.utils import set_seed, to_device, get_labels, get_parameter_number, save_model_parameters
    from src.multi_files_stream_dataloader import MultiFilesStreamLoader
    from src.models.lucaone_gplm import LucaGPLM
    from src.models.alphabet import Alphabet
    from src.models.lucaone_gplm_config import LucaGPLMConfig
    from src.batch_converter import BatchConverter
    from src.do_eval_all_checkpoints import do_eval_all_checkpoints
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ['TRANSFORMERS_NO_ADVISORY_WARNINGS'] = 'true'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:1024'


def get_args():
    parser = argparse.ArgumentParser(description='LucaOne/LucaGPLM Eval')
    # for logging
    parser.add_argument("--tb_log_dir", type=str, default=None, required=True,
                        help="TensorBoard log every X updates steps.")
    parser.add_argument("--log_dir", type=str, default=None, required=True, help="Log every X updates steps.")

    # for model
    # modeling time str
    parser.add_argument("--time_str", type=str, default=None, help="the modeling time str")
    parser.add_argument("--hidden_size", type=int, default=None, required=True, help="hidden size(embedding vector size)")
    parser.add_argument("--num_attention_heads", type=int, default=None, required=True, help="num attention heads")
    parser.add_argument("--num_hidden_layers", type=int, default=None, required=True, help="num hidden layers")

    # for input sequence
    parser.add_argument("--max_length", default=10240, type=int, help="the max length of input sequence")
    parser.add_argument('--do_lower_case', action='store_true', help='whether to lower')
    parser.add_argument("--tokenization", action="store_true", help="whether to use tokenization")
    parser.add_argument("--tokenizer_dir", default=None, type=str, help="the pretrained tokenizer info path(subword-level)")
    parser.add_argument("--vocab_path", default=None, type=str, help="vocab path(char-level)")
    parser.add_argument('--add_special_tokens', action='store_true', help='add special tokens in the start and end position([CLS], [SEP]) of the input sequence')
    parser.add_argument('--padding', default='right', type=str, choices=["right", "left"], help='padding side type')
    parser.add_argument('--truncation', default='right', type=str, choices=["right", "left"], help='truncation side type')
    parser.add_argument('--no_token_type_embeddings', action='store_true', help='whether no token type embeddings')
    parser.add_argument('--no_position_embeddings', action='store_true', help='whether not to use absolute position embeddings')
    parser.add_argument('--dropout_prob', default=0.1, type=float, help="dropout_prob")

    # pooling_type
    parser.add_argument("--pooling_type", type=str, default=None,
                        choices=["none", "sum", "max", "avg", "attention", "context_attention", "weighted_attention",
                                 "value_attention", "transformer"],
                        help="pooling type for encoder")

    # for model selection
    parser.add_argument(
        '--model_type', 
        default="lucaone_gplm",
        type=str,
        choices=["lucaone_gplm"],
        help='the model type'
    )
    parser.add_argument(
        '--model_config', 
        type=str, 
        default=None, 
        help='the model config file path'
    )

    # for dataset
    parser.add_argument(
        "--train_data_dir",
        default=None,
        type=str,
        help="the train dataset dir path."
    )
    parser.add_argument(
        "--dev_data_dir",
        default=None,
        type=str,
        required=True,
        help="the validation dataset dir path."
    )
    parser.add_argument(
        "--test_data_dir",
        default=None,
        type=str,
        required=True,
        help="the testing dataset dir path."
    )

    # for label list
    parser.add_argument("--gene_mask_label_filepath", default=None, type=str,
                        help="the label filepath of token-level/gene_mask task(vocab).")
    parser.add_argument("--prot_mask_label_filepath", default=None, type=str,
                        help="the label filepath of token-level/prot_mask task(vocab).")
    parser.add_argument("--gene_type_label_filepath", default=None, type=str,
                        help="the label filepath of gene span-level/gene_type level task.")
    parser.add_argument("--prot_homo_label_filepath", default=None, type=str,
                        help="the label filepath of protein span-level/prot_homo level task.")
    parser.add_argument("--prot_site_label_filepath", default=None, type=str,
                        help="the label filepath of protein span-level/prot_site level task.")
    parser.add_argument("--prot_domain_label_filepath", default=None, type=str,
                        help="the label filepath of protein span-level/prot_domain level task.")
    parser.add_argument("--gene_taxonomy_label_filepath", default=None, type=str,
                        help="the label filepath of gene seq-level/gene_taxonomy task.")
    parser.add_argument("--prot_taxonomy_label_filepath", default=None, type=str,
                        help="the label filepath of protein seq-level/prot_taxonomy task.")
    parser.add_argument("--prot_keyword_label_filepath", default=None, type=str,
                        help="the label filepath of protein seq-level/prot_keyword task.")
    parser.add_argument("--prot_structure_label_filepath", default=None, type=str,
                        help="the label filepath of protein structure-level/prot_structure task.")
    parser.add_argument("--prot_secondary_label_filepath", default=None, type=str,
                        help="the label filepath of protein structure-level/prot_secondary task.")
    parser.add_argument("--prot_contact_label_filepath", default=None, type=str,
                        help="the label filepath of protein structure-level/prot_contact task.")
    parser.add_argument("--trans_label_filepath", default=None, type=str,
                        help="the label filepath of gene-protein pair-level/trans task.")

    # for pretraining task output mode
    parser.add_argument("--gene_mask_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of token-level/gene_mask task.")
    parser.add_argument("--prot_mask_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of token-level/prot_mask task.")
    parser.add_argument("--gene_type_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of gene span-level/gene_type task.")
    parser.add_argument("--prot_homo_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein span-level/prot_homo task.")
    parser.add_argument("--prot_site_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein span-level/prot_site task.")
    parser.add_argument("--prot_domain_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein span-level/prot_domain task.")
    parser.add_argument("--gene_taxonomy_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of gene seq-level/gene_taxonomy task.")
    parser.add_argument("--prot_taxonomy_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein seq-level/prot_taxonomy task.")
    parser.add_argument("--prot_keyword_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein seq-level/prot_keyword task.")
    parser.add_argument("--prot_structure_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein structure-level/prot_structure task.")
    parser.add_argument("--prot_secondary_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein structure-level/prot_secondary task.")
    parser.add_argument("--prot_contact_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of protein structure-level/prot_contact task.")
    parser.add_argument("--trans_output_mode", default=None, type=str,
                        choices=["binary_class", "multi_class", "multi_label", "regression"],
                        help="the output mode of gene-protein pair-level/trans task.")

    # the loss info for the pretraining tasks
    parser.add_argument("--ignore_index", type=int, default=-100, help="the ignore index.")
    parser.add_argument("--non_ignore", type=str, default=None, help="none ignore tasks.")
    parser.add_argument("--gene_mask_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of token-level/gene_mask task.")
    parser.add_argument("--prot_mask_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of token-level/prot_mask task.")
    parser.add_argument("--gene_type_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of gene span-level/gene_type task.")
    parser.add_argument("--prot_homo_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of protein span-level/prot_homo task.")
    parser.add_argument("--prot_site_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of protein span-level/prot_site task.")
    parser.add_argument("--prot_domain_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of protein span-level/prot_domain task.")
    parser.add_argument("--gene_taxonomy_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of gene seq-level/gene_taxonomy task.")
    parser.add_argument("--prot_taxonomy_loss_type", type=str, default="cce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of protein seq-level/prot_taxonomy task.")
    parser.add_argument("--prot_keyword_loss_type", type=str, default="bce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of protein seq-level/prot_keyword task.")
    parser.add_argument("--prot_structure_loss_type", type=str, default="mae",
                        choices=["l1", "l2"],
                        help="the loss type of protein structure-level/prot_structure task.")
    parser.add_argument("--prot_secondary_loss_type", type=str, default="mae",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of protein structure-level/prot_secondary task.")
    parser.add_argument("--prot_contact_loss_type", type=str, default="mae",
                        choices=["l1", "l2"],
                        help="the loss type of protein structure-level/prot_contact task.")
    parser.add_argument("--trans_loss_type", type=str, default="bce",
                        choices=["focal_loss", "bce", "multilabel_cce", "asl", "cce"],
                        help="the loss type of gene-protein pair-level/trans task.")

    # for classification fully connected layer
    parser.add_argument("--gene_mask_classifier_size", type=int, default=3072,
                        help="the classifier size of token-level/gene_mask task.")
    parser.add_argument("--prot_mask_classifier_size", type=int, default=3072,
                        help="the classifier size of token-level/prot_mask task.")
    parser.add_argument("--gene_type_classifier_size", type=int, default=3072,
                        help="the classifier size of span-level/gene_type task.")
    parser.add_argument("--prot_homo_classifier_size", type=int, default=3072,
                        help="the classifier size of span-level/prot_homo task.")
    parser.add_argument("--prot_site_classifier_size", type=int, default=3072,
                        help="the classifier size of span-level/prot_site task.")
    parser.add_argument("--prot_domain_classifier_size", type=int, default=3072,
                        help="the classifier size of span-level/prot_domain task.")
    parser.add_argument("--gene_taxonomy_classifier_size", type=int, default=3072,
                        help="the classifier size of seq-level/gene_taxonomy task.")
    parser.add_argument("--prot_taxonomy_classifier_size", type=int, default=3072,
                        help="the classifier size of seq-level/prot_taxonomy task.")
    parser.add_argument("--prot_keyword_classifier_size", type=int, default=3072,
                        help="the classifier size of seq-level/prot_keyword task.")
    parser.add_argument("--prot_structure_classifier_size", type=int, default=3072,
                        help="the classifier size of structure-level/prot_structure task.")
    parser.add_argument("--prot_secondary_classifier_size", type=int, default=3072,
                        help="the classifier size of structure-level/prot_secondary task.")
    parser.add_argument("--prot_contact_classifier_size", type=int, default=3072,
                        help="the classifier size of structure-level/prot_contact task.")
    parser.add_argument("--trans_classifier_size", type=int, default=3072,
                        help="the classifier size of gene protein pair-level/trans task.")

    # for stream dataloader
    parser.add_argument('--buffer_size', default=10240, type=int, help='buffer size for the dataset loading')
    parser.add_argument('--worker_num', default=1, type=int, help='worker number for the data loader.')
    parser.add_argument("--pretrain_task_level_type", default="all", type=str, required=True, help="pre train task level type")
    parser.add_argument("--pretrain_task_level_name", default="gene_mask,gene_type,gene_taxonomy,prot_mask,prot_site,prot_domain,prot_homo,prot_taxonomy,prot_keyword,prot_structure,trans",
                        type=str, required=True, help="pretrain task level name")

    # for training
    parser.add_argument("--local_rank", default=-1, type=int, help="main node local rank.")
    parser.add_argument("--seed", default=1111, type=int, help="random seed value.")
    parser.add_argument('--no_cuda', action='store_true', help='whether not to use GPU')
    parser.add_argument("--fp16", action="store_true",
                        help="whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument("--fp16_opt_level", type=str, default="O1",
                        help="for fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']. See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="batch size per GPU/CPU for evaluation.")
    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="the initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="weight decay if we apply some.")
    parser.add_argument("--decay_rate", default=0.9, type=float,
                        help="weight decay of learning rate.")
    parser.add_argument("--lr_update_steps", default=30000, type=int,
                        help="weight decay of learning rate.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20, type=int,
                        help="total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="set total number of training steps to perform.")
    parser.add_argument("--warmup_steps", default=-1, type=int,
                        help="linear warmup over warmup_steps.")
    parser.add_argument("--beta1", default=0.9, type=float,
                        help="Adamw beta1.")
    parser.add_argument("--beta2", default=0.98, type=float, help="Adamw beta2.")
    parser.add_argument("--do_train", action="store_true",
                        help="whether to run training.")
    parser.add_argument("--do_eval", action="store_true",
                        help="whether to run eval on the dev set.")
    parser.add_argument("--do_test", action="store_true",
                        help="whether to run predict on the test set.")
    parser.add_argument("--do_metrics", action="store_true",
                        help="whether to run eval metrics on the test set.")
    parser.add_argument("--evaluate_during_training", action="store_true",
                        help="where to evaluate during training.")
    parser.add_argument("--best_metric_type", type=str, default="f1",
                        choices=["loss", "acc", "jaccard", "prec", "recall", "f1", "fmax", "roc_auc", "pr_auc"],
                        help="which metric for model selected")
    parser.add_argument("--loss_logging_steps", type=int, default=100,
                        help="Loss log every X updates steps.")
    parser.add_argument("--logging_steps", type=int, default=10000,
                        help="Log every X updates steps.")
    parser.add_argument("--save_steps", type=int, default=10000,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="gradient accumulation steps.")
    parser.add_argument("--eval_start_epoch", type=int, default=-1,
                        help="the start epoch to eval.")
    parser.add_argument("--scheduler_type", type=str, default="step", choices=["step", "epoch"],
                        help="lr update scheduler type.")

    # for model save
    parser.add_argument("--save_all", action="store_true",
                        help="save all check-point")
    parser.add_argument("--delete_old", action="store_true",
                        help="delete old check-point")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="the output dir path")

    # for loss
    parser.add_argument("--multi_loss_strategy", default="manual_weight", type=str,
                        choices=["none", "manual_weight", "auto_weight", "dynamic_weight_average"],
                        help="multi-task loss fusion strategy")
    parser.add_argument("--pos_weight", default=None, type=float,
                        help="positive weight")
    parser.add_argument("--gene_mask_weight", type=float, default=1.0,
                        help="token-level/gene_mask task weight")
    parser.add_argument("--prot_mask_weight", type=float, default=1.0,
                        help="token-level/prot_mask task weight")
    parser.add_argument("--gene_type_weight", type=float, default=1.0,
                        help="gene span-level/gene_type task weight.")
    parser.add_argument("--prot_homo_weight", type=float, default=1.0,
                        help="protein span-level/prot_homo task weight.")
    parser.add_argument("--prot_site_weight", type=float, default=1.0,
                        help="protein span-level/prot_site task weight.")
    parser.add_argument("--prot_domain_weight", type=float, default=1.0,
                        help="protein span-level/prot_domain task weight.")
    parser.add_argument("--gene_taxonomy_weight", type=float, default=1.0,
                        help="gene seq-level/gene_taxonomy task weight.")
    parser.add_argument("--prot_taxonomy_weight", type=float, default=1.0,
                        help="protein seq-level/prot_taxonomy task weight.")
    parser.add_argument("--prot_keyword_weight", type=float, default=1.0,
                        help="protein seq-level/prot_keyword task weight.")
    parser.add_argument("--prot_structure_weight", type=float, default=1.0,
                        help="protein structure-level/prot_structure task weight")
    parser.add_argument("--prot_secondary_weight", type=float, default=1.0,
                        help="protein structure-level/prot_secondary task weight")
    parser.add_argument("--prot_contact_weight", type=float, default=1.0,
                        help="protein structure-level/prot_contact task weight")
    parser.add_argument("--trans_weight", type=float, default=1.0,
                        help="gene-protein pair-level/trans task weight")

    # pretrained model path
    parser.add_argument("--model_dirpath", default=None, type=str,
                        help="the pretrained model path")

    parser.add_argument("--no_token_dropout", action="store_true",
                        help="whether not to token dropout")
    parser.add_argument("--no_use_embed_layer_norm", action="store_true",
                        help="whether not to use emb layer norm")
    parser.add_argument("--no_use_last_layer_norm", action="store_true",
                        help="whether not to use last layer norm")
    parser.add_argument("--embed_scale", type=float, default=1.0,
                        help="embed scale")

    parser.add_argument("--pretrained_model_name", type=str, default=None,
                        help="pretrained_model_name")

    parser.add_argument("--processed_sample_cnt", default=1000000, type=int,
                        help="processed how many samples to write sample ids")
    parser.add_argument("--trained_checkpoint", default=None, type=int,
                        help="the checkpoint of continue to pretraining")
    parser.add_argument("--trained_checkpoint_continue", default=None, type=int,
                        help="the checkpoint of continue to pretraining")
    parser.add_argument("--trained_epoch", default=None, type=int,
                        help="the epoch of continue to pretraining")
    parser.add_argument("--removed_continue", action="store_true",
                        help="whether to remove done samples to continue training")
    parser.add_argument("--global_loss", default=0, type=float,
                        help="the global loss to continue training")
    parser.add_argument("--epoch_loss", default=0, type=float,
                        help="the epoch loss to continue training")

    # eval all checkpoints
    parser.add_argument("--all_checkpoints_dirpath", default=None, type=str,
                        help="the pretrained checkpoints model path")
    parser.add_argument("--all_checkpoints_part", default=None, type=str,
                        help="the pretrained checkpoints part")
    parser.add_argument("--eval_interval_step", default=400000, type=int,
                        help="the pretrained checkpoints eval_interval_step")
    parser.add_argument("--eval_min_step", default=0, type=int,
                        help="the pretrained checkpoints eval_min_step")
    parser.add_argument("--eval_max_step", default=20000000, type=int,
                        help="the pretrained checkpoints eval_max_step")
    parser.add_argument(
        "--use_bf16",
        action="store_true",
        help="whether to use bf16"
    )
    parser.add_argument(
        "--has_contact_head",
        action="store_true",
        help="whether to contain contact head"
    )
    input_args = parser.parse_args()
    return input_args


def put_value(obj, key1, key2, value):
    '''
    put the value
    :param obj:
    :param key1:
    :param key2:
    :param value:
    :return:
    '''
    if key1 not in obj:
        if value is None:
            obj[key1] = []
        else:
            obj[key1] = {}
    if value is None:
        obj[key1].append(key2)
    else:
        obj[key1][key2] = value


def check_args(args):
    '''
    check the args
    :param args:
    :return:
    '''
    assert args.log_dir is not None
    assert args.tb_log_dir is not None
    if args.tokenization:
        assert args.tokenizer_dir is not None and os.path.exists(args.tokenizer_dir)
    else:
        assert args.vocab_path is not None and os.path.exists(args.vocab_path)
    # assert args.train_data_dir is not None and os.path.exists(args.train_data_dir)
    assert args.output_dir is not None
    assert args.model_config is not None and os.path.exists(args.model_config)
    if not hasattr(args, "time_str") or args.time_str is None:
        now = datetime.datetime.now()
        args.time_str = now.strftime('%Y%m%d%H%M%S')
    assert args.all_checkpoints_dirpath is not None and os.path.exists(args.all_checkpoints_dirpath)
    args.pretrain_task_level_type = list(args.pretrain_task_level_type.split(","))
    for v in args.pretrain_task_level_type:
        assert v in ["all", "seq2seq_level", "token_level", "span_level", "seq_level", "structure_level", "pair_level"]
    loss_type = {}
    output_mode = {}
    gene_output_keys = {}
    gene_output_mode_dict = {}
    prot_output_keys = {}
    prot_output_mode_dict = {}
    pair_output_mode_dict = {}
    pair_output_keys = {}
    pooling_type = {}
    classifier_size = {}
    pretrain_tasks = {}
    loss_weights = {}
    output_keys = {}
    if "all" in args.pretrain_task_level_type or "token_level" in args.pretrain_task_level_type:
        if "gene_mask" in args.pretrain_task_level_name:
            assert args.gene_mask_label_filepath is not None
            assert args.gene_mask_output_mode is not None
            assert args.gene_mask_loss_type is not None
            if "token_level" not in pretrain_tasks:
                pretrain_tasks["token_level"] = []
            pretrain_tasks["token_level"].append("gene_mask")
            put_value(pooling_type, "token_level", "gene_mask", args.pooling_type)
            put_value(classifier_size, "token_level", "gene_mask", args.gene_mask_classifier_size)
            put_value(output_mode, "token_level", "gene_mask", args.gene_mask_output_mode)
            put_value(gene_output_mode_dict, "token_level", "gene_mask", args.gene_mask_output_mode)
            put_value(gene_output_keys, "token_level", "gene_mask", None)
            put_value(output_keys, "token_level", "gene_mask", None)
            put_value(loss_type, "token_level", "gene_mask", args.gene_mask_loss_type)
            put_value(loss_weights, "token_level", "gene_mask", args.gene_mask_weight)
        if "prot_mask" in args.pretrain_task_level_name:
            assert args.prot_mask_label_filepath is not None
            assert args.prot_mask_output_mode is not None
            assert args.prot_mask_loss_type is not None
            if "token_level" not in pretrain_tasks:
                pretrain_tasks["token_level"] = []
            pretrain_tasks["token_level"].append("prot_mask")
            put_value(pooling_type, "token_level", "prot_mask", args.pooling_type)
            put_value(classifier_size, "token_level", "prot_mask", args.prot_mask_classifier_size)
            put_value(output_mode, "token_level", "prot_mask", args.prot_mask_output_mode)
            put_value(prot_output_mode_dict, "token_level", "prot_mask", args.prot_mask_output_mode)
            put_value(prot_output_keys, "token_level", "prot_mask", None)
            put_value(output_keys, "token_level", "prot_mask", None)
            put_value(loss_type, "token_level", "prot_mask", args.prot_mask_loss_type)
            put_value(loss_weights, "token_level", "prot_mask", args.prot_mask_weight)
    if "whole_level" in args.pretrain_task_level_type:
        if "gene_mask" in args.pretrain_task_level_name:
            assert args.gene_mask_label_filepath is not None
            assert args.gene_mask_output_mode is not None
            assert args.gene_mask_loss_type is not None
            if "whole_level" not in pretrain_tasks:
                pretrain_tasks["whole_level"] = []
            pretrain_tasks["whole_level"].append("gene_mask")
            put_value(pooling_type, "whole_level", "gene_mask", args.pooling_type)
            put_value(classifier_size, "whole_level", "gene_mask", args.gene_mask_classifier_size)
            put_value(output_mode, "whole_level", "gene_mask", args.gene_mask_output_mode)
            put_value(loss_type, "whole_level", "gene_mask", args.gene_mask_loss_type)
            put_value(gene_output_mode_dict, "whole_level", "gene_mask", args.gene_mask_output_mode)
            put_value(loss_weights, "token_level", "gene_mask", args.gene_mask_weight)
            put_value(gene_output_keys, "token_level", "gene_mask", None)
            put_value(output_keys, "token_level", "gene_mask", None)
        if "prot_mask" in args.pretrain_task_level_name:
            assert args.prot_mask_label_filepath is not None
            assert args.prot_mask_output_mode is not None
            assert args.prot_mask_loss_type is not None
            if "whole_level" not in pretrain_tasks:
                pretrain_tasks["whole_level"] = []
            pretrain_tasks["whole_level"].append("prot_mask")
            put_value(pooling_type, "whole_level", "prot_mask", args.pooling_type)
            put_value(classifier_size, "whole_level", "prot_mask", args.prot_mask_classifier_size)
            put_value(output_mode, "whole_level", "prot_mask", args.prot_mask_output_mode)
            put_value(loss_type, "whole_level", "prot_mask", args.prot_mask_loss_type)
            put_value(prot_output_mode_dict, "whole_level", "prot_mask", args.prot_mask_output_mode)
            put_value(loss_weights, "token_level", "prot_mask", args.prot_mask_weight)
            put_value(prot_output_keys, "token_level", "prot_mask", None)
            put_value(output_keys, "token_level", "prot_mask", None)

    if "all" in args.pretrain_task_level_type or "span_level" in args.pretrain_task_level_type:
        if "gene_type" in args.pretrain_task_level_name:
            assert args.gene_type_label_filepath is not None
            assert args.gene_type_output_mode is not None
            assert args.gene_type_loss_type is not None
            if "span_level" not in pretrain_tasks:
                pretrain_tasks["span_level"] = []
            pretrain_tasks["span_level"].append("gene_type")
            put_value(pooling_type, "span_level", "gene_type", args.pooling_type)
            put_value(classifier_size, "span_level", "gene_type", args.gene_type_classifier_size)
            put_value(output_mode, "span_level", "gene_type", args.gene_type_output_mode)
            put_value(loss_type, "span_level", "gene_type", args.gene_type_loss_type)
            put_value(gene_output_mode_dict, "span_level", "gene_type", args.gene_type_output_mode)
            put_value(loss_weights, "span_level", "gene_type", args.gene_type_weight)
            put_value(gene_output_keys, "span_level", "gene_type", None)
            put_value(output_keys, "span_level", "gene_type", None)
        if "prot_homo" in args.pretrain_task_level_name:
            assert args.prot_homo_label_filepath is not None
            assert args.prot_homo_output_mode is not None
            assert args.prot_homo_loss_type is not None
            if "span_level" not in pretrain_tasks:
                pretrain_tasks["span_level"] = []
            pretrain_tasks["span_level"].append("prot_homo")
            put_value(pooling_type, "span_level", "prot_homo", args.pooling_type)
            put_value(classifier_size, "span_level", "prot_homo", args.prot_homo_classifier_size)
            put_value(output_mode, "span_level", "prot_homo", args.prot_homo_output_mode)
            put_value(loss_type, "span_level", "prot_homo", args.prot_homo_loss_type)
            put_value(prot_output_mode_dict, "span_level", "prot_homo", args.prot_homo_output_mode)
            put_value(loss_weights, "span_level", "prot_homo", args.prot_homo_weight)
            put_value(prot_output_keys, "span_level", "prot_homo", None)
            put_value(output_keys, "span_level", "prot_homo", None)
        if "prot_site" in args.pretrain_task_level_name:
            assert args.prot_site_label_filepath is not None
            assert args.prot_site_output_mode is not None
            assert args.prot_site_loss_type is not None
            if "span_level" not in pretrain_tasks:
                pretrain_tasks["span_level"] = []
            pretrain_tasks["span_level"].append("prot_site")
            put_value(pooling_type, "span_level", "prot_site", args.pooling_type)
            put_value(classifier_size, "span_level", "prot_site", args.prot_site_classifier_size)
            put_value(output_mode, "span_level", "prot_site", args.prot_site_output_mode)
            put_value(loss_type, "span_level", "prot_site", args.prot_site_loss_type)
            put_value(prot_output_mode_dict, "span_level", "prot_site", args.prot_site_output_mode)
            put_value(loss_weights, "span_level", "prot_site", args.prot_site_weight)
            put_value(prot_output_keys, "span_level", "prot_site", None)
            put_value(output_keys, "span_level", "prot_site", None)
        if "prot_domain" in args.pretrain_task_level_name:
            assert args.prot_domain_label_filepath is not None
            assert args.prot_domain_output_mode is not None
            assert args.prot_domain_loss_type is not None
            if "span_level" not in pretrain_tasks:
                pretrain_tasks["span_level"] = []
            pretrain_tasks["span_level"].append("prot_domain")
            put_value(pooling_type, "span_level", "prot_domain", args.pooling_type)
            put_value(classifier_size, "span_level", "prot_domain", args.prot_domain_classifier_size)
            put_value(output_mode, "span_level", "prot_domain", args.prot_domain_output_mode)
            put_value(loss_type, "span_level", "prot_domain", args.prot_domain_loss_type)
            put_value(prot_output_mode_dict, "span_level", "prot_domain", args.prot_domain_output_mode)
            put_value(loss_weights, "span_level", "prot_domain", args.prot_domain_weight)
            put_value(prot_output_keys, "span_level", "prot_domain", None)
            put_value(output_keys, "span_level", "prot_domain", None)
        # pretrain_tasks["span_level"] = ["gene_type", "prot_homo", "prot_site", "prot_domain]

    if "all" in args.pretrain_task_level_type or "seq_level" in args.pretrain_task_level_type:
        if "gene_taxonomy" in args.pretrain_task_level_name:
            assert args.gene_taxonomy_label_filepath is not None
            assert args.gene_taxonomy_output_mode is not None
            assert args.gene_taxonomy_loss_type is not None
            if "seq_level" not in pretrain_tasks:
                pretrain_tasks["seq_level"] = []
            pretrain_tasks["seq_level"].append("gene_taxonomy")
            put_value(pooling_type, "seq_level", "gene_taxonomy", args.pooling_type)
            put_value(classifier_size, "seq_level", "gene_taxonomy", args.gene_taxonomy_classifier_size)
            put_value(output_mode, "seq_level", "gene_taxonomy", args.gene_taxonomy_output_mode)
            put_value(loss_type, "seq_level", "gene_taxonomy", args.gene_taxonomy_loss_type)
            put_value(gene_output_mode_dict, "seq_level", "gene_taxonomy", args.gene_taxonomy_output_mode)
            put_value(loss_weights, "seq_level", "gene_taxonomy", args.gene_taxonomy_weight)
            put_value(gene_output_keys, "seq_level", "gene_taxonomy", None)
            put_value(output_keys, "seq_level", "gene_taxonomy", None)
        if "prot_taxonomy" in args.pretrain_task_level_name:
            assert args.prot_taxonomy_label_filepath is not None
            assert args.prot_taxonomy_output_mode is not None
            assert args.prot_taxonomy_loss_type is not None
            if "seq_level" not in pretrain_tasks:
                pretrain_tasks["seq_level"] = []
            pretrain_tasks["seq_level"].append("prot_taxonomy")
            put_value(pooling_type, "seq_level", "prot_taxonomy", args.pooling_type)
            put_value(classifier_size, "seq_level", "prot_taxonomy", args.prot_taxonomy_classifier_size)
            put_value(output_mode, "seq_level", "prot_taxonomy", args.prot_taxonomy_output_mode)
            put_value(loss_type, "seq_level", "prot_taxonomy", args.prot_taxonomy_loss_type)
            put_value(prot_output_mode_dict, "seq_level", "prot_taxonomy", args.prot_taxonomy_output_mode)
            put_value(loss_weights, "seq_level", "prot_taxonomy", args.prot_taxonomy_weight)
            put_value(prot_output_keys, "seq_level", "prot_taxonomy", None)
            put_value(output_keys, "seq_level", "prot_taxonomy", None)
        if "prot_keyword" in args.pretrain_task_level_name:
            assert args.prot_keyword_label_filepath is not None
            assert args.prot_keyword_output_mode is not None
            assert args.prot_keyword_loss_type is not None
            if "seq_level" not in pretrain_tasks:
                pretrain_tasks["seq_level"] = []
            pretrain_tasks["seq_level"].append("prot_keyword")
            put_value(pooling_type, "seq_level", "prot_keyword", args.pooling_type)
            put_value(classifier_size, "seq_level", "prot_keyword", args.prot_keyword_classifier_size)
            put_value(output_mode, "seq_level", "prot_keyword", args.prot_keyword_output_mode)
            put_value(loss_type, "seq_level", "prot_keyword", args.prot_keyword_loss_type)
            put_value(prot_output_mode_dict, "seq_level", "prot_keyword", args.prot_keyword_output_mode)
            put_value(loss_weights, "seq_level", "prot_keyword", args.prot_keyword_weight)
            put_value(prot_output_keys, "seq_level", "prot_keyword", None)
            put_value(output_keys, "seq_level", "prot_keyword", None)
        # pretrain_tasks["seq_level"] = ["gene_taxonomy", "prot_taxonomy", "prot_keyword"]

    if "all" in args.pretrain_task_level_type or "pair_level" in args.pretrain_task_level_type:
        if "trans" in args.pretrain_task_level_name:
            assert args.trans_label_filepath is not None
            assert args.trans_output_mode is not None
            assert args.trans_loss_type is not None
            if "pair_level" not in pretrain_tasks:
                pretrain_tasks["pair_level"] = []
            pretrain_tasks["pair_level"].append("trans")
            put_value(pooling_type, "pair_level", "trans", args.pooling_type)
            put_value(classifier_size, "pair_level", "trans", args.trans_classifier_size)
            put_value(output_mode, "pair_level", "trans", args.trans_output_mode)
            put_value(loss_type, "pair_level", "trans", args.trans_loss_type)
            put_value(pair_output_mode_dict, "pair_level", "trans", args.trans_output_mode)
            put_value(loss_weights, "pair_level", "trans", args.trans_weight)
            put_value(pair_output_keys, "pair_level", "trans", None)
            put_value(output_keys, "pair_level", "trans", None)
    if "all" in args.pretrain_task_level_type or "structure_level" in args.pretrain_task_level_type:
        if "prot_structure" in args.pretrain_task_level_name:
            assert args.prot_structure_label_filepath is not None
            assert args.prot_structure_output_mode is not None
            assert args.prot_structure_loss_type is not None
            if "structure_level" not in pretrain_tasks:
                pretrain_tasks["structure_level"] = []
            pretrain_tasks["structure_level"].append("prot_structure")
            put_value(pooling_type, "structure_level", "prot_structure", args.pooling_type)
            put_value(classifier_size, "structure_level", "prot_structure", args.prot_structure_classifier_size)
            put_value(output_mode, "structure_level", "prot_structure", args.prot_structure_output_mode)
            put_value(loss_type, "structure_level", "prot_structure", args.prot_structure_loss_type)
            put_value(prot_output_mode_dict, "structure_level", "prot_structure", args.prot_structure_output_mode)
            put_value(loss_weights, "structure_level", "prot_structure", args.prot_structure_weight)
            put_value(prot_output_keys, "structure_level", "prot_structure", None)
            put_value(output_keys, "structure_level", "prot_structure", None)

        if "prot_secondary" in args.pretrain_task_level_name:
            assert args.prot_secondary_label_filepath is not None
            assert args.prot_secondary_output_mode is not None
            assert args.prot_secondary_loss_type is not None
            if "structure_level" not in pretrain_tasks:
                pretrain_tasks["structure_level"] = []
            pretrain_tasks["structure_level"].append("prot_secondary")
            put_value(pooling_type, "structure_level", "prot_secondary", args.pooling_type)
            put_value(classifier_size, "structure_level", "prot_secondary", args.prot_secondary_classifier_size)
            put_value(output_mode, "structure_level", "prot_secondary", args.prot_secondary_output_mode)
            put_value(loss_type, "structure_level", "prot_secondary", args.prot_secondary_loss_type)
            put_value(prot_output_mode_dict, "structure_level", "prot_secondary", args.prot_secondary_output_mode)
            put_value(loss_weights, "structure_level", "prot_secondary", args.prot_secondary_weight)
            put_value(prot_output_keys, "structure_level", "prot_secondary", None)
            put_value(output_keys, "structure_level", "prot_secondary", None)

        if "prot_contact" in args.pretrain_task_level_name:
            assert args.prot_contact_label_filepath is not None
            assert args.prot_contact_output_mode is not None
            assert args.prot_contact_loss_type is not None
            if "structure_level" not in pretrain_tasks:
                pretrain_tasks["structure_level"] = []
            pretrain_tasks["structure_level"].append("prot_contact")
            put_value(pooling_type, "structure_level", "prot_contact", args.pooling_type)
            put_value(classifier_size, "structure_level", "prot_contact", args.prot_contact_classifier_size)
            put_value(output_mode, "structure_level", "prot_contact", args.prot_contact_output_mode)
            put_value(loss_type, "structure_level", "prot_contact", args.prot_contact_loss_type)
            put_value(prot_output_mode_dict, "structure_level", "prot_contact", args.prot_contact_output_mode)
            put_value(loss_weights, "structure_level", "prot_contact", args.prot_contact_weight)
            put_value(prot_output_keys, "structure_level", "prot_contact", None)
            put_value(output_keys, "structure_level", "prot_contact", None)

    # non_ignore
    if args.non_ignore:
        non_ignore_values = args.non_ignore.split(",")
        non_ignore_values = [v.strip() for v in non_ignore_values if v.strip()]
        for s in non_ignore_values:
            assert s.strip() in ["gene_mask", "prot_mask", "gene_type", "gene_taxonomy",
                                 "prot_homo", "prot_site", "prot_domain", "prot_taxonomy", "prot_keyword",
                                 "prot_structure", "prot_secondary", "prot_contact",
                                 "trans"]
        args.non_ignore = list(set(non_ignore_values))
    args.loss_type = loss_type
    args.output_mode = output_mode
    args.pooling_type = pooling_type
    args.classifier_size = classifier_size
    args.pretrain_tasks = pretrain_tasks
    args.gene_output_mode_dict = gene_output_mode_dict
    args.prot_output_mode_dict = prot_output_mode_dict
    args.pair_output_mode_dict = pair_output_mode_dict
    args.loss_weights = loss_weights
    args.gene_output_keys = gene_output_keys
    args.prot_output_keys = prot_output_keys
    args.pair_output_keys = pair_output_keys
    args.output_keys = output_keys


def get_model(args):
    '''
    create tokenizer, model config, model
    :param args:
    :return:
    '''

    # four type of models
    if args.model_type in ["lucaone_gplm"]:
        config_class, model_class = LucaGPLMConfig, LucaGPLM
    else:
        raise Exception("Not support model_type=%s" % args.model_type)

    # model config
    model_config: PretrainedConfig = config_class.from_json_file(args.model_config)

    # create tokenizer
    if args.model_dirpath:
        print("model_dirpath:")
        print(args.model_dirpath)
        # load exists checkpoint
        tokenizer_dir = os.path.join(args.model_dirpath, "tokenizer")
        assert os.path.exists(tokenizer_dir)
        if args.tokenization:
            print("AutoTokenizer, tokenizer dir: %s" % tokenizer_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_dir,
                do_lower_case=args.do_lower_case,
                truncation_side=args.truncation
            )
        elif args.model_type in ["lucaone_gplm"]:
            print("Alphabet, vocab path: %s" % tokenizer_dir)
            tokenizer = Alphabet.from_pretrained(tokenizer_dir)
        else:
            print("BertTokenizer, vocab path: %s" % tokenizer_dir)
            tokenizer = BertTokenizer.from_pretrained(
                tokenizer_dir,
                do_lower_case=args.do_lower_case,
                truncation_side=args.truncation
            )
    else:
        if args.tokenization:
            print("AutoTokenizer, tokenizer dir: %s" % args.tokenizer_dir)
            tokenizer = AutoTokenizer.from_pretrained(
                args.tokenizer_dir,
                do_lower_case=args.do_lower_case,
                truncation_side=args.truncation
            )
        elif args.model_type in ["lucaone_gplm"]:
            print("Alphabet, tokenizer_dir: %s" % args.tokenizer_dir)
            tokenizer = Alphabet.from_predefined(model_config.alphabet)
        else:
            print("BertTokenizer, vocab path: %s" % args.vocab_path)
            tokenizer = BertTokenizer(
                args.vocab_path,
                do_lower_case=args.do_lower_case,
                truncation_side=args.truncation
            )

    # model important parameters
    model_config.hidden_size = args.hidden_size
    model_config.num_attention_heads = args.num_attention_heads
    model_config.num_hidden_layers = args.num_hidden_layers
    if args.dropout_prob is not None and args.dropout_prob > -1:
        model_config.attention_probs_dropout_prob = args.dropout_prob
        model_config.classifier_dropout_prob = args.dropout_prob
        model_config.classifier_dropout = args.dropout_prob
        model_config.hidden_dropout_prob = args.dropout_prob
    model_config.ignore_index = args.ignore_index
    model_config.token_dropout = not args.no_token_dropout
    model_config.use_embed_layer_norm = not args.no_use_embed_layer_norm
    model_config.use_last_layer_norm = not args.no_use_last_layer_norm
    if 0 < args.embed_scale < 1:
        model_config.embed_scale = args.embed_scale

    # input parameters
    model_config.has_contact_head = args.has_contact_head
    model_config.vocab_size = tokenizer.vocab_size
    model_config.max_position_embeddings = args.max_length
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.no_token_type_embeddings = args.no_token_type_embeddings
    model_config.no_position_embeddings = args.no_position_embeddings

    # all the classifiers
    model_config.gene_mask_classifier_output_size = args.gene_mask_classifier_size
    model_config.prot_mask_classifier_output_size = args.prot_mask_classifier_size
    model_config.gene_type_classifier_output_size = args.gene_type_classifier_size
    model_config.prot_homo_classifier_output_size = args.prot_homo_classifier_size
    model_config.prot_site_classifier_output_size = args.prot_site_classifier_size
    model_config.prot_domain_classifier_output_size = args.prot_domain_classifier_size
    model_config.gene_taxonomy_classifier_output_size = args.gene_taxonomy_classifier_size
    model_config.prot_taxonomy_classifier_output_size = args.prot_taxonomy_classifier_size
    model_config.prot_keyword_classifier_output_size = args.prot_keyword_classifier_size
    model_config.prot_structure_classifier_output_size = args.prot_structure_classifier_size
    model_config.prot_secondary_classifier_output_size = args.prot_secondary_classifier_size
    model_config.prot_contact_classifier_output_size = args.prot_contact_classifier_size
    model_config.trans_classifier_output_size = args.trans_classifier_size

    # all the label nums
    if "token_level" in args.label_size and "gene_mask" in args.label_size["token_level"]:
        model_config.gene_mask_label_num = args.label_size["token_level"]["gene_mask"]
    if "token_level" in args.label_size and "prot_mask" in args.label_size["token_level"]:
        model_config.prot_mask_label_num = args.label_size["token_level"]["prot_mask"]
    if "span_level" in args.label_size and "gene_type" in args.label_size["span_level"]:
        model_config.gene_type_label_num = args.label_size["span_level"]["gene_type"]
    if "span_level" in args.label_size and "prot_homo" in args.label_size["span_level"]:
        model_config.prot_homo_label_num = args.label_size["span_level"]["prot_homo"]
    if "span_level" in args.label_size and "prot_site" in args.label_size["span_level"]:
        model_config.prot_site_label_num = args.label_size["span_level"]["prot_site"]
    if "span_level" in args.label_size and "prot_domain" in args.label_size["span_level"]:
        model_config.prot_domain_label_num = args.label_size["span_level"]["prot_domain"]
    if "seq_level" in args.label_size and "gene_taxonomy" in args.label_size["seq_level"]:
        model_config.gene_taxonomy_label_num = args.label_size["seq_level"]["gene_taxonomy"]
    if "seq_level" in args.label_size and "prot_taxonomy" in args.label_size["seq_level"]:
        model_config.prot_taxonomy_label_num = args.label_size["seq_level"]["prot_taxonomy"]
    if "seq_level" in args.label_size and "prot_keyword" in args.label_size["seq_level"]:
        model_config.prot_keyword_label_num = args.label_size["seq_level"]["prot_keyword"]
    if "structure_level" in args.label_size and "prot_structure" in args.label_size["structure_level"]:
        model_config.prot_structure_label_num = args.label_size["structure_level"]["prot_structure"]
    if "structure_level" in args.label_size and "prot_secondary" in args.label_size["structure_level"]:
        model_config.prot_secondary_label_num = args.label_size["structure_level"]["prot_secondary"]
    if "structure_level" in args.label_size and "prot_contact" in args.label_size["structure_level"]:
        model_config.prot_contact_label_num = args.label_size["structure_level"]["prot_contact"]
    if "pair_level" in args.label_size and "trans" in args.label_size["pair_level"]:
        model_config.trans_label_num = args.label_size["pair_level"]["trans"]

    # load the pretrained model or create the model
    if args.model_dirpath:
        # load exists checkpoint
        print("load pretrained model: %s" % args.model_dirpath)
        try:
            model = model_class.from_pretrained(args.model_dirpath, args=args)
        except Exception as e:
            model = model_class(model_config, args=args)
            pretrained_net_dict = torch.load(
                os.path.join(args.model_dirpath, "pytorch.pth"),
                map_location=torch.device("cpu"),
                weights_only=True
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
                else:
                    print("name: %s removed" % name)
            diff = model_state_dict_keys.difference(new_state_dict.keys())
            if diff:
                print("diff:")
                print(diff)
            model.load_state_dict(new_state_dict)
    else:
        # create model
        model = model_class(model_config, args)
   # save_model_parameters(model, os.path.join(args.output_dir, "init_parameters"))
    '''
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(name, param.data)
    '''

    return model_config, model, tokenizer


def get_label_size(label_filepath):
    '''
    load label size
    :param label_filepath: label list
    :return:
    '''
    if label_filepath:
        cur_labels = get_labels(label_filepath, header=True if label_filepath.endswith(".csv") else False)
        return len(cur_labels)
    else:
        raise Exception("Label path: %s not exists." % label_filepath)


def load_label_size_dict(args):
    '''
    load all the label size
    :param args:
    :return:
    '''
    label_size = {}
    gene_label_size_dict = {}
    prot_label_size_dict = {}
    pair_label_size_dict = {}
    if "all" in args.pretrain_task_level_type or "token_level" in args.pretrain_task_level_type:
        if "gene_mask" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.gene_mask_label_filepath)
            put_value(label_size, "token_level", "gene_mask", cur_label_size)
            put_value(gene_label_size_dict, "token_level", "gene_mask", cur_label_size)
            # put_value(prot_label_size_dict, "token_level", "gene_mask", cur_label_size)
        if "prot_mask" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_mask_label_filepath)
            put_value(label_size, "token_level", "prot_mask", cur_label_size)
            # put_value(gene_label_size_dict, "token_level", "prot_mask", cur_label_size)
            put_value(prot_label_size_dict, "token_level", "prot_mask", cur_label_size)
    if "all" in args.pretrain_task_level_type or "span_level" in args.pretrain_task_level_type:
        if "gene_type" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.gene_type_label_filepath)
            put_value(label_size, "span_level", "gene_type", cur_label_size)
            put_value(gene_label_size_dict, "span_level", "gene_type", cur_label_size)
        if "prot_homo" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_homo_label_filepath)
            put_value(label_size, "span_level", "prot_homo", cur_label_size)
            put_value(prot_label_size_dict, "span_level", "prot_homo", cur_label_size)
        if "prot_site" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_site_label_filepath)
            put_value(label_size, "span_level", "prot_site", cur_label_size)
            put_value(prot_label_size_dict, "span_level", "prot_site", cur_label_size)
        if "prot_domain" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_domain_label_filepath)
            put_value(label_size, "span_level", "prot_domain", cur_label_size)
            put_value(prot_label_size_dict, "span_level", "prot_domain", cur_label_size)
    if "all" in args.pretrain_task_level_type or "seq_level" in args.pretrain_task_level_type:
        if "gene_taxonomy" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.gene_taxonomy_label_filepath)
            put_value(label_size, "seq_level", "gene_taxonomy", cur_label_size)
            put_value(gene_label_size_dict, "seq_level", "gene_taxonomy", cur_label_size)
        if "prot_taxonomy" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_taxonomy_label_filepath)
            put_value(label_size, "seq_level", "prot_taxonomy", cur_label_size)
            put_value(prot_label_size_dict, "seq_level", "prot_taxonomy", cur_label_size)
        if "prot_keyword" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_keyword_label_filepath)
            put_value(label_size, "seq_level", "prot_keyword", cur_label_size)
            put_value(prot_label_size_dict, "seq_level", "prot_keyword", cur_label_size)
    if "all" in args.pretrain_task_level_type or "structure_level" in args.pretrain_task_level_type:
        if "prot_structure" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_structure_label_filepath)
            put_value(label_size, "structure_level", "prot_structure", cur_label_size)
            put_value(prot_label_size_dict, "structure_level", "prot_structure", cur_label_size)
        if "prot_secondary" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_secondary_label_filepath)
            put_value(label_size, "structure_level", "prot_secondary", cur_label_size)
            put_value(prot_label_size_dict, "structure_level", "prot_secondary", cur_label_size)
        if "prot_contact" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.prot_contact_label_filepath)
            put_value(label_size, "structure_level", "prot_contact", cur_label_size)
            put_value(prot_label_size_dict, "structure_level", "prot_contact", cur_label_size)

    if "all" in args.pretrain_task_level_type or "pair_level" in args.pretrain_task_level_type:
        if "trans" in args.pretrain_task_level_name:
            cur_label_size = get_label_size(args.trans_label_filepath)
            put_value(label_size, "pair_level", "trans", cur_label_size)
            put_value(pair_label_size_dict, "pair_level", "trans", cur_label_size)
    return gene_label_size_dict, prot_label_size_dict, pair_label_size_dict, label_size


def create_logger(args):
    '''
    create logger
    :param args:
    :return:
    '''
    if args.local_rank in [-1, 0]:
        # create the output dir
        if os.path.exists(args.output_dir):
            raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
        else:
            os.makedirs(args.output_dir)
        # create the logs dir
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir)
        log_fp = open(os.path.join(args.log_dir, "logs.txt"), "a+")
        # create tensorboard logs dir
        if not os.path.exists(args.tb_log_dir):
            os.makedirs(args.tb_log_dir)
        for dataset_type in ["train", "dev", "test"]:
            processed_samples_dir_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                                      "processed_samples",
                                                      args.time_str,
                                                      dataset_type)
            if not os.path.exists(processed_samples_dir_path):
                os.makedirs(processed_samples_dir_path)
        exception_path = "../exception/%s/" % args.time_str
        if not os.path.exists(exception_path):
            os.makedirs(exception_path)
        debug_path = "../debug/%s/" % args.time_str
        if not os.path.exists(debug_path):
            os.makedirs(debug_path)
        print("Output dir, logger, tb-logger created succeed.")
    else:
        log_fp = None
    return log_fp


def create_device(args):
    '''
    create device
    :param args:
    :return:
    '''
    if args.no_cuda or not torch.cuda.is_available():
        device = torch.device("cpu")
        args.n_gpu = 0
    else:
        args.n_gpu = torch.cuda.device_count()
        if args.n_gpu > 1:
            torch.cuda.set_device(args.local_rank)
            device = torch.device("cuda", args.local_rank)
            dist.init_process_group(backend="nccl", timeout=datetime.timedelta(seconds=54000))
            if args.local_rank == 0:
                print('world size: %d' % dist.get_world_size())
        else:
            device = torch.device("cuda")
    return device


def create_collator(args, tokenizer):
    '''
    create data collactor
    :param args:
    :param tokenizer:
    :return:
    '''
    dcForLanguageModeling, dcForWholeWordMask, \
    dcForTokenClassification, dcForSequenceClassification, \
    dcForStructureRegression, dcForPairClassification, dcForSeq2Seq = None, None, None, None, None, None, None
    if "token_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
        dcForLanguageModeling = DataCollatorForLanguageModeling(tokenizer)
    if "whole_level" in args.pretrain_task_level_type:
        dcForWholeWordMask = DataCollatorForWholeWordMask(tokenizer)
    if "span_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
        dcForTokenClassification = DataCollatorForTokenClassification(tokenizer,
                                                                      padding="max_length",
                                                                      max_length=args.max_length,
                                                                      label_pad_token_id=args.ignore_index,
                                                                      label_keys=["gene_type", "prot_homo", "prot_site", "prot_domain"],
                                                                      multi_label={
                                                                          "gene_type": args.output_mode["span_level"]["gene_type"] == "multi_label",
                                                                          "prot_homo": args.output_mode["span_level"]["prot_homo"] == "multi_label",
                                                                          "prot_site": args.output_mode["span_level"]["prot_site"] == "multi_label",
                                                                          "prot_domain": args.output_mode["span_level"]["prot_domain"] == "multi_label"
                                                                      },
                                                                      label_num={
                                                                          "gene_type": args.label_size["span_level"]["gene_type"],
                                                                          "prot_homo": args.label_size["span_level"]["prot_homo"],
                                                                          "prot_site": args.label_size["span_level"]["prot_site"],
                                                                          "prot_domain": args.label_size["span_level"]["prot_domain"],
                                                                      })
    if "seq_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
        dcForSequenceClassification = DataCollatorForSequenceClassification(tokenizer,
                                                                            padding="max_length",
                                                                            max_length=args.max_length,
                                                                            label_keys=["gene_taxonomy",
                                                                                        "prot_taxonomy",
                                                                                        "prot_keyword"],
                                                                            multi_label={
                                                                                "gene_taxonomy":
                                                                                    args.output_mode["seq_level"]["gene_taxonomy"] == "multi_label",
                                                                                "prot_taxonomy":
                                                                                    args.output_mode["seq_level"]["prot_taxonomy"] == "multi_label",
                                                                                "prot_keyword":
                                                                                    args.output_mode["seq_level"]["prot_keyword"] == "multi_label"
                                                                            },
                                                                            label_num={
                                                                                "gene_taxonomy":
                                                                                    args.label_size["seq_level"]["gene_taxonomy"],
                                                                                "prot_taxonomy":
                                                                                    args.label_size["seq_level"]["prot_taxonomy"],
                                                                                "prot_keyword":
                                                                                    args.label_size["seq_level"]["prot_keyword"]
                                                                            })
    if "structure_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
        dcForStructureRegression = DataCollatorForStructureRegression(tokenizer,
                                                                      padding="max_length",
                                                                      max_length=args.max_length,
                                                                      label_keys=["prot_structure", "prot_secondary", "prot_contact"],
                                                                      multi_label={
                                                                          "prot_structure":
                                                                              args.output_mode["structure_level"]["prot_structure"] == "multi_label",
                                                                          "prot_secondary":
                                                                              args.output_mode["structure_level"]["prot_secondary"] == "multi_label",
                                                                          "prot_contact":
                                                                              args.output_mode["structure_level"]["prot_contact"] == "multi_label"
                                                                      },
                                                                      label_num={
                                                                          "prot_structure":
                                                                              args.label_size["structure_level"]["prot_structure"],
                                                                          "prot_secondary":
                                                                              args.label_size["structure_level"]["prot_secondary"],
                                                                          "prot_contact":
                                                                              args.label_size["structure_level"]["prot_contact"]
                                                                      })
    if "pair_level" in args.pretrain_task_level_type or "all" in args.pretrain_task_level_type:
        dcForPairClassification = DataCollatorForPairClassification(tokenizer,
                                                                    padding="max_length",
                                                                    max_length=args.max_length,
                                                                    label_keys=["trans"],
                                                                    multi_label={
                                                                        "trans": args.output_mode["pair_level"]["trans"] == "multi_label"
                                                                    },
                                                                    label_num={
                                                                        "trans": args.label_size["pair_level"]["trans"]
                                                                    })

    if "seq2seq_level" in args.pretrain_task_level_type:
        raise Exception("not support seq2seq_level task.")
    return dcForLanguageModeling, dcForWholeWordMask, \
           dcForTokenClassification, dcForSequenceClassification, \
           dcForStructureRegression, dcForPairClassification, dcForSeq2Seq


def main():
    # get args
    args = get_args()
    # print(os.environ['LOCAL_RANK'])
    # args.local_rank = os.environ['LOCAL_RANK']

    # check args
    check_args(args)

    # create log dir
    log_fp = create_logger(args)

    # load all labels size
    args.gene_label_size_dict, args.prot_label_size_dict, args.pair_label_size_dict, args.label_size \
        = load_label_size_dict(args)

    # device
    args.device = create_device(args)

    # create model
    model_config, model, tokenizer = get_model(args)
    args.vocab_size = tokenizer.vocab_size

    # encoder config
    encoder_config = {
        "add_special_tokens": args.add_special_tokens,
        "max_length": args.max_length,
        "truncation": args.truncation,
        "padding": args.padding,
        "tokenizer_dir": args.tokenizer_dir,
        "vocab_path": args.gene_mask_label_filepath,
        "vocab_size": args.vocab_size,
    }

    # file row parser
    # 文件记录解析函数
    encoder = Encoder(config=encoder_config,
                      tokenizer=tokenizer,
                      tokenization=args.tokenization,
                      no_token_type_embeddings=args.no_token_type_embeddings,
                      non_ignore=args.non_ignore,
                      ignore_index=args.ignore_index,
                      model_type=args.model_type)
    # 基因-蛋白pair对数据集
    if "all" in args.pretrain_task_level_type or "pair_level" in args.pretrain_task_level_type:
        if args.model_type in ["lucaone_gplm"]:
            parse_row_func = encoder.encode_char_pair
        else:
            parse_row_func = encoder.encode_pair
    else:
        # 基因或者蛋白单记录数据集
        if args.model_type in ["lucaone_gplm"]:
            parse_row_func = encoder.encode_char_single
        else:
            parse_row_func = encoder.encode_single

    # encoding
    if args.model_type in ["lucaone_gplm"]:
        # lucagplm独特的batch转换器
        batch_data_func = BatchConverter(
            tokenizer,
            no_position_embeddings=model_config.no_position_embeddings,
            no_token_type_embeddings=model_config.no_token_type_embeddings,
            truncation_seq_length=model_config.max_position_embeddings,
            ignore_index=model_config.ignore_index
        )
    else:
        dcForLanguageModeling, dcForWholeWordMask, \
        dcForTokenClassification, dcForSequenceClassification, \
        dcForStructureRegression, dcForPairClassification, dcForSeq2Seq = create_collator(args, tokenizer=tokenizer)
        batch_data_func = DataCollatorForAll(
            dcForLanguageModeling, dcForWholeWordMask,
            dcForTokenClassification, dcForSequenceClassification,
            dcForStructureRegression, dcForPairClassification,
            dcForSeq2Seq
        )

    # write logs
    if args.local_rank in [0, -1]:
        print("n_gpu: %d" % args.n_gpu)
        args_dict = {}
        for attr, value in sorted(args.__dict__.items()):
            if attr != "device":
                args_dict[attr] = value
        log_fp.write(json.dumps(args_dict, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        log_fp.write("n_gpu: %d\n" % args.n_gpu)
        log_fp.write("#" * 50 + "\n")
        if "token_level" in args.label_size and "gene_mask" in args.label_size["token_level"]:
            log_fp.write("gene/prot token level/gene_mask label num: %d\n" % args.label_size["token_level"]["gene_mask"])
        if "token_level" in args.label_size and "prot_mask" in args.label_size["token_level"]:
            log_fp.write("gene/prot token level/prot_mask label num: %d\n" % args.label_size["token_level"]["prot_mask"])
        if "span_level" in args.label_size and "gene_type" in args.label_size["span_level"]:
            log_fp.write("gene span level/gene_type label num: %d\n" % args.label_size["span_level"]["gene_type"])
        if "span_level" in args.label_size and "prot_homo" in args.label_size["span_level"]:
            log_fp.write("prot span level/prot_homo label num: %d\n" % args.label_size["span_level"]["prot_homo"])
        if "span_level" in args.label_size and "prot_site" in args.label_size["span_level"]:
            log_fp.write("prot span level/prot_site label num: %d\n" % args.label_size["span_level"]["prot_site"])
        if "span_level" in args.label_size and "prot_domain" in args.label_size["span_level"]:
            log_fp.write("prot span level/prot_domain label num: %d\n" % args.label_size["span_level"]["prot_domain"])
        if "seq_level" in args.label_size and "gene_taxonomy" in args.label_size["seq_level"]:
            log_fp.write("gene seq level/gene_taxonomy label num: %d\n" % args.label_size["seq_level"]["gene_taxonomy"])
        if "seq_level" in args.label_size and "prot_taxonomy" in args.label_size["seq_level"]:
            log_fp.write("prot seq level/prot_taxonomy label num: %d\n" % args.label_size["seq_level"]["prot_taxonomy"])
        if "seq_level" in args.label_size and "prot_keyword" in args.label_size["seq_level"]:
            log_fp.write("prot seq level/prot_keyword label num: %d\n" % args.label_size["seq_level"]["prot_keyword"])
        if "structure_level" in args.label_size and "prot_structure" in args.label_size["structure_level"]:
            log_fp.write("prot structure level/prot_structure num labels: %d\n" % args.label_size["structure_level"]["prot_structure"])
        if "structure_level" in args.label_size and "prot_secondary" in args.label_size["structure_level"]:
            log_fp.write("prot structure level/prot_secondary num labels: %d\n" % args.label_size["structure_level"]["prot_secondary"])
        if "structure_level" in args.label_size and "prot_secondary" in args.label_size["structure_level"]:
            log_fp.write("prot structure level/prot_secondary num labels: %d\n" % args.label_size["structure_level"]["prot_secondary"])
        if "pair_level" in args.label_size and "trans" in args.label_size["pair_level"]:
            log_fp.write("pair level num labels: %d\n" % args.label_size["pair_level"]["trans"])
        log_fp.write("#" * 50 + "\n")

        log_fp.write("Encoder Config:\n %s\n" % str(encoder_config))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Model Config:\n %s\n" % str(model_config))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Mode Architecture:\n %s\n" % str(model))
        log_fp.write("#" * 50 + "\n")
        log_fp.write("Model parameters: %d \n" % sum(p.numel() for p in model.parameters()))
        log_fp.write("#" * 50 + "\n")

        # model size
        model_size_info = get_parameter_number(model)
        log_fp.write(json.dumps(model_size_info, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        log_fp.flush()

    # set seed
    set_seed(args)

    # model to device
    model.to(args.device)

    if args.local_rank != -1:
        dist.barrier(device_ids=[args.local_rank])

    print("device:", args.device)

    if args.local_rank in [0, -1]:
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
        all_checkpoints = []
        for dirname in os.listdir(args.all_checkpoints_dirpath):
            if "init_parameters" in dirname:
                all_checkpoints.append(["init_parameters", 0])
            elif"checkpoint-step" in dirname:
                global_step = int(dirname.replace("checkpoint-step", ""))
                all_checkpoints.append([dirname, global_step])
            else:
                print("skip dirname=%s" % dirname)
                continue
        all_checkpoints = sorted(all_checkpoints, key=lambda x: x[1])
        print("raw checkpoints: %d" % len(all_checkpoints))
        if args.eval_min_step and args.eval_min_step > 0:
            all_checkpoints = [v for v in all_checkpoints if v[1] >= args.eval_min_step]
        if args.eval_max_step and args.eval_max_step > 0:
            all_checkpoints = [v for v in all_checkpoints if v[1] <= args.eval_max_step]
        if args.eval_interval_step > 0:
            all_checkpoints = [v for v in all_checkpoints if v[1] % args.eval_interval_step == 0]

        print("filtered checkpoints: %d" % len(all_checkpoints))
        print(all_checkpoints)
        print("*" * 50)
        # 切分
        if args.all_checkpoints_part is None:
            cur_checkpoints = all_checkpoints
            args.all_checkpoints_part = "1/1"
        else:
            strs = args.all_checkpoints_part.split("/")
            cur_part_idx, total_part = int(strs[0]), int(strs[1])
            assert 1 <= cur_part_idx <= total_part
            per_part = (len(all_checkpoints) + total_part - 1)//total_part
            cur_checkpoints = all_checkpoints[(cur_part_idx - 1) * per_part: min(cur_part_idx * per_part, len(all_checkpoints))]
        print("Eval: %s, size: %d" % (args.all_checkpoints_part, len(cur_checkpoints)))
        for checkpoint in cur_checkpoints:
            if checkpoint[0] == "init_parameters":
                args.model_dirpath = None
            else:
                args.model_dirpath = os.path.join(args.all_checkpoints_dirpath, checkpoint[0])
            global_step = checkpoint[1]
            print("Eval Checkpoints: %d" % global_step)
            model_config, model, tokenizer = get_model(args)
            model.to(args.device)
            do_eval_all_checkpoints(
                args,
                model,
                parse_row_func,
                batch_data_func,
                global_step,
                tb_writer=tb_writer,
                log_fp=log_fp
            )


if __name__ == "__main__":
    main()
    '''
    python run.py --do_train --add_special_tokens --pretrain_task_level_type seq_level --span_level_num_labels 11 --seq_level_num_labels 2
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    python -m torch.distributed.launch --nproc_per_node=8 --use_env run.py --do_train --add_special_tokens --pretrain_task_level_type all --span_level_num_labels 11 --seq_level_num_labels 2
    '''
