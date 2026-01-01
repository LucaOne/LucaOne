#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/26 15:35
@project: LucaOne
@file: do_eval_mask
@desc: xxxx
'''
import os
import sys, torch
import numpy as np
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, concat_output, calc_avg_loss, calc_eval_test_loss, \
        print_batch_device, writer_info_tb, \
        process_outputs, eval_metrics, print_shape, metrics_merge, print_batch, topk_values_indices
    from multi_files_stream_dataloader import MultiFilesStreamLoader
    from common.multi_label_metrics import metrics_multi_label
    from common.metrics import metrics_multi_class, metrics_binary
    from src.file_operator import csv_writer
except ImportError:
    from src.utils import to_device, concat_output, calc_avg_loss, calc_eval_test_loss,\
        print_batch_device, writer_info_tb, \
        process_outputs, eval_metrics, print_shape, metrics_merge, print_batch, topk_values_indices
    from src.multi_files_stream_dataloader import MultiFilesStreamLoader
    from src.common.multi_label_metrics import metrics_multi_label
    from src.common.metrics import metrics_multi_class, metrics_binary
    from src.file_operator import csv_writer
from torch.utils.tensorboard import SummaryWriter


def do_eval_mask(
        args,
        model,
        parse_row_func,
        batch_data_func,
        global_step,
        tb_writer=None,
        log_fp=None
):
    if hasattr(model, "module"):
        model = model.module
    save_output_dir = os.path.join(args.output_dir, "checkpoints-step%d" % global_step)
    print("\nEvaluating information dir: ", save_output_dir)
    if args.local_rank in [-1, 0] and not os.path.exists(save_output_dir):
        os.makedirs(save_output_dir)
    # logger
    if args.local_rank in [0, -1]:
        if tb_writer is None:
            tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
        if log_fp is None:
            log_fp = open(os.path.join(args.log_dir, "logs.txt"), "a+")
    test_dataloader = MultiFilesStreamLoader(
        args.test_data_dir,
        batch_size=1,
        buffer_size=args.buffer_size,
        parse_row_func=parse_row_func,
        batch_data_func=batch_data_func,
        pretrain_task_level_type=args.pretrain_task_level_type,
        gene_label_size_dict=args.gene_label_size_dict,
        gene_output_mode_dict=args.gene_output_mode_dict,
        prot_label_size_dict=args.prot_label_size_dict,
        prot_output_mode_dict=args.prot_output_mode_dict,
        pair_label_size_dict=args.pair_label_size_dict,
        pair_output_mode_dict=args.pair_output_mode_dict,
        dataset_type="test",
        header=True,
        shuffle=False
    )
    # evaluate
    dataset_name, _ = os.path.splitext(os.path.basename(args.test_data_dir))
    if log_fp:
        log_fp.write("***** Running Evaluation %s on Checkpoints-%d*****\n" % (dataset_name, global_step))
        log_fp.write("%s Dataset Instantaneous Batch Size per GPU = %d\n" % (dataset_name, args.per_gpu_eval_batch_size))
        log_fp.write("*" * 50 + "\n")
        log_fp.flush()

    nb_steps = 0
    # total losses
    total_losses = {}
    # total steps
    total_steps = {}

    # truth
    truths = {}
    # predicted prob
    preds = {}

    if args.do_metrics:
        # preds_details
        preds_details = []
    # truth
    truths_b = {}
    # predicted prob
    preds_b = {}
    # truth
    pair_truths = {}
    # predicted prob
    pair_preds = {}

    total_loss = 0
    done_sample_num = 0
    model.eval()
    for step, batch_input in enumerate(test_dataloader):
        sample_ids = None
        if "sample_ids" in batch_input:
            sample_ids = batch_input["sample_ids"]
            del batch_input["sample_ids"]
        # evaluate
        with torch.no_grad():
            batch_input, cur_sample_num = to_device(args.device, batch_input)
            """
            print_batch_device(batch_input, level=1)
            print("*" * 50)
            input("continue:")
            print_batch(batch_input)
            print("*" * 50)
            input("continue:")
            """
            done_sample_num += cur_sample_num
            output = model(
                **batch_input,
                output_keys=args.gene_output_keys,
                output_keys_b=args.prot_output_keys,
                pair_output_keys=args.pair_output_keys,
                output_attentions=False,
                output_hidden_states=False
            )
            if isinstance(output, dict):
                losses = []
                outputs = []
                if output.losses:
                    losses.append(output.losses)
                if output.losses_b:
                    losses.append(output.losses_b)
                if output.pair_losses:
                    losses.append(output.pair_losses)
                if output.outputs:
                    outputs.append(output.outputs)
                if output.outputs_b:
                    outputs.append(output.outputs_b)
                if output.pair_outputs:
                    outputs.append(output.pair_outputs)
            else:
                losses, outputs = output[:2]
            current_losses, total_losses, total_steps, total_loss, cur_loss = calc_eval_test_loss(
                losses,
                total_losses,
                total_steps,
                total_loss
            )

            if (step + 1) % 10000 == 0:
                print("\rEval %s, Batch: %06d, Sample Num: %d, Cur Loss: %0.6f, Avg Loss: %0.6f" % (
                    dataset_name,
                    step + 1,
                    done_sample_num,
                    cur_loss,
                    total_loss/(nb_steps + 1)), end="", flush=True)
                if log_fp is not None:
                    all_result, merged_loss, loss_detail = calc_avg_loss(
                        total_losses,
                        nb_steps + 1,
                        total_steps=total_steps
                    )
                    log_fp.write("Eval %s, Checkpoints: %d, Batch: %06d, Sample Num: %d, Cur Loss: %0.6f, Avg Loss: %0.6f \n" % (
                        dataset_name,
                        global_step,
                        step + 1,
                        done_sample_num,
                        cur_loss,
                        total_loss/(nb_steps + 1)))
                    log_fp.write("current_losses:" + str(current_losses) + "\n")
                    log_fp.write("total_losses:" + str(total_losses) + "\n")
                    log_fp.write("total_loss:" + str(total_loss) + "\n")
                    log_fp.write("cur_loss:" + str(cur_loss) + "\n")
                    log_fp.write("total_steps:" + str(total_steps) + "\n")
                    log_fp.write("all_result:" + str(all_result) + "\n")
                    log_fp.write("merged_loss:" + str(merged_loss) + "\n")
                    log_fp.write("loss_detail:" + str(loss_detail) + "\n")
                    log_fp.flush()
            nb_steps += 1
            # only for batch size = 1, batch size > 1 todo
            if args.do_metrics:
                mask_type = None
                outputs_idx = 0
                if "labels" in batch_input:
                    mask_type = "gene"
                    truths, preds, masked_truth, masked_pred = process_outputs(
                        args.output_mode,
                        batch_input["labels"],
                        outputs[outputs_idx],
                        truths,
                        preds,
                        ignore_index=args.ignore_index,
                        keep_seq=False,
                        return_masked_truth_pred=True
                    )
                    outputs_idx += 1
                if "labels_b" in batch_input:
                    mask_type = "prot"
                    truths_b, preds_b, masked_truth, masked_pred = process_outputs(
                        args.output_mode,
                        batch_input["labels_b"],
                        outputs[outputs_idx],
                        truths_b,
                        preds_b,
                        ignore_index=args.ignore_index,
                        keep_seq=False,
                        return_masked_truth_pred=True
                    )
                    outputs_idx += 1
                if "pair_label" in batch_input:
                    pair_truths, pair_preds, masked_truth, masked_pred = process_outputs(
                        args.output_mode,
                        batch_input["pair_label"],
                        outputs[outputs_idx],
                        pair_truths,
                        pair_preds,
                        ignore_index=args.ignore_index,
                        keep_seq=False,
                        return_masked_truth_pred=True
                    )
                if mask_type is not None:
                    # batch size only 1, > 1 to do
                    # masked prob
                    masked_prob = masked_pred["token_level"]["gene_mask"] if mask_type == "gene" else masked_pred["token_level"]["prot_mask"]
                    # masked truth
                    masked_truth = masked_truth["token_level"]["gene_mask"] if mask_type == "gene" else masked_truth["token_level"]["prot_mask"]
                    if masked_prob is not None or masked_truth is not None:
                        cur_record = []
                        # sample_id
                        cur_record.append(sample_ids[0])
                        cur_prob = np.max(masked_prob, axis=-1)
                        cur_record.append(cur_prob.tolist())
                        # pred
                        cur_argmax_pred = np.argmax(masked_prob, axis=-1)
                        cur_argmax_pred = cur_argmax_pred.tolist()
                        cur_record.append(cur_argmax_pred)

                        # topk
                        topk_values, topk_indices = topk_values_indices(masked_prob, topk=4)
                        cur_record.append(topk_values.tolist())
                        cur_record.append(topk_indices.tolist())
                        # truth
                        masked_truth = masked_truth.tolist()
                        cur_record.append(masked_truth)

                        acc = 0
                        for idx, t_ch in enumerate(masked_truth):
                            p_ch = cur_argmax_pred[idx]
                            if t_ch == p_ch:
                                acc += 1
                        cur_record.append(acc/len(masked_truth))
                        preds_details.append(cur_record)
    if log_fp is not None:
        line = "#" * 50
        log_fp.write(line + "\n")
    all_result, merged_loss, loss_detail = calc_avg_loss(
        total_losses,
        nb_steps,
        total_steps=total_steps
    )
    writer_info_tb(tb_writer, {
        "done_sample_num": done_sample_num
    }, global_step, prefix=dataset_name)
    writer_info_tb(tb_writer, {
        "merged_loss": merged_loss
    }, global_step, prefix=dataset_name)
    writer_info_tb(
        tb_writer,
        loss_detail,
        global_step,
        prefix=dataset_name + '_' + "avg_loss"
    )
    writer_info_tb(
        tb_writer,
        total_steps,
        global_step,
        prefix=dataset_name + '_' + "total_steps"
    )
    if args.do_metrics:
        if truths is not None and len(truths) > 0:
            results = eval_metrics(
                args.output_mode,
                truths, preds,
                threshold=0.5
            )
            all_result = metrics_merge(results, all_result)
        if truths_b is not None and len(truths_b) > 0:
            results_b = eval_metrics(
                args.output_mode,
                truths_b,
                preds_b,
                threshold=0.5
            )
            all_result = metrics_merge(results_b, all_result)
        if pair_truths is not None and len(pair_truths) > 0:
            pair_results = eval_metrics(
                args.output_mode,
                pair_truths,
                pair_preds,
                threshold=0.5
            )
            all_result = metrics_merge(pair_results, all_result)
    with open(os.path.join(save_output_dir, "eval_%s_checkpoints-step%d_metrics.txt" % (dataset_name, global_step)), "w") as wfp:
        wfp.write("***** Eval results %s on Checkpoints %d *****\n" % (dataset_name, global_step))
        wfp.write("%s average merged_loss = %0.6f\n" % (dataset_name, merged_loss))
        wfp.write("%s detail loss = %s\n" % (dataset_name, str(loss_detail)))
        for key in sorted(all_result.keys()):
            wfp.write("%s = %s\n" % (key, str(all_result[key])))
    if args.do_metrics and preds_details:
        print("preds_details: %d" % len(preds_details))
        predicted_results_savepath = os.path.join(save_output_dir, "eval_%s_checkpoints-step%d_pred_results.csv" % (dataset_name, global_step))
        csv_writer(preds_details, predicted_results_savepath, header=[
            "sample_id", "prob", "pred" ,
            "topk_prob", "topk_pred", "truth",
            "accuracy"
        ])
