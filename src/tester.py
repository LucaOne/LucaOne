#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/5/5 09:55
@project: LucaOne
@file: tester
@desc: tester for LucaOne
'''
import os
import sys, torch
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from utils import to_device, concat_output, calc_avg_loss, calc_eval_test_loss, \
        process_outputs, eval_metrics, print_shape, metrics_merge, print_batch, writer_info_tb
    from multi_files_stream_dataloader import MultiFilesStreamLoader
    from common.multi_label_metrics import metrics_multi_label
    from common.metrics import metrics_multi_class, metrics_binary
except ImportError:
    from src.utils import to_device, concat_output, calc_avg_loss, calc_eval_test_loss,\
        process_outputs, eval_metrics, print_shape, metrics_merge, print_batch, writer_info_tb
    from src.multi_files_stream_dataloader import MultiFilesStreamLoader
    from src.common.multi_label_metrics import metrics_multi_label
    from src.common.metrics import metrics_multi_class, metrics_binary


def test(args, model, label_list, parse_row_func, batch_data_func, global_step, prefix="", tb_writer=None, log_fp=None):
    """
    evaluation on test set
    :param args:
    :param model:
    :param label_list:
    :param parse_row_func:
    :param batch_data_func:
    :param global_step
    :param prefix:
    :param tb_writer
    :param log_fp:
    :return:
    """
    if hasattr(model, "module"):
        model = model.module
    save_output_dir = os.path.join(args.output_dir, prefix)
    print("\nTesting information dir: ", save_output_dir)
    if not os.path.exists(save_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(save_output_dir)
    test_dataloader = MultiFilesStreamLoader(
        args.test_data_dir,
        args.per_gpu_eval_batch_size,
        args.buffer_size,
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
    # Testing
    if log_fp:
        log_fp.write("***** Running testing {} *****\n".format(prefix))
        log_fp.write("Test Dataset Instantaneous batch size per GPU = %d\n" % args.per_gpu_eval_batch_size)
        log_fp.write("#" * 50 + "\n")
        log_fp.flush()

    dataset_name = "test"

    nb_steps = 0
    # loss
    total_losses = {}
    # step
    total_steps = {}

    # truth
    truths = {}
    # predicted prob
    preds = {}
    # truth
    truths_b = {}
    # predicted prob
    preds_b = {}
    # truth
    pair_truths = {}
    # predicted prob
    pair_preds = {}

    total_loss = 0

    model.eval()

    done_sample_num = 0

    for step, batch in enumerate(test_dataloader):
        if "sample_ids" in batch:
            del batch["sample_ids"]
        # testing
        with torch.no_grad():
            batch, cur_sample_num = to_device(args.device, batch)
            done_sample_num += cur_sample_num
            try:
                output = model(
                    **batch,
                    output_keys=args.gene_output_keys,
                    output_keys_b=args.prot_output_keys,
                    pair_output_keys=args.pair_output_keys,
                    output_attentions=True,
                    output_hidden_states=True
                )
            except Exception as e:
                exception_path = "../exception/%s" % args.time_str
                if not os.path.exists(exception_path):
                    os.makedirs(exception_path)
                with open(os.path.join(exception_path, "test_exception_info_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(e) + "\n")
                    afp.flush()
                with open(os.path.join(exception_path, "test_exception_input_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(batch) + "\n")
                    afp.flush()
                debug_path = "../debug/test/local_rank%s/%d/" % ("_" + str(args.local_rank) if args.local_rank >= 0 else "", step)
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                with open(os.path.join(debug_path, "test_exception_input_details.txt"), "a+") as afp:
                    print_batch(batch, key=None, debug_path=debug_path, wfp=afp, local_rank=args.local_rank)
                    afp.flush()
                continue

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

            print("\rTest, Batch: %06d, Sample Num: %d, Cur Loss: %0.6f, Avg Loss: %0.6f" %
                  (step + 1, done_sample_num, cur_loss, total_loss/(nb_steps + 1)),
                  end="", flush=True)
            nb_steps += 1
            if args.do_metrics:
                outputs_idx = 0
                if "labels" in batch:
                    truths, preds = process_outputs(
                        args.output_mode, batch["labels"],
                        outputs[outputs_idx], truths, preds,
                        ignore_index=args.ignore_index,
                        keep_seq=False
                    )
                    outputs_idx += 1
                if "labels_b" in batch:
                    truths_b, preds_b = process_outputs(
                        args.output_mode, batch["labels_b"],
                        outputs[outputs_idx], truths_b, preds_b,
                        ignore_index=args.ignore_index,
                        keep_seq=False
                    )
                    outputs_idx += 1
                if "pair_label" in batch:
                    pair_truths, pair_preds = process_outputs(
                        args.output_mode, batch["pair_label"],
                        outputs[outputs_idx], pair_truths, pair_preds,
                        ignore_index=args.ignore_index,
                        keep_seq=False
                    )

    all_result, merged_loss, loss_detail = calc_avg_loss(
        total_losses,
        nb_steps,
        total_steps=total_steps
    )
    if args.do_metrics:
        if truths is not None and len(truths) > 0:
            results = eval_metrics(args.output_mode, truths, preds, threshold=0.5)
            all_result = metrics_merge(results, all_result)
        if truths_b is not None and len(truths_b) > 0:
            results_b = eval_metrics(args.output_mode, truths_b, preds_b, threshold=0.5)
            all_result = metrics_merge(results_b, all_result)
        if pair_truths is not None and len(pair_truths) > 0:
            pair_results = eval_metrics(args.output_mode, pair_truths, pair_preds, threshold=0.5)
            all_result = metrics_merge(pair_results, all_result)

    writer_info_tb(
        tb_writer, {
            "done_sample_num": done_sample_num
        },
        global_step,
        prefix=dataset_name
    )
    writer_info_tb(
        tb_writer, {
            "merged_loss": merged_loss
        },
        global_step,
        prefix=dataset_name
    )
    writer_info_tb(
        tb_writer,
        loss_detail,
        global_step,
        prefix=dataset_name + "_avg_loss"
    )
    writer_info_tb(
        tb_writer,
        total_steps,
        global_step,
        prefix=dataset_name + "_total_steps"
    )

    with open(os.path.join(save_output_dir, "eval_%s_checkpoints-step%d_metrics.txt" % (dataset_name, global_step)), "w") as writer:
        writer.write("***** Eval results %s on Checkpoints %d *****\n" % (dataset_name, global_step))
        writer.write("%s average merged_loss = %0.6f\n" % (dataset_name, merged_loss))
        writer.write("%s detail loss = %s\n" % (dataset_name, str(loss_detail)))
        for key in sorted(all_result.keys()):
            writer.write("%s = %s\n" % (key, str(all_result[key])))

    return all_result
