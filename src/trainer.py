#!/usr/bin/env python
# encoding: utf-8
'''
@license: (C) Copyright 2021, Hey.
@author: Hey
@email: sanyuan.hy@alibaba-inc.com
@tel: 137****6540
@datetime: 2023/4/26 15:35
@project: LucaOne
@file: trainer
@desc: trainer of LucaOne
'''
import shutil
import os, sys, time, json
import torch.distributed as dist
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup
sys.path.append(".")
sys.path.append("..")
sys.path.append("../src")
try:
    from common.multi_task_loss import *
    from utils import to_device, get_lr, calc_loss, calc_loss_index, writer_info_tb, \
        print_batch_input, print_batch_output, print_batch, lcm, write_processed_sample_ids
    from evaluator import evaluate
    from tester import test
except ImportError:
    from src.common.multi_task_loss import *
    from src.utils import to_device, get_lr, calc_loss, calc_loss_index, writer_info_tb, \
        print_batch_input, print_batch_output, print_batch, lcm, write_processed_sample_ids
    from src.evaluator import evaluate
    from src.tester import test


def reduce_tensor(tensor, world_size):
    # 用于平均所有gpu上的运行结果，比如loss
    # Reduces the tensor data across all machines
    # Example: If we print the tensor, we can get:
    # tensor(334.4330, device='cuda:1') *********************, here is cuda:  cuda:1
    # tensor(359.1895, device='cuda:3') *********************, here is cuda:  cuda:3
    # tensor(263.3543, device='cuda:2') *********************, here is cuda:  cuda:2
    # tensor(340.1970, device='cuda:0') *********************, here is cuda:  cuda:0
    rt = tensor.clone()
    dist.all_reduce(rt, op=torch.distributed.ReduceOp.SUM)
    rt /= world_size
    return rt


def train(args, model, model_config, dataloader, label_size_dict, parse_row_func, batch_data_func, tokenizer, train_sampler=None, log_fp=None):
    # logger
    if args.local_rank in [0, -1]:
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
        if log_fp is None:
            log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    no_decay = ["bias", "layer_norm.weight"]
    no_decay_keys = [n for n, _ in model.named_parameters() if any(nd in n.lower() for nd in no_decay)]
    '''
    print("no_decay_keys: ")
    print(no_decay_keys)
    '''
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    if args.multi_loss_strategy == "auto_weight":
        args.awl = AutomaticWeightedLoss(loss_mum=args.task_num)
        optimizer_grouped_parameters.append({
            "params": args.awl.parameters(),
            "weight_decay": 0.0
        })

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=[args.beta1 if args.beta1 > 0 else 0.9, args.beta2 if args.beta2 > 0 else 0.98],
                      eps=args.adam_epsilon)
    print("Init lr: ", get_lr(optimizer))
    print("Peak lr: ", args.learning_rate)
    print("Scheduler_type: %s" % args.scheduler_type)
    args.warmup_steps = int(args.warmup_steps / args.gradient_accumulation_steps)
    if args.warmup_steps < 1000:
        args.warmup_steps = 2000
    if args.scheduler_type == "step" and args.max_steps >= 100000:
        # https://blog.csdn.net/orangerfun/article/details/120400247
        '''
        optimizer： 优化器
        num_warmup_steps：初始预热步数
        num_training_steps：整个训练过程的总步数
        '''
        print("Use Warmup, warmup_steps=%d, max_steps=%d" % (args.warmup_steps, args.max_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    else:
        print("Use ExponentialLR")
        args.scheduler_type = "epoch"
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=args.decay_rate if args.decay_rate > 0 else 0.9)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # find_unused_parameters=True
        if "all" in args.pretrain_task_level_type:
            find_unused_parameters = False
        else:
            find_unused_parameters = True
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=find_unused_parameters)
    optimizer.zero_grad()
    if args.local_rank in [0, -1]:
        global_step = 0
        best_metric_type = args.best_metric_type
        best_metric_flag = True
        if "loss" in best_metric_type:
            # argmin
            best_metric_value = 10000000.0
            best_metric_flag = False
        else:
            # argmax
            best_metric_value = 0.0
        best_metric_model_info = {}
        run_begin_time = time.time()
        total_loss, logging_loss = 0.0, 0.0
        real_epoch = 0
        total_use_time = 0
        done_sample_num = 0
    total_loss_detail = {}
    last_last_loss_list = None
    last_loss_list = None
    cur_global_steps = 0
    for epoch in range(args.num_train_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if args.local_rank in [0, -1]:
            print("\n=====Epoch: %06d=====" % (epoch + 1))
        batch_total = 0

        cur_epoch_step = 0
        cur_epoch_loss = 0.0
        cur_epoch_time = 0.0
        # total_step = 0
        # num = 0
        no_grad_gradient_accumulation_step = False
        trained_sample_ids = []
        # 训练模式
        model.train()
        for step, batch in enumerate(dataloader):
            # 用于训练中途失败来恢复现场
            sample_ids = batch["sample_ids"]
            trained_sample_ids.extend(sample_ids)
            if len(trained_sample_ids) >= args.processed_sample_cnt and (cur_global_steps + 1) % args.save_steps == 0:
                write_processed_sample_ids(dataset_type="train",
                                           sample_ids=trained_sample_ids,
                                           time_str=args.time_str,
                                           epoch=epoch + 1,
                                           local_rank=args.local_rank)
                trained_sample_ids = []
            del batch["sample_ids"]

            batch_total += 1
            if args.local_rank in [-1, 0]:
                begin_time = time.time()
            batch, cur_sample_num = to_device(args.device, batch)
            if args.local_rank in [-1, 0]:
                done_sample_num += cur_sample_num
            # print_batch(batch)
            # print("----" * 10)
            # print_batch_output(batch)
            try:
                output = model(
                    **batch,
                    output_keys=args.gene_output_keys,
                    output_keys_b=args.prot_output_keys,
                    pair_output_keys=args.pair_output_keys,
                    output_attentions=True,
                    output_hidden_states=True
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
                # print_batch_output(outputs)
                # print(losses)
                # print("#####" * 10)
                loss = calc_loss(args, losses, last_last_loss_list=last_last_loss_list, last_loss_list=last_loss_list)
                '''
                if args.n_gpu > 1:
                    reduced_loss = reduce_tensor(loss.data, dist.get_world_size())
                else:
                    reduced_loss = loss
                '''
                if args.local_rank in [0, -1]:
                    # cur_loss = reduced_loss.item()
                    cur_loss = loss.item()
                    end_time = time.time()
                    cur_use_time = end_time - begin_time
                    total_use_time += cur_use_time
                    total_loss += cur_loss
                    logging_loss += cur_loss
                    cur_epoch_loss += cur_loss
                    cur_epoch_time += cur_use_time
                    global_step += 1
                    cur_epoch_step += 1

                    # print(str(losses))
                    # print(str(loss))
                    if global_step % args.gradient_accumulation_steps == 0:
                        print("\rTraining, Epoch: %04d, Batch: %06d, Sample Num: %d, Cur Loss: %.08f, Avg Loss: %.08f" % (
                            epoch + 1,
                            cur_epoch_step,
                            done_sample_num,
                            cur_loss,
                            total_loss/global_step), end="", flush=True
                              )
                        if global_step == 1 or global_step % args.loss_logging_steps == 0:
                            writer_info_tb(tb_writer, {
                                "loss": cur_loss
                            }, global_step, prefix="training")
                        if global_step % args.logging_steps == 0:
                            log_fp.write("Training, Epoch: %04d, Batch: %06d, Sample Num: %d, Cur Loss: %.08f, Log Avg loss: %.08f, Global Avg Loss: %.08f, Time: %0.4f\n"
                                         % (
                                             epoch + 1,
                                             cur_epoch_step,
                                             done_sample_num,
                                             cur_loss,
                                             logging_loss / lcm(args.logging_steps, args.gradient_accumulation_steps),
                                             total_loss / global_step,
                                             cur_use_time)
                                         )
                            log_fp.write(str(losses) + "\n")
                            log_fp.flush()
                            writer_info_tb(tb_writer,
                                           {
                                               "epoch": epoch + 1,
                                               "cur_epoch_step": cur_epoch_step,
                                               "cur_epoch_done_sample_num": done_sample_num,
                                               "cur_epoch_avg_loss": cur_epoch_loss / cur_epoch_step,
                                               "cur_batch_loss": cur_loss,
                                               "global_avg_loss": total_loss / global_step,
                                               "cur_use_time": cur_use_time,
                                               "global_step": global_step,
                                               "log_avg_loss": logging_loss / lcm(args.logging_steps, args.gradient_accumulation_steps),
                                           }, global_step, prefix="logging")
                            logging_loss = 0.0

                '''
                for k, v in model.named_parameters():
                    print(k)
                    print(v.grad)
                '''
                if args.gradient_accumulation_steps > 1:
                    # The loss of each batch will be divided by gradient_accumulation_steps
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()
                no_grad_gradient_accumulation_step = False
                cur_global_steps += 1
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    if args.fp16:
                        torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                    else:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                    if scheduler is not None:
                        scheduler_update_flag = False
                        if args.scheduler_type == "epoch":
                            if epoch == 0 and args.lr_update_steps > 0 and cur_global_steps % args.lr_update_steps == 0:
                                # 第一次epoch内部根据steps调整
                                # Update learning rate schedule
                                scheduler.step()
                                scheduler_update_flag = True
                        else:
                            scheduler.step()
                            scheduler_update_flag = True
                        if args.local_rank in [0, -1] and scheduler_update_flag and global_step % args.logging_steps == 0:
                            if args.scheduler_type == "epoch":
                                updated_lr = scheduler.get_last_lr()[0]
                            else:
                                updated_lr = get_lr(optimizer)
                            print("\ncur steps: %d,  lr: %f" % (cur_global_steps, updated_lr))
                            log_fp.write("Steps: %d, Updated lr: %f\n" % (cur_global_steps, updated_lr))
                            log_fp.flush()
                            writer_info_tb(tb_writer, {"updated_lr": updated_lr}, cur_global_steps, prefix="logging")

                    optimizer.zero_grad()
                    no_grad_gradient_accumulation_step = True
                    # print("lr: ", get_lr(optimizer))
                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                    output_dir = os.path.join(args.output_dir, "checkpoint-step{}".format(global_step))
                    save_check_point(args, model, model_config, tokenizer, output_dir)
            except Exception as e:
                exception_path = "../exception/%s" % args.time_str
                if not os.path.exists(exception_path):
                    os.makedirs(exception_path)
                with open(os.path.join(exception_path, "train_exception_info_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(e) + "\n")
                    afp.flush()
                with open(os.path.join("train_exception_input_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(batch) + "\n")
                    afp.flush()
                debug_path = "../debug/train/local_rank%s/%s/" % ("_" + str(args.local_rank) if args.local_rank >= 0 else "", str(epoch) + "_" + str(step))
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                with open(os.path.join(debug_path, "train_exception_input_details.txt"), "a+") as afp:
                    print_batch(batch, key=None, debug_path=debug_path, wfp=afp, local_rank=args.local_rank)
                    afp.flush()
                raise Exception(e)
        # 一个epoch完成
        if not no_grad_gradient_accumulation_step:
            optimizer.step()
            optimizer.zero_grad()
            print("Has retained gard: rank=%d" % args.local_rank)

        if len(trained_sample_ids) > 0:
            write_processed_sample_ids(dataset_type="train",
                                       sample_ids=trained_sample_ids,
                                       time_str=args.time_str,
                                       epoch=epoch + 1,
                                       local_rank=args.local_rank
                                       )
        # epoch = 1的时候不调整（也就是第二次不调整，后面开始每一个epoch调整一次）
        if epoch > 1 and scheduler is not None and args.scheduler_type == "epoch":
            scheduler.step()
            if args.local_rank in [-1, 0]:
                updated_lr = scheduler.get_last_lr()[0]
                writer_info_tb(tb_writer, {"updated_lr": updated_lr}, cur_global_steps, prefix="logging")
        '''
        if args.n_gpu > 1:
            dist.barrier()
        '''
        if args.local_rank in [-1, 0]:
            logs = {}
            update_flag = False
            # Only evaluate at local_rank=0 or single GPU
            if args.local_rank in [-1, 0] and args.evaluate_during_training and args.dev_data_dir \
                    and (args.start_epoch < 0 or epoch + 1 >= args.start_epoch):
                eval_result = evaluate(args, model, parse_row_func, batch_data_func,
                                       prefix="checkpoint-{}".format(global_step),
                                       log_fp=log_fp)
                print("Eval result:")
                print(eval_result)
                for key, value in eval_result.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value
                    if key == best_metric_type:
                        if best_metric_flag and best_metric_value < value or \
                                not best_metric_flag and best_metric_value > value:
                            best_metric_value = value
                            update_flag = True

                logs["update_flag"] = update_flag
                if update_flag and args.test_data_dir:
                    best_metric_model_info.update({"epoch": epoch + 1, "global_step": global_step})
                    test_result = test(args, model, label_size_dict, parse_row_func, batch_data_func,
                                       prefix="checkpoint-{}".format(global_step),
                                       log_fp=log_fp)
                    print("Test result:")
                    print(test_result)
                    for key, value in test_result.items():
                        eval_key = "test_{}".format(key)
                        logs[eval_key] = value
                    best_metric_model_info.update(logs)
            avg_batch_time = round(cur_epoch_time / cur_epoch_step, 2)
            log_fp.write("Epoch Time: %f, Avg time per batch (s): %f\n" % (cur_epoch_time, avg_batch_time))
            if scheduler is not None and args.scheduler_type == "epoch":
                logs["lr"] = scheduler.get_last_lr()[0]
            else:
                logs["lr"] = get_lr(optimizer)
            logs["batch_avg_loss"] = total_loss / global_step
            logs["cur_epoch_loss"] = cur_epoch_loss
            logs["cur_epoch_avg_loss"] = cur_epoch_loss / cur_epoch_step
            logs["epoch"] = epoch + 1
            # print(logs)
            writer_info_tb(tb_writer, logs, global_step, prefix=None)
            log_fp.write(json.dumps({**logs, **{"step": global_step}}, ensure_ascii=False) + "\n")
            log_fp.write("#" * 50 + "\n")
            log_fp.flush()
            print("End epoch: %d" % (epoch + 1))
            # save checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if args.save_all:
                save_check_point(args, model, model_config, tokenizer, output_dir)
            elif update_flag:
                if args.delete_old:
                    # delete the old CheckPoint
                    filename_list = os.listdir(args.output_dir)
                    for filename in filename_list:
                        if "checkpoint-" in filename and filename != "checkpoint-{}".format(global_step):
                            shutil.rmtree(os.path.join(args.output_dir, filename))
                save_check_point(args, model, model_config, tokenizer, output_dir)

        last_last_loss_list = last_loss_list
        last_loss_list = [v/batch_total for v in total_loss_detail]

        if args.local_rank in [0, -1]:
            if scheduler is not None and args.scheduler_type == "epoch":
                cur_lr = scheduler.get_last_lr()[0]
            else:
                cur_lr = get_lr(optimizer)
            print("Epoch: %d, batch total: %d, lr: %0.10f" % (epoch + 1, batch_total, cur_lr))
            real_epoch += 1
        '''
        if args.n_gpu > 1:
            dist.barrier()
        '''
        torch.cuda.empty_cache()

    if args.local_rank in [0, -1]:
        run_end_time = time.time()
        tb_writer.close()
        log_fp.write("#" * 25 + "Best Metric" + "#" * 25 + "\n")
        log_fp.write(json.dumps(best_metric_model_info, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        avg_time_per_epoch = round((run_end_time - run_begin_time)/real_epoch, 2)
        log_fp.write("Total Time: %f, Avg time per epoch(%d epochs): %f\n" % (run_end_time - run_begin_time, real_epoch, avg_time_per_epoch))
        log_fp.flush()

    if args.n_gpu > 1:
        cleanup()

    if args.local_rank in [0, -1]:
        return global_step, total_loss / global_step, best_metric_model_info

    return None, None, None


def train_continue(args, model, model_config, dataloader, label_size_dict, parse_row_func, batch_data_func, tokenizer, train_sampler=None, log_fp=None):
    # logger
    if args.local_rank in [0, -1]:
        tb_writer = SummaryWriter(log_dir=args.tb_log_dir)
        if log_fp is None:
            log_fp = open(os.path.join(args.log_dir, "logs.txt"), "w")
    no_decay = ["bias", "layer_norm.weight"]
    no_decay_keys = [n for n, _ in model.named_parameters() if any(nd in n.lower() for nd in no_decay)]
    '''
    print("no_decay_keys: ")
    print(no_decay_keys)
    '''
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n.lower() for nd in no_decay)],
            "weight_decay": args.weight_decay
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n.lower() for nd in no_decay)],
            "weight_decay": 0.0
        }
    ]
    if args.multi_loss_strategy == "auto_weight":
        args.awl = AutomaticWeightedLoss(loss_mum=args.task_num)
        optimizer_grouped_parameters.append({
            "params": args.awl.parameters(),
            "weight_decay": 0.0
        })

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=args.learning_rate,
                      betas=[args.beta1 if args.beta1 > 0 else 0.9, args.beta2 if args.beta2 > 0 else 0.98],
                      eps=args.adam_epsilon)
    print("Init lr: ", get_lr(optimizer))
    print("Peak: ", args.learning_rate)
    print("Scheduler_type: %s" % args.scheduler_type)
    args.warmup_steps = int(args.warmup_steps / args.gradient_accumulation_steps)
    if args.warmup_steps < 1000:
        args.warmup_steps = 2000
    if args.scheduler_type == "step" and args.max_steps >= 100000:
        # https://blog.csdn.net/orangerfun/article/details/120400247
        '''
        optimizer： 优化器
        num_warmup_steps：初始预热步数
        num_training_steps：整个训练过程的总步数
        '''
        print("Use Warmup, warmup_steps=%d, max_steps=%d" % (args.warmup_steps, args.max_steps))
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=args.max_steps)
    else:
        print("Use ExponentialLR")
        args.scheduler_type = "epoch"
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,
                                                           gamma=args.decay_rate if args.decay_rate > 0 else 0.9)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # Distributed training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        # find_unused_parameters=True
        if "all" in args.pretrain_task_level_type:
            find_unused_parameters = False
        else:
            find_unused_parameters = True
        model = torch.nn.parallel.DistributedDataParallel(model,
                                                          device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=find_unused_parameters)
    optimizer.zero_grad()
    if args.local_rank in [0, -1]:
        global_step = 0
        best_metric_type = args.best_metric_type
        best_metric_flag = True
        if "loss" in best_metric_type: # argmin
            best_metric_value = 10000000.0
            best_metric_flag = False
        else: # argmax
            best_metric_value = 0.0
        best_metric_model_info = {}
        run_begin_time = time.time()
        total_loss, logging_loss = args.global_loss, 0.0
        real_epoch = 0
        total_use_time = 0
        done_sample_num = 0
    total_loss_detail = {}
    last_last_loss_list = None
    last_loss_list = None
    cur_global_steps = 0
    for epoch in range(args.num_train_epochs):
        if train_sampler:
            train_sampler.set_epoch(epoch)
        if args.local_rank in [0, -1]:
            print("\n=====Epoch: %06d=====" % (epoch + 1))
        batch_total = 0

        cur_epoch_step = 0
        cur_epoch_loss = args.epoch_loss
        cur_epoch_time = 0.0
        # total_step = 0
        # num = 0
        no_grad_gradient_accumulation_step = False
        trained_sample_ids = []
        # 训练模式
        model.train()
        for step, batch in enumerate(dataloader):
            # 用于训练中途失败来恢复现场
            sample_ids = batch["sample_ids"]
            trained_sample_ids.extend(sample_ids)
            # 训练的样本个数超过指定大小，并且与mode save策略保持一致
            if len(trained_sample_ids) >= args.processed_sample_cnt and (cur_global_steps + 1) % args.save_steps == 0:
                write_processed_sample_ids(dataset_type="train",
                                           sample_ids=trained_sample_ids,
                                           time_str=args.time_str,
                                           epoch=epoch + 1,
                                           local_rank=args.local_rank
                                           )
                trained_sample_ids = []
            del batch["sample_ids"]

            batch_total += 1
            if args.local_rank in [-1, 0]:
                begin_time = time.time()
            batch, cur_sample_num = to_device(args.device, batch)
            if args.local_rank in [-1, 0]:
                done_sample_num += cur_sample_num
            # print_batch(batch)
            # print("----" * 10)
            # print_batch_output(batch)
            try:
                if args.trained_checkpoint is not None and cur_global_steps <= args.trained_checkpoint:
                    # 空跑
                    if args.local_rank in [0, -1]:
                        global_step += 1
                        cur_epoch_step += 1
                    cur_global_steps += 1
                    # 学习速率需要更新
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if scheduler is not None:
                            scheduler_update_flag = False
                            if args.scheduler_type == "epoch":
                                if epoch == 0 and args.lr_update_steps > 0 and cur_global_steps % args.lr_update_steps == 0:
                                    # 第一次epoch内部根据steps调整
                                    # Update learning rate schedule
                                    scheduler.step()
                                    scheduler_update_flag = True
                            else:
                                scheduler.step()
                                scheduler_update_flag = True
                            if args.local_rank in [0, -1] and scheduler_update_flag and global_step % args.logging_steps == 0:
                                if args.scheduler_type == "epoch":
                                    updated_lr = scheduler.get_last_lr()[0]
                                else:
                                    updated_lr = get_lr(optimizer)
                                print("\ncur steps: %d,  lr: %f" % (cur_global_steps, updated_lr))
                                log_fp.write("Steps: %d, Updated lr: %f\n" % (cur_global_steps, updated_lr))
                                log_fp.flush()
                                writer_info_tb(tb_writer, {"updated_lr": updated_lr}, cur_global_steps, prefix="logging")
                else:
                    output = model(
                        **batch,
                        output_keys=args.gene_output_keys,
                        output_keys_b=args.prot_output_keys,
                        pair_output_keys=args.pair_output_keys,
                        output_attentions=True,
                        output_hidden_states=True
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
                    # print_batch_output(outputs)
                    # print(losses)
                    # print("#####" * 10)
                    loss = calc_loss(args, losses, last_last_loss_list=last_last_loss_list, last_loss_list=last_loss_list)
                    '''
                    if args.n_gpu > 1:
                        reduced_loss = reduce_tensor(loss.data, dist.get_world_size())
                    else:
                        reduced_loss = loss
                    '''
                    if args.local_rank in [0, -1]:
                        # cur_loss = reduced_loss.item()
                        cur_loss = loss.item()
                        end_time = time.time()
                        cur_use_time = end_time - begin_time
                        total_use_time += cur_use_time
                        total_loss += cur_loss
                        logging_loss += cur_loss
                        cur_epoch_loss += cur_loss
                        cur_epoch_time += cur_use_time
                        global_step += 1
                        cur_epoch_step += 1

                        # print(str(losses))
                        # print(str(loss))
                        if global_step % args.gradient_accumulation_steps == 0:
                            print("\rTraining, Epoch: %04d, Batch: %06d, Sample Num: %d, Cur Loss: %.08f, Avg Loss: %.08f" % (
                                epoch + 1,
                                cur_epoch_step,
                                done_sample_num,
                                cur_loss,
                                total_loss/global_step), end="", flush=True
                                  )
                            if global_step == 1 or global_step % args.loss_logging_steps == 0:
                                writer_info_tb(tb_writer, {
                                    "loss": cur_loss
                                }, global_step, prefix="training")
                            if global_step % args.logging_steps == 0:
                                log_fp.write("Training, Epoch: %04d, Batch: %06d, Sample Num: %d, Cur Loss: %.08f, Log Avg loss: %.08f, Global Avg Loss: %.08f, Time: %0.4f\n"
                                             % (
                                                 epoch + 1,
                                                 cur_epoch_step,
                                                 done_sample_num,
                                                 cur_loss,
                                                 logging_loss / lcm(args.logging_steps, args.gradient_accumulation_steps),
                                                 total_loss / global_step,
                                                 cur_use_time)
                                             )
                                log_fp.write(str(losses) + "\n")
                                log_fp.flush()
                                writer_info_tb(tb_writer,
                                               {
                                                   "epoch": epoch + 1,
                                                   "cur_epoch_step": cur_epoch_step,
                                                   "cur_epoch_done_sample_num": done_sample_num,
                                                   "cur_epoch_avg_loss": cur_epoch_loss / cur_epoch_step,
                                                   "cur_batch_loss": cur_loss,
                                                   "global_avg_loss": total_loss / global_step,
                                                   "cur_use_time": cur_use_time,
                                                   "global_step": global_step,
                                                   "log_avg_loss": logging_loss / lcm(args.logging_steps, args.gradient_accumulation_steps),
                                               }, global_step, prefix="logging")
                                logging_loss = 0.0

                    '''
                    for k, v in model.named_parameters():
                        print(k)
                        print(v.grad)
                    '''
                    if args.gradient_accumulation_steps > 1:
                        # The loss of each batch will be divided by gradient_accumulation_steps
                        loss = loss / args.gradient_accumulation_steps
                    if args.fp16:
                        with amp.scale_loss(loss, optimizer) as scaled_loss:
                            scaled_loss.backward()
                    else:
                        loss.backward()
                    no_grad_gradient_accumulation_step = False
                    cur_global_steps += 1
                    if (step + 1) % args.gradient_accumulation_steps == 0:
                        if args.fp16:
                            torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                        else:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        optimizer.step()
                        if scheduler is not None:
                            scheduler_update_flag = False
                            if args.scheduler_type == "epoch":
                                if epoch == 0 and args.lr_update_steps > 0 and cur_global_steps % args.lr_update_steps == 0:
                                    # 第一次epoch内部根据steps调整
                                    # Update learning rate schedule
                                    scheduler.step()
                                    scheduler_update_flag = True
                            else:
                                scheduler.step()
                                scheduler_update_flag = True
                            if args.local_rank in [0, -1] and scheduler_update_flag and global_step % args.logging_steps == 0:
                                if args.scheduler_type == "epoch":
                                    updated_lr = scheduler.get_last_lr()[0]
                                else:
                                    updated_lr = get_lr(optimizer)
                                print("\ncur steps: %d,  lr: %f" % (cur_global_steps, updated_lr))
                                log_fp.write("Steps: %d, Updated lr: %f\n" % (cur_global_steps, updated_lr))
                                log_fp.flush()
                                writer_info_tb(tb_writer, {"updated_lr": updated_lr}, cur_global_steps, prefix="logging")

                        optimizer.zero_grad()
                        no_grad_gradient_accumulation_step = True
                        # print("lr: ", get_lr(optimizer))
                    if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:
                        output_dir = os.path.join(args.output_dir, "checkpoint-step{}".format(global_step))
                        save_check_point(args, model, model_config, tokenizer, output_dir)
            except Exception as e:
                exception_path = "../exception/%s" % args.time_str
                if not os.path.exists(exception_path):
                    os.makedirs(exception_path)
                with open(os.path.join(exception_path, "train_exception_info_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(e) + "\n")
                    afp.flush()
                with open(os.path.join(exception_path, "train_exception_input_%d" % args.local_rank), "a+") as afp:
                    afp.write(str(batch) + "\n")
                    afp.flush()
                debug_path = "../debug/train/local_rank%s/%s/" % ("_" + str(args.local_rank) if args.local_rank >= 0 else "", str(epoch) + "_" + str(step))
                if not os.path.exists(debug_path):
                    os.makedirs(debug_path)
                with open(os.path.join(debug_path, "train_exception_input_details.txt"), "a+") as afp:
                    print_batch(batch, key=None, debug_path=debug_path, wfp=afp, local_rank=args.local_rank)
                    afp.flush()
                raise Exception(e)
        # 一个epoch完成
        if not no_grad_gradient_accumulation_step:
            optimizer.step()
            optimizer.zero_grad()
            print("Has retained gard: rank=%d" % args.local_rank)

        if len(trained_sample_ids) > 0:
            write_processed_sample_ids(dataset_type="train",
                                       sample_ids=trained_sample_ids,
                                       time_str=args.time_str,
                                       epoch=epoch + 1,
                                       local_rank=args.local_rank
                                       )
        # epoch = 1的时候不调整（也就是第二次不调整，后面开始每一个epoch调整一次）
        if epoch > 1 and scheduler is not None and args.scheduler_type == "epoch":
            scheduler.step()
            if args.local_rank in [-1, 0]:
                updated_lr = scheduler.get_last_lr()[0]
                writer_info_tb(tb_writer, {"updated_lr": updated_lr}, cur_global_steps, prefix="logging")
        '''
        if args.n_gpu > 1:
            dist.barrier()
        '''
        if args.local_rank in [-1, 0]:
            logs = {}
            update_flag = False
            # Only evaluate at local_rank=0 or single GPU
            if args.local_rank in [-1, 0] and args.evaluate_during_training and args.dev_data_dir \
                    and (args.start_epoch < 0 or epoch + 1 >= args.start_epoch):
                eval_result = evaluate(args, model, parse_row_func, batch_data_func,
                                       prefix="checkpoint-{}".format(global_step),
                                       log_fp=log_fp)
                print("Eval result:")
                print(eval_result)
                for key, value in eval_result.items():
                    eval_key = "eval_{}".format(key)
                    logs[eval_key] = value
                    if key == best_metric_type:
                        if best_metric_flag and best_metric_value < value or \
                                not best_metric_flag and best_metric_value > value:
                            best_metric_value = value
                            update_flag = True

                logs["update_flag"] = update_flag
                if update_flag and args.test_data_dir:
                    best_metric_model_info.update({"epoch": epoch + 1, "global_step": global_step})
                    test_result = test(args, model, label_size_dict, parse_row_func, batch_data_func,
                                       prefix="checkpoint-{}".format(global_step),
                                       log_fp=log_fp)
                    print("Test result:")
                    print(test_result)
                    for key, value in test_result.items():
                        eval_key = "test_{}".format(key)
                        logs[eval_key] = value
                    best_metric_model_info.update(logs)
            avg_batch_time = round(cur_epoch_time / cur_epoch_step, 2)
            log_fp.write("Epoch Time: %f, Avg time per batch (s): %f\n" % (cur_epoch_time, avg_batch_time))
            if scheduler is not None and args.scheduler_type == "epoch":
                logs["lr"] = scheduler.get_last_lr()[0]
            else:
                logs["lr"] = get_lr(optimizer)
            logs["batch_avg_loss"] = total_loss / global_step
            logs["cur_epoch_loss"] = cur_epoch_loss
            logs["cur_epoch_avg_loss"] = cur_epoch_loss / cur_epoch_step
            logs["epoch"] = epoch + 1
            # print(logs)
            writer_info_tb(tb_writer, logs, global_step, prefix=None)
            log_fp.write(json.dumps({**logs, **{"step": global_step}}, ensure_ascii=False) + "\n")
            log_fp.write("#" * 50 + "\n")
            log_fp.flush()
            print("End epoch: %d" % (epoch + 1))
            # save checkpoint
            output_dir = os.path.join(args.output_dir, "checkpoint-{}".format(global_step))
            if args.save_all:
                save_check_point(args, model, model_config, tokenizer, output_dir)
            elif update_flag:
                if args.delete_old:
                    # delete the old CheckPoint
                    filename_list = os.listdir(args.output_dir)
                    for filename in filename_list:
                        if "checkpoint-" in filename and filename != "checkpoint-{}".format(global_step):
                            shutil.rmtree(os.path.join(args.output_dir, filename))
                save_check_point(args, model, model_config, tokenizer, output_dir)

        last_last_loss_list = last_loss_list
        last_loss_list = [v/batch_total for v in total_loss_detail]

        if args.local_rank in [0, -1]:
            if scheduler is not None and args.scheduler_type == "epoch":
                cur_lr = scheduler.get_last_lr()[0]
            else:
                cur_lr = get_lr(optimizer)
            print("Epoch: %d, batch total: %d, lr: %0.10f" % (epoch + 1, batch_total, cur_lr))
            real_epoch += 1
        '''
        if args.n_gpu > 1:
            dist.barrier()
        '''
        torch.cuda.empty_cache()

    if args.local_rank in [0, -1]:
        run_end_time = time.time()
        tb_writer.close()
        log_fp.write("#" * 25 + "Best Metric" + "#" * 25 + "\n")
        log_fp.write(json.dumps(best_metric_model_info, ensure_ascii=False) + "\n")
        log_fp.write("#" * 50 + "\n")
        avg_time_per_epoch = round((run_end_time - run_begin_time)/real_epoch, 2)
        log_fp.write("Total Time: %f, Avg time per epoch(%d epochs): %f\n" % (run_end_time - run_begin_time, real_epoch, avg_time_per_epoch))
        log_fp.flush()

    if args.n_gpu > 1:
        cleanup()

    if args.local_rank in [0, -1]:
        return global_step, total_loss / global_step, best_metric_model_info

    return None, None, None


def cleanup():
    dist.destroy_process_group()


def save_check_point(args, model, model_config, tokenizer, output_dir):
    '''
    save checkpoint
    :param args:
    :param model:
    :param tokenizer
    :param model_config
    :param output_dir:
    :return:
    '''

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # Take care of distributed/parallel training
    model_to_save = model.module if hasattr(model, "module") else model
    try:
        model_to_save.save_pretrained(output_dir)
    except Exception:
        '''
        model = Model()
        torch.save(model.state_dict(),path)
        state_dict = torch.load(state_dict_path)
        model = model.load_state_dict(state_dict)
        '''
        model_config.save_pretrained(output_dir)
        torch.save(model_to_save, os.path.join(output_dir, "pytorch.pt"))
        torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch.pth"))
    # torch.save(model_to_save, os.path.join(output_dir + "model.pth"))
    if tokenizer:
        tokenizer_dir = os.path.join(output_dir, "tokenizer")
        if not os.path.exists(tokenizer_dir):
            os.makedirs(tokenizer_dir)
        tokenizer.save_pretrained(tokenizer_dir)

    torch.save(args, os.path.join(output_dir, "training_args.bin"))
    print("Saving model checkpoint to %s" % output_dir)
