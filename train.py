import argparse
import logging
import os
import sys
from collections import namedtuple
from typing import Dict

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.tensorboard import SummaryWriter

import criterions
import lr_schedulers
import models
import optimizers
import options
import tasks
import metrics
from utils import Checkpoint, convert_logging_out

logging.basicConfig(format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO,
                    stream=sys.stdout)
# file_handler = logging.FileHandler()
logger = logging.getLogger("train")

MODULE_DICT = {"task": tasks,
               "model": models,
               "criterion": criterions,
               "lr_scheduler": lr_schedulers,
               "optimizer": optimizers}


class ArgsAdapter(object):
    def __init__(self, **entries):
        self.__dict__.update(entries)


def add_specific_args(parser: argparse.ArgumentParser, known_args: argparse.Namespace):
    for name, module in MODULE_DICT.items():
        if not hasattr(module, "registry"):
            raise AttributeError("module {} has not registry attribute".format(module.__name__))
        if not hasattr(known_args, name):
            raise AttributeError("args has not {} attribute".format(name))

        # cls_name maybe str or tuple
        cls_name = getattr(known_args, name)
        if isinstance(cls_name, str):
            module.registry[cls_name]["cls"].add_args(parser)  # noqa
        elif isinstance(cls_name, (tuple, list)):
            for el in cls_name:
                module.registry[el]["cls"].add_args(parser)  # noqa


def main(proc_id: int, args: argparse.Namespace):
    # 1 setup device and distributed training
    if args.device == "cuda":
        args.device = "cuda:{}".format(proc_id)
        torch.cuda.set_device(args.device)

    args.rank = proc_id
    dist.init_process_group(backend=args.backend,
                            init_method=args.init_method,
                            world_size=args.gpu_count,
                            rank=args.rank)

    # 2 reset all modules configuration and build all modules
    for name, module in MODULE_DICT.items():
        cls_name = getattr(args, name)
        if isinstance(cls_name, str):
            module_default_config = module.registry[cls_name]["default_config_dict"]
            module_config = {args_name: getattr(args, args_name) for args_name in module_default_config.keys()}
            config = ArgsAdapter(**module_config)
            module.registry[cls_name]["default_config_dict"] = config
        elif isinstance(cls_name, (tuple, list)):
            for el in cls_name:
                module_default_config = module.registry[el]["default_config_dict"]
                module_config = {args_name: getattr(args, args_name) for args_name in module_default_config.keys()}
                config = ArgsAdapter(**module_config)
                module.registry[el]["default_config_dict"] = config

    logger.info(args)
    task_cls = tasks.registry[args.task]["cls"]
    config = tasks.registry[args.task]["default_config_dict"]
    task = task_cls(args, config)

    task.load_dataset(args.train_prefix)
    task.load_dataset(args.valid_prefix)
    task.load_dataset(args.test_prefix)

    train_data_handler = task.build_data_handler(args.train_prefix, args.train_chunk_size, args.max_tokens, args.max_sentences,
                                                 args.rank, args.gpu_count, args.device, args.buffer_size)
    dev_data_handler = task.build_data_handler(args.valid_prefix, args.dev_chunk_size, args.max_tokens, args.max_sentences,
                                               args.rank, args.gpu_count, args.device, args.buffer_size)
    test_data_handler = task.build_data_handler(args.test_prefix, args.test_chunk_size, args.max_tokens, args.max_sentences,
                                                args.rank, args.gpu_count, args.device, args.buffer_size)

    logger.info("{} samples in train dataset".format(len(train_data_handler.dataset)))
    logger.info("{} batches(updates) in every gpu train dataset over {} gpu".format(train_data_handler.rank_batches_num,
                                                                         train_data_handler.gpu_count))
    logger.info("{} samples in dev dataset".format(len(dev_data_handler.dataset)))
    logger.info("{} samples in test dataset".format(len(test_data_handler.dataset)))

    model = task.build_model(args.model)
    criterion = task.build_criterion(args.criterion)
    optimizer = task.build_optimizer(args.optimizer, model)
    lr_scheduler = task.build_lr_scheduler(args.lr_scheduler, optimizer)
    task.build_hooks(args.hooks_name)
    task.build_metrics(args.metrics_name)

    # 3 load state_dict and start training
    checkpoint = Checkpoint(args.ckpt_dir, args, task, model, args.save_metric, optimizer, train_data_handler)
    checkpoint.load_checkpoint(args.ckpt_name)

    train_logging_outputs = {}
    task.hook_list.on_train_begin()
    while task.is_training_state:
        task.train_epoch(checkpoint)

    logger.info("start inference")
    task.inference(test_data_handler, model, criterion)
    task.hook_list.on_train_end(train_logging_outputs)


def main_bak(proc_id: int, args: argparse.Namespace):
    # 1 setup device and distributed training
    if args.device == "cuda":
        args.device = "cuda:{}".format(proc_id)
        torch.cuda.set_device(args.device)

    args.rank = proc_id
    dist.init_process_group(backend=args.backend,
                            init_method=args.init_method,
                            world_size=args.gpu_count,
                            rank=args.rank)

    if args.rank == 0:
        if args.log_dir is None:
            args.log_dir = os.path.basename(args.data_dir)
        train_writer = SummaryWriter(log_dir=f'logs/{args.log_dir}_train')
        valid_writer = SummaryWriter(log_dir=f'logs/{args.log_dir}_valid')

    # 2 reset all modules configuration and build all modules
    for name, module in MODULE_DICT.items():
        # replace default_config_dict values from command line arguments or original default values.
        cls_name = getattr(args, name)
        module_default_config = module.registry[cls_name]["default_config_dict"]
        module_config = {args_name: getattr(args, args_name) for args_name in module_default_config.keys()}

        # encapsulate the module_config in a namedtuple then assign the new config to default_config_dict
        # Config = namedtuple("Config", list(module_config.keys()))
        # config = Config(*list(module_config.values()))  # noqa
        config = ArgsAdapter(**module_config)
        module.registry[cls_name]["default_config_dict"] = config

    logger.info(args)
    task_cls = tasks.registry[args.task]["cls"]
    config = tasks.registry[args.task]["default_config_dict"]
    task = task_cls(args, config)

    task.load_dataset("train")
    task.load_dataset("valid")
    train_iterator = task.build_data_handler("train", args.train_chunk_size, args.max_tokens, args.max_sentences,
                                             args.rank, args.gpu_count, args.device, args.buffer_size)
    dev_iterator = task.build_data_handler("valid", args.dev_chunk_size, args.max_tokens, args.max_sentences,
                                           args.rank, args.gpu_count, args.device, args.buffer_size)

    logger.info("%d samples in train dataset" % len(train_iterator.dataset))
    logger.info("%d batches(updates) in one GPU over train dataset " % train_iterator.rank_batches_num)
    logger.info("%d samples in dev dataset" % len(dev_iterator.dataset))

    model = task.build_model(args.model)
    criterion = task.build_criterion(args.criterion, train_iterator)
    optimizer = task.build_optimizer(args.optimizer, model)
    lr_sheduler = task.build_lr_scheduler(args.lr_scheduler, optimizer)

    # 3 load state_dict and start training
    checkpoint = Checkpoint(args.ckpt_dir, args, task, model, args.save_metric, optimizer, train_iterator)
    checkpoint.load_checkpoint(args.ckpt_name, args.reset)

    startup_counter, train_tgt_emb_counter, train_copy_emb_counter, = 0, 0, 0
    startup_threshold, train_tgt_emb_threshold, train_copy_emb_threshold = 0.2, 0.3, 0.2

    startup_decay_tolerance, train_tgt_emb_decay_tolerance, train_copy_emb_decay_tolerance = 0, 50, 15

    train_bridge_counter = 0
    train_bridge_threshold = 0.2
    bridge_decay_tolerance = 0

    # startup_min_value, train_tgt_emb_min_value, train_copy_emb_min_value = np.inf, np.inf, np.inf
    min_value = np.inf
    best_epoch = None

    def switch_state(train_copy_embedding_flag):
        if train_copy_embedding_flag:
            logger.info("*" * 50)
            logger.info("train copy embedding")
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = True
            for param in model.decoder.token_emb.parameters():
                param.requires_grad = False

            logger.info(
                "num. model params: {:,} (num. trained: {:,})".format(
                    sum(p.numel() for p in model.parameters()),
                    sum(p.numel() for p in model.parameters() if p.requires_grad)
                )
            )

            model.config.use_ground_truth_target = False
            model.config.src_embedding_copy = True
            criterion.config.kl_loss = True
        else:
            logger.info("*" * 50)
            logger.info("train target embedding")
            for name, child in model.named_children():
                for param in child.parameters():
                    param.requires_grad = False
            for param in model.decoder.token_emb.parameters():
                param.requires_grad = True

            logger.info(
                "num. model params: {:,} (num. trained: {:,})".format(
                    sum(p.numel() for p in model.parameters()),
                    sum(p.numel() for p in model.parameters() if p.requires_grad)
                )
            )

            model.config.use_ground_truth_target = True
            model.config.src_embedding_copy = False
            criterion.config.kl_loss = False

        nonlocal min_value, best_epoch
        min_value = np.inf
        best_epoch = None

    def train_epoch(threshold, counter):
        nonlocal min_value, best_epoch
        train_epoch_iterator = train_iterator.get_epoch_iterator()
        task.train_timer.begin_epoch_timer()
        for chunk in train_epoch_iterator:
            optimizer.zero_grad()
            logging_outputs, _ = task.chunk_step(chunk, model, criterion, train_flag=True)
            # loss = logging_outputs["loss"]
            # loss.backward()
            # token_ave_loss = logging_outputs["token_ave_loss"]
            # token_ave_loss.backward(retain_graph=True)
            # sentence_ave_loss = logging_outputs["sentence_ave_loss"]
            # sentence_ave_loss.backward()

            # optimizer.step(logging_outputs["ntokens"], logging_outputs["valid_batch_flag"])

            total_updates = train_iterator.rank_total_updates
            lr_sheduler.step_update(total_updates)

            # if args.rank == 0:
            #     train_writer.add_scalar('loss', label_smoothing_loss, total_updates)  # noqa

            if total_updates % args.log_interval == 0:
                if args.rank == 0:
                    logger.info("epoch:{} | updates in epoch:{} | total updates:{} | {}".format(
                        train_iterator.epoch_num,
                        train_iterator.updates_in_epoch,
                        total_updates,
                        convert_logging_out(logging_outputs)))

            if (
                    args.save_period != 0 and total_updates % args.save_period == 0) or train_iterator.cur_iterator.epoch_end:
                task.eval_timer.begin_epoch_timer()
                eval_output = task.eval(dev_iterator, model, criterion)
                eval_output["epoch_end"] = train_iterator.cur_iterator.epoch_end
                task.eval_timer.epoch_terminate()

                metric_value = eval_output[args.performance_indicator]
                if args.performance_indicator == "BLEU":
                    metric_value = -metric_value

                if min_value - metric_value > threshold:
                    min_value = metric_value
                    counter = 0
                    best_epoch = train_iterator.epoch_num

                elif train_epoch_iterator.epoch_end:
                    counter += 1  # noqa

                eval_output["best " + args.performance_indicator + " epoch"] = best_epoch

                if train_iterator.epoch_num > args.save_after_epoch:
                    checkpoint.save(eval_output)

                if args.rank == 0:
                    if args.performance_indicator == "BLEU":
                        eval_output["best " + args.performance_indicator] = round(-min_value, 3)
                    else:
                        eval_output["best " + args.performance_indicator] = round(min_value, 3)

                    valid_res = convert_logging_out(eval_output)

                    logger.info("valid {}".format(valid_res))
                    # valid_writer.add_scalar('loss', valid_res["loss"], total_updates)  # noqa
                    # valid_writer.add_scalar('BLEU', valid_res["BLEU"], total_updates)

        task.train_timer.epoch_terminate()
        return counter

    logger.info("*" * 50)
    logger.info("starting phase, not freeze parameters")
    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )
    model.config.use_ground_truth_target = False
    model.config.use_bridge = False
    model.config.src_embedding_copy = True
    criterion.config.kl_loss = False

    while startup_counter < startup_decay_tolerance:
        startup_counter = train_epoch(startup_threshold, startup_counter)

    # train bridge
    logger.info("*" * 50)
    logger.info("train bridge")
    for name, child in model.named_children():
        for param in child.parameters():
            param.requires_grad = False
    for param in model.decoder.bridge.parameters():
        param.requires_grad = True

    model.config.use_ground_truth_target = False
    model.config.use_bridge = True
    model.config.src_embedding_copy = True
    criterion.config.kl_loss = True

    logger.info(
        "num. model params: {:,} (num. trained: {:,})".format(
            sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad)
        )
    )

    while True:
        train_bridge_counter = train_epoch(train_bridge_threshold, train_bridge_counter)

    # train_copy_embedding_flag = True
    # switch_state(train_copy_embedding_flag)
    # while True:
    #     if train_copy_embedding_flag:
    #         if train_copy_emb_counter < train_copy_emb_decay_tolerance:
    #             train_copy_emb_counter = train_epoch(train_copy_emb_threshold, train_copy_emb_counter)
    #         else:
    #             train_copy_embedding_flag = False
    #             switch_state(train_copy_embedding_flag)
    #             train_tgt_emb_counter = 0
    #     else:
    #         if train_tgt_emb_counter < train_tgt_emb_decay_tolerance:
    #             train_tgt_emb_counter = train_epoch(train_tgt_emb_threshold, train_tgt_emb_counter)
    #         else:
    #             train_copy_embedding_flag = True
    #             switch_state(train_copy_embedding_flag)
    #             train_copy_emb_counter = 0


def args_postprocess(args: argparse.Namespace):
    args.max_tokens = int(args.max_tokens / args.gpu_count)


if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    parser = argparse.ArgumentParser(allow_abbrev=False)
    options.get_train_options(parser)
    known_args, _ = parser.parse_known_args()
    add_specific_args(parser, known_args)

    args = parser.parse_args()
    args_postprocess(args)
    torch.multiprocessing.spawn(fn=main, args=(args,), nprocs=args.gpu_count)
