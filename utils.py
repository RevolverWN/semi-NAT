import bisect
import functools
import json
import os
import random
import sys
import time
import warnings
from typing import List, Dict, Iterable, Union
import logging
from dataclasses import dataclass, field

import torch
import torch.distributed as dist
import numpy as np

logger = logging.getLogger(__name__)

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func



def convert_logging_out(logging_outputs: Dict) -> Dict:
    res = {}
    for key, value in logging_outputs.items():
        if torch.is_tensor(value):
            res[key] = value.item()
        else:
            res[key] = value

    for key, value in res.items():
        if key == "lr":
            continue
        if isinstance(value, float) or (torch.is_tensor(value) and torch.is_floating_point(value)):
            res[key] = round(value, 3)
    return res


def infer_language_suffix(path):
    suffix = None
    return suffix


def tensors_all_reduce(collection: Union[List, Dict, torch.Tensor]):
    if isinstance(collection, list) or torch.is_tensor(collection):
        for val in collection:
            dist.all_reduce(val)
    elif isinstance(collection, dict):
        for key in collection.keys():
            dist.all_reduce(collection[key])


@dataclass(frozen=True, order=True)
class CkptData(object):
    filepath: str = field(compare=False)
    BLEU_value: float = field()
    loss_value: float = field()
    best_flag: bool = field(default=False)


class Clock(object):
    def __init__(self, device):
        self.start_time = None
        self.end_time = None
        self.batch_chuck_interval = []
        self.epoch_interval = []
        self.device = device

    def timer(self, func):

        def wrapper(*args, **kwargs):
            self.start()
            func(*args, **kwargs)
            self.pause()

        return wrapper

    def start(self):
        self.start_time = torch.tensor(time.perf_counter(), device=self.device)

    def pause(self):
        self.end_time = torch.tensor(time.perf_counter(), device=self.device)
        interval = self.end_time - self.start_time
        self.batch_chuck_interval.append(interval)

    def begin_epoch_timer(self):
        self.batch_chuck_interval.clear()

    def epoch_terminate(self):
        self.epoch_interval.append(sum(self.batch_chuck_interval))

    @property
    def current_time_interval(self):
        return self.batch_chuck_interval[-1]


class Checkpoint(object):
    def __init__(self, ckpt_dir, args, task, model, save_metric=None, optimizer=None, train_iterator=None):
        self.ckpt_dir = ckpt_dir
        self.args = args
        self.task = task
        self.model = model
        self.optimizer = optimizer
        self.train_iterator = train_iterator
        self.metric = save_metric

        self.best_metric_value = None
        self.best_metric_ori_path = None
        self.BEST_METRIC_PATH = os.path.join(self.ckpt_dir, "best_checkpoint.pt")

        self.last_metric_value = None
        self.last_metric_ori_path = None
        self.LAST_METRIC_PATH = os.path.join(self.ckpt_dir, "last_checkpoint.pt")

        self.metric_list = []
        self.path_list = []

    def load_checkpoint(self, ckpt_name=None, reset=False):
        """
        here we have variable name abuse,
        :param rank:
        """
        if os.path.exists(self.ckpt_dir):
            if self.args.rank == 0:
                with open(os.path.join(self.ckpt_dir, "all_args.json"), 'r', encoding='utf-8') as f:
                    restore_args = json.load(f)
                args_dict = vars(self.args)
                Checkpoint.check_dict_consistency(restore_args, args_dict)

            if ckpt_name is not None:
                check_point = os.path.join(self.ckpt_dir, ckpt_name)
                if os.path.exists(check_point):
                    state_dict = torch.load(check_point)
                    self.task.load_state_dict(state_dict, self.model, self.train_iterator, self.optimizer, self, reset=reset)
                    logger.info("load checkpoint from {}.".format(ckpt_name))
                else:
                    logger.info("no checkpoint found in {}, train model from scratch.".format(self.ckpt_dir))

        elif self.args.rank == 0:
            os.makedirs(self.ckpt_dir, exist_ok=True)
            # save command line arguments first, so we can get user's specific configuration other than default
            # configuration. if we save all or partial parsed arguments, these arguments includes plenty of default
            # value, resulting in a mess without knowing user's preference. this file can be a reference as recovery
            # training.
            with open(os.path.join(self.ckpt_dir, "cmd_args.txt"), 'w', encoding='utf-8') as f:
                f.write(' '.join(sys.argv[1:]))
            # we use this to check recovery training configuration.
            with open(os.path.join(self.ckpt_dir, "all_args.json"), 'w', encoding='utf-8') as f:
                json.dump(vars(self.args), f)

            logger.info("no checkpoint found in {}, train model from scratch.".format(self.ckpt_dir))

    def save(self, logging_outputs: Dict):

        if self.train_iterator.cur_iterator.epoch_end:
            file_name = "epoch{}_end_updates{}.pt".format(self.train_iterator.epoch_num,
                                                          self.train_iterator.rank_total_updates)
        else:
            file_name = "epoch{}_updates{}.pt".format(self.train_iterator.epoch_num,
                                                      self.train_iterator.rank_total_updates)

        save_path = os.path.join(self.ckpt_dir, file_name)

        if self.args.rank == 0:
            metric_value = logging_outputs[self.metric]
            if self.metric == "loss":
                metric_value = -metric_value

            if self.best_metric_value is None:
                self.best_metric_value = metric_value
                self.best_metric_ori_path = save_path
            else:
                if metric_value > self.best_metric_value:
                    if os.path.exists(self.BEST_METRIC_PATH):
                        os.rename(self.BEST_METRIC_PATH, self.best_metric_ori_path)

                    self.best_metric_value = metric_value
                    self.best_metric_ori_path = save_path
                else:
                    if os.path.exists(self.best_metric_ori_path):
                        os.rename(self.best_metric_ori_path, self.BEST_METRIC_PATH)

            if os.path.exists(self.LAST_METRIC_PATH):
                if self.last_metric_ori_path == self.best_metric_ori_path:
                    os.rename(self.LAST_METRIC_PATH, self.BEST_METRIC_PATH)
                else:
                    os.rename(self.LAST_METRIC_PATH, self.last_metric_ori_path)

            self.last_metric_ori_path = save_path
            save_path = self.LAST_METRIC_PATH

            torch.save(self.task.state_dict(self.train_iterator, self.model, self.optimizer, self), save_path)

    def state_dict(self):
        return {'best_metric_value': self.best_metric_value,
                'best_metric_ori_path': self.best_metric_ori_path,
                'last_metric_value': self.last_metric_value,
                'last_metric_ori_path': self.last_metric_ori_path}

    def load_state_dict(self, state_dict):
        self.best_metric_value = state_dict['best_metric_value']
        self.best_metric_ori_path = state_dict['best_metric_ori_path']
        self.last_metric_value = state_dict['last_metric_value']
        self.last_metric_ori_path = state_dict['last_metric_ori_path']
        # self.last_metric_ori_path = './check_points/distilled_data/iwslt14de-en_tgt_emb2/epoch430_end_updates55470.pt'

    def ckpt_path_reorder(self, metric_value: float, path: str):
        """
        insert metric_value and path into ordered metrics and paths, then rename the saved path based on metric order.
        :param metric_value:
        :param path:
        """
        insert_point = bisect.bisect_right(self.metric_list, metric_value)
        self.metric_list.insert(insert_point, metric_value)
        self.path_list.insert(insert_point, path)
        length = len(self.metric_list)

        for rank, (_, path) in enumerate((list(zip(self.metric_list, self.path_list)))):
            dst_path = path[: path.rfind("_rank")] + "_rank{}".format(length - 1 - rank)
            os.rename(path, dst_path)
            self.path_list[rank] = dst_path

    @staticmethod
    def check_dict_consistency(dictionary1: dict, dictionary2: dict):
        if dictionary1 != dictionary2:
            if dictionary1.keys() != dictionary2.keys():
                dict1_key_set = set(dictionary1.keys())
                dict2_key_set = set(dictionary2.keys())
                # raise KeyError(
                #     "these keys {} are not consistent".format(dict2_key_set.symmetric_difference(dict1_key_set)))
                logger.warning(
                    "these keys {} are not consistent".format(dict2_key_set.symmetric_difference(dict1_key_set)))
            else:
                # dump a tuple datatype to a json file, but when loaded, it turns to list datatype, so we must check this
                # condition further.
                val_mismatch_key = []
                for key in dictionary1.keys():
                    if key == "init_method":  # init_method port is a random number, so not to check consistence.
                        continue
                    val1, val2 = dictionary1[key], dictionary2[key]
                    if val1 != val2 and isinstance(val1, Iterable) and \
                            isinstance(val1, Iterable) and tuple(val1) != tuple(val2):
                        val_mismatch_key.append(key)

                if len(val_mismatch_key) > 0:
                    # raise ValueError(
                    #     "these keys {} values are different in the two dictionarys".format(val_mismatch_key))
                    logger.warning("these keys {} values are different in the two dictionarys".format(val_mismatch_key))


if __name__ == '__main__':
    pass