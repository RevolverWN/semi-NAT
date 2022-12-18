import argparse
import gc
import logging
import os
from collections import namedtuple
from typing import List, Union, Dict

import sacrebleu
import torch
import torch.distributed as dist
from sacrebleu.metrics import BLEUScore
from torch import Tensor

import criterions
import metrics
import hooks
import models
from dataset.dataset import IndexDataset, PairDataset
from dataset.iterator import DataHandler
from dictionary import Dictionary
from tasks import register_task
from tasks.task_base import TaskBase
from tasks.trans_utils import remove_invalid_token, remove_bpe, assign_single_value_long, assign_multi_value_long
from hooks.hook_base import HookList
from metrics.metric_base import MetricList
from utils import tensors_all_reduce, Clock, Checkpoint

logger = logging.getLogger(__name__)

register_name = "translation"

default_dict: Dict[str, Dict] = {
    "src_lang": {"type": str, "help": "give the source language prefix, eg: en"},
    "tgt_lang": {"type": str, "help": "give the target language prefix, eg: de"},
    "seg_num": {"type": int, "default": 4,  "help": ""}
}


@register_task(register_name)
class TranslationTask(TaskBase):
    config = default_dict
    default_metric = "bleu"

    def __init__(self, global_config: namedtuple, task_config: namedtuple):
        self.config = task_config
        self.global_config = global_config

        self.early_stop = False
        self.src_lang = task_config.src_lang
        self.tgt_lang = task_config.tgt_lang

        super(TranslationTask, self).__init__(global_config.data_dir)

        src_dict_path = os.path.join(self.data_dir, "dict." + self.src_lang + ".txt")
        tgt_dict_path = os.path.join(self.data_dir, "dict." + self.tgt_lang + ".txt")
        self.src_dict = Dictionary.load_dictionary_from_file(src_dict_path)
        self.tgt_dict = Dictionary.load_dictionary_from_file(tgt_dict_path)

    @classmethod
    def add_args(cls, parser: argparse.ArgumentParser) -> None:
        for param_name, param_attr in default_dict.items():
            parser.add_argument("--" + param_name, **param_attr)

    def build_data_handler(self, split, chunk_size, max_tokens, max_sentences, rank, gpu_count, device, buffer_size):
        if split == self.global_config.train_prefix:
            self.train_data_handler = DataHandler(self.dataset[split], chunk_size, rank, device, gpu_count, max_tokens,
                                                  max_sentences, buffer_size)
            return self.train_data_handler
        elif split == self.global_config.valid_prefix:
            self.dev_data_handler = DataHandler(self.dataset[split], chunk_size, rank, device, gpu_count, max_tokens,
                                                max_sentences, buffer_size)
            return self.dev_data_handler
        elif split == self.global_config.test_prefix:
            self.test_data_handler = DataHandler(self.dataset[split], chunk_size, rank, device, gpu_count, max_tokens,
                                                 max_sentences, buffer_size)
            return self.test_data_handler

    def load_dataset(self, split: str) -> None:
        src_path = os.path.join(self.data_dir, split + "." + self.src_lang)
        tgt_path = os.path.join(self.data_dir, split + "." + self.tgt_lang)
        src_dataset = IndexDataset(src_path, self.src_dict)
        tgt_dataset = IndexDataset(tgt_path, self.tgt_dict)
        self.dataset[split] = PairDataset(src_dataset, tgt_dataset)

    def build_model(self, model_name: str):
        model_cls = models.registry[model_name]["cls"]
        config = models.registry[model_name]["default_config_dict"]
        self.model = model_cls(config, self.src_dict, self.tgt_dict)
        self.model.to(device=self.global_config.device)
        return self.model

    def build_criterion(self, criterion_name):
        if isinstance(criterion_name, str):
            criterion_name = [criterion_name]

        criterion_dict = {}
        for name in criterion_name:
            criterion_cls = criterions.registry[name]["cls"]
            config = criterions.registry[name]["default_config_dict"]
            criterion_dict[name] = criterion_cls(config)

        self.criterion = criterions.Criterion(criterion_dict, self.global_config.gpu_count)
        return self.criterion

    def build_hooks(self, hooks_name: List):
        self.hook_list = HookList()
        for name in hooks_name:
            hook = self.hook_factory(name)
            self.hook_list.add_hook(hook)

        # if log_hook exists, adjust log_hook as the last element in the container so that log_hook can
        # log all the output that other hooks modify.
        self.hook_list.move_hook_to_last(hooks.registry["log_hook"])

    def hook_factory(self, name):
        hook_cls = hooks.registry[name]
        if name == "log_hook":
            return hook_cls(self, self.global_config.log_interval, self.global_config.rank)
        elif name == "early_stop_hook":
            return hook_cls(self,
                            self.global_config.max_update,
                            self.global_config.stop_min_lr,
                            self.global_config.performance_decay_tolerance,
                            self.global_config.performance_indicator)
        elif name == "test_log_hook":
            return hook_cls(self.global_config.rank)
        elif name == "time_hook":
            return hook_cls()
        elif name == "tensorboard":
            if self.global_config.log_dir is None:
                self.global_config.log_dir = os.path.basename(self.global_config.ckpt_dir)
            return hook_cls(self, self.global_config.rank, self.global_config.log_dir, self.global_config.metrics_name)

    def build_metrics(self, metrics_name: List):
        self.metric_list = MetricList()
        for name in metrics_name:
            metric = self.metric_factory(name)
            self.metric_list.add_metric(name, metric)

    def metric_factory(self, name):
        metric_cls = metrics.registry[name]
        if name == "BLEU":
            return metric_cls(self.global_config.length_beam_size, self.tgt_dict, self.global_config.device)
        elif name == "length_accuracy":
            return metric_cls(self.global_config.length_beam_size, self.tgt_dict, self.global_config.device)

    def convert_batch_on_gpu(self, batch: Dict):
        for key, value in batch.items():
            if isinstance(value, dict):
                for inner_key, inner_value in value.items():
                    batch[key][inner_key] = inner_value.to(device=self.global_config.device)
            else:
                batch[key] = value.to(device=self.global_config.device)

    def train_epoch(self, checkpoint):
        train_epoch_iterator = self.train_data_handler.get_epoch_iterator()
        self.train_data_handler.batch_count_reset()

        self.lr_scheduler.step_begin_epoch(self.train_data_handler.epoch_num)

        logging_outputs = {}
        self.hook_list.on_train_epoch_begin()
        for chunk in train_epoch_iterator:
            self.optimizer.zero_grad()
            loss = self.train_chunk(chunk, self.model, self.criterion)
            loss.backward()
            self.optimizer.step()

            total_updates = self.train_data_handler.total_updates
            self.lr_scheduler.step_update(total_updates)

            if self.validate:
                eval_output = self.eval(self.dev_data_handler, self.model, self.criterion)
                eval_output["epoch_end"] = self.train_data_handler.cur_iterator.epoch_end

                if self.save:
                    checkpoint.save(eval_output)

        self.hook_list.on_train_epoch_end(logging_outputs)

    def train_chunk(self, chunk: List, model, criterion):
        self.hook_list.on_train_chunk_begin()
        loss_list = []
        output_list = []
        for batch in chunk:
            loss, logging_outputs = self.train_step(batch, model, criterion)
            output_list.append(logging_outputs)
            loss_list.append(loss)

        self.hook_list.on_train_chunk_end()
        return sum(loss_list)

    def train_step(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model, criterion):
        self.hook_list.on_train_batch_begin()
        self.convert_batch_on_gpu(batch)
        model.train()

        dist.all_reduce(batch["nsentences"])
        dist.all_reduce(batch["ntokens"])

        loss, logging_outputs = self.train_and_eval(batch, model, criterion)

        logging_outputs.update({"ntokens": batch["ntokens"], "nsentences": batch["nsentences"]})

        self.train_data_handler.batch_count += 1
        self.hook_list.on_train_batch_end(logging_outputs)
        return loss, logging_outputs

    def eval(self, dev_data_handler, model, criterion):
        self.hook_list.on_eval_begin()
        self.metric_list.reset()

        dev_epoch_iterator = dev_data_handler.get_epoch_iterator()
        model.eval()
        logging_output_colletion = []
        with torch.no_grad():
            for chunk in dev_epoch_iterator:
                logging_output_list, hypo_list = self.eval_chunk(chunk, model, criterion)
                for batch, hypos, logging_output in zip(chunk, hypo_list, logging_output_list):
                    logging_output_colletion.append(logging_output)
                    self.metric_list.update(batch["target"], hypos)

        sync_logging_outputs = {}
        keys = logging_output_colletion[0].keys()
        for k in keys:
            temp = 0
            for el in logging_output_colletion:
                temp += el[k]
            sync_logging_outputs[k] = temp
        for k in keys:
            dist.all_reduce(sync_logging_outputs[k])

        ntokens = sync_logging_outputs["ntokens"]
        nsentences = sync_logging_outputs["nsentences"]

        sync_logging_outputs["token_loss"] = sync_logging_outputs["token_loss"] / ntokens
        sync_logging_outputs["length_loss"] = sync_logging_outputs["length_loss"] / nsentences

        sync_logging_outputs["loss"] = sum(
            [loss for name, loss in sync_logging_outputs.items() if name not in ("ntokens", "nsentences")]) * self.global_config.gpu_count

        for name, metric in self.metric_list.items():
            sync_logging_outputs[name] = metric.result()

        self.hook_list.on_eval_end(sync_logging_outputs)
        return sync_logging_outputs

    def eval_chunk(self, chunk: List, model, criterion):
        self.hook_list.on_eval_chunk_begin()
        output_list, hypo_list = [], []
        for batch in chunk:
            logging_outputs, logits = self.eval_step(batch, model, criterion)
            output_list.append(logging_outputs)
            hypo_list.append(logits)

        self.hook_list.on_eval_chunk_end()
        return output_list, hypo_list

    def train_and_eval(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model, criterion, train_flag=True):
        batch_size, seq_len = batch["target"].size()
        tgt_tokens = batch["target"]

        # 1 compute length loss
        model_outputs: dict = model(batch["net_input"]["src_tokens"],
                                    batch["net_input"]["src_masks"],
                                    batch["target"],
                                    batch["target_masks"])
        encoder_outputs = model_outputs["encoder_outputs"]
        length_loss = self.criterion.train_length(model_outputs["length_logits"], (~batch["target_masks"]).sum(-1))

        # 2 compute token loss

        # ---data preprocess---#
        if seq_len < self.config.seg_num:
            supplement = tgt_tokens.new_zeros(tgt_tokens.size(0), self.config.seg_num - seq_len,
                                              device=tgt_tokens.device).fill_(self.tgt_dict.padding_id)
            batch["target"] = torch.cat((tgt_tokens, supplement), dim=1)

        index_range = torch.arange(seq_len)[None, :].expand(batch_size, seq_len)

        all_permutation = []
        for i in range(batch_size):
            all_permutation.append(torch.randperm(seq_len) + i * seq_len)

        all_permutation = torch.stack(all_permutation)
        index_range = index_range.contiguous().view(-1)[all_permutation.view(-1)].view(index_range.size())
        splits_tuple = torch.tensor_split(index_range, self.config.seg_num, dim=1)
        # ---data preprocess---#

        token_loss_list = []

        # first time copy src_emb src_mask tgt_mask
        # second time

        input_tgt_tokens = batch["target"].clone()
        for i, split in enumerate(splits_tuple):
            if i == 0:
                mask_index = torch.cat(splits_tuple, dim=1)
                input_tgt_tokens = assign_single_value_long(input_tgt_tokens, mask_index, self.tgt_dict.mask_id)

                input_tgt_tokens = input_tgt_tokens.masked_fill(batch["target_masks"], self.tgt_dict.padding_id)
            if i - 1 >= 0:
                mask_index = splits_tuple[i - 1]
                input_tgt_tokens = assign_multi_value_long(input_tgt_tokens, mask_index, batch["target"])

            # token_masks_idx = torch.cat([splits_tuple[j] for j in range(len(splits_tuple)) if j != i], dim=1)
            criterion_targets = tgt_tokens.new_zeros(batch_size, seq_len,
                                                     device=tgt_tokens.device).fill_(self.tgt_dict.padding_id)
            criterion_targets = assign_multi_value_long(criterion_targets, splits_tuple[i], batch["target"])

            model_outputs: dict = model(batch["net_input"]["src_tokens"],
                                        batch["net_input"]["src_masks"],
                                        input_tgt_tokens,
                                        batch["target_masks"],
                                        mask_index, # noqa
                                        encoder_outputs)

            token_loss = self.criterion.train_token(model_outputs["logits"],
                                                    criterion_targets,
                                                    batch["target_masks"],
                                                    self.tgt_dict.padding_id
                                                    )
            token_loss_list.append(token_loss)

        token_loss = sum(token_loss_list)
        if train_flag:
            token_loss = token_loss / batch["ntokens"]
            length_loss = length_loss / batch["nsentences"]
            loss = token_loss + length_loss
            loss = loss * self.global_config.gpu_count
            loss_outputs = {"loss": loss, "token_loss": token_loss, "length_loss": length_loss}

            return  loss, loss_outputs
        else:
            loss_outputs = {"token_loss": token_loss, "length_loss": length_loss}

            return loss_outputs

    def eval_step(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model, criterion):
        self.hook_list.on_eval_batch_begin()

        self.convert_batch_on_gpu(batch)
        model.eval()

        logging_output = self.train_and_eval(batch, model, criterion, train_flag=False)

        decoding_outputs: dict = model.generate(batch["net_input"]["src_tokens"],
                                                batch["net_input"]["src_masks"],
                                                batch["target"],
                                                batch["target_masks"],
                                                self.config.seg_num)

        logging_output["ntokens"] = batch["ntokens"]
        logging_output["nsentences"] = batch["nsentences"]

        hypo_tokens = decoding_outputs["hypo_tokens"]

        self.hook_list.on_eval_batch_end()
        return logging_output, hypo_tokens

    def inference(self, test_data_handler, model, criterion):
        """
        we keep this inference function for extension
        """
        self.hook_list.on_inference_begin()

        test_epoch_iterator = test_data_handler.get_epoch_iterator()
        model.eval()
        logging_output_colletion = []
        with torch.no_grad():
            for chunk in test_epoch_iterator:
                logging_output_list, hypo_list = self.inference_chunk(chunk, model, criterion)
                for batch, hypos, logging_output in zip(chunk, hypo_list, logging_output_list):
                    logging_output_colletion.append(logging_output)
                    self.metric_list.update(batch["target"], hypos)

        sync_logging_outputs = {}
        keys = logging_output_colletion[0].keys()
        for k in keys:
            temp = 0
            for el in logging_output_colletion:
                temp += el[k]
            sync_logging_outputs[k] = temp
        for k in keys:
            dist.all_reduce(sync_logging_outputs[k])

        ntokens = sync_logging_outputs["ntokens"]
        nsentences = sync_logging_outputs["nsentences"]
        for name, loss in sync_logging_outputs.items():
            if name not in ("ntokens", "nsentences"):
                if self.criterion.criterions[name].samples_reduce == "ntokens":
                    sync_logging_outputs[name] = loss / ntokens
                else:
                    sync_logging_outputs[name] = loss / nsentences

        sync_logging_outputs["loss"] = sum(
            [loss for name, loss in sync_logging_outputs.items() if name not in ("ntokens", "nsentences")])
        for name, metric in self.metric_list.items():
            sync_logging_outputs[name] = metric.result()

        self.hook_list.on_inference_end(sync_logging_outputs)
        return sync_logging_outputs

    def inference_chunk(self, chunk: List, model, criterion):
        self.hook_list.on_inference_chunk_begin()
        output_list, hypo_list = [], []
        for batch in chunk:
            logging_outputs, logits = self.inference_step(batch, model, criterion)
            output_list.append(logging_outputs)
            hypo_list.append(logits)

        self.hook_list.on_inference_chunk_end()

        return output_list, hypo_list

    def inference_step(self, batch: Union[Dict[str, Union[Tensor, Dict]], List], model, criterion):
        self.hook_list.on_inference_batch_begin()

        self.convert_batch_on_gpu(batch)
        model.eval()
        model_outputs: dict = model(batch["net_input"]["src_tokens"],
                                    batch["net_input"]["src_masks"],
                                    batch["target"],
                                    batch["target_masks"])

        logging_output = self.criterion.eval(model_outputs, batch["target"], batch["target_masks"],
                                             self.tgt_dict.padding_id)

        if not hasattr(model, self.global_config.model_decoding_strategy):
            raise AttributeError(
                "model have not {} decoding_strategy".format(self.global_config.model_decoding_strategy))

        decoding_strategy_method = getattr(model, self.global_config.model_decoding_strategy)
        decoding_outputs: dict = decoding_strategy_method(batch["net_input"]["src_tokens"],
                                                          batch["net_input"]["src_masks"],
                                                          batch["target"],
                                                          batch["target_masks"])

        logging_output["ntokens"] = batch["ntokens"]
        logging_output["nsentences"] = batch["nsentences"]

        hypo_tokens = decoding_outputs["hypo_tokens"]
        self.hook_list.on_inference_batch_end()

        return logging_output, hypo_tokens

    def state_dict(self, train_iterator, model, optimizer, checkpoint) -> Dict:
        # basically, we need save the dataset iterator state, model state, criterion state, optimizer state,
        # lr scheduler state, maybe other config arguments. These are now in a mess, we should arrange them
        # in order.
        return {
            "train_iterator": train_iterator.state_dict(),
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "checkpoint": checkpoint.state_dict()
        }

    def load_state_dict(self, state_dict: Dict, model, train_iterator=None, optimizer=None, checkpoint=None,
                        reset=False) -> None:
        model.load_state_dict(state_dict["model"], strict=False)
        if train_iterator and not reset:
            train_iterator.load_state_dict(state_dict["train_iterator"])
        if optimizer and not reset:
            optimizer.load_state_dict(state_dict["optimizer"])
        if checkpoint:
            checkpoint.load_state_dict(state_dict["checkpoint"])

    @property
    def validate(self) -> bool:
        res = (self.global_config.validate_update_interval != 0 and
               self.train_data_handler.total_updates % self.global_config.validate_update_interval == 0) \
              or self.train_data_handler.cur_iterator.epoch_end
        return res

    @property
    def save(self) -> bool:
        res = self.train_data_handler.epoch_num > self.global_config.save_after_epoch and (
                (self.global_config.save_period != 0 and
                 self.train_data_handler.rank_total_updates % self.global_config.save_period == 0)
                or self.train_data_handler.cur_iterator.epoch_end)
        return res

    @property
    def is_training_state(self) -> bool:
        # here we cab only control early stop condition on epoch number level
        res = self.train_data_handler.epoch_num < self.global_config.max_epoch \
              and (not self.early_stop)

        return res
