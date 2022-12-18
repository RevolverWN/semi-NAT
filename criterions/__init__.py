import importlib
import os
from collections import defaultdict
from typing import Tuple
import torch.distributed as dist

registry = defaultdict(dict)


def register_criterion(criterion_name):
    def register_criterion_cls(cls):
        registry[criterion_name]["cls"] = cls
        registry[criterion_name]["default_config_dict"] = cls.config
        return cls
    return register_criterion_cls


dir_name, base_name = os.path.split(__file__)
for filename in os.listdir(dir_name):
    if filename.endswith(".py") and filename != base_name:
        importlib.import_module("criterions." + filename[:filename.rfind(".py")])


class Criterion(object):
    def __init__(self, criterions, gpu_count):
        self.criterions = criterions
        self.gpu_count = gpu_count


    def train_length(self, length_logits, length_target):
        loss = self.criterions["length_criterion"](length_logits, length_target)
        dist.all_reduce(loss)
        return loss

    def train_token(self, logits, target, target_masks, padding_id):
        loss = self.criterions["label_smoothed_cross_entropy"](logits, target, target_masks, padding_id)
        dist.all_reduce(loss)

        return loss

    def eval(self, model_outputs, target, target_masks, padding_id):
        logging_outputs = {}
        for cri_name, cri in self.criterions.items():
            if cri_name == "label_smoothed_cross_entropy":
                logging_outputs[cri_name] = cri(model_outputs["logits"], target, target_masks, padding_id)
            elif cri_name == "length_criterion":
                logging_outputs[cri_name] = cri(model_outputs["length_logits"], (~target_masks).sum(-1))

        return logging_outputs



