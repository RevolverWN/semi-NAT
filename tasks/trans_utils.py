from typing import List

import torch


def remove_invalid_token(sentences: List[List], invalid_token: List) -> List[List]:
    return [list(filter(lambda x: x not in invalid_token, line)) for line in sentences]


def remove_bpe(sentence: str) -> str:
    return (sentence + " ").replace("@@ ", "").rstrip()


def remove_repeat_token(sentence: str) -> str:
    sentence = sentence.strip().split()
    res = []
    for token in sentence:
        if len(res) == 0:
            res.append(token)
        elif token != res[-1]:
            res.append(token)

    return " ".join(res)


def assign_single_value_long(x, i, y):
    """

    :param x: [batch_size, seq_len]
    :param i: index [batch_size, < seq_len]
    :param y:
    :return:
    """
    temp = x.clone()
    b, l = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)  # n times of l   shape [batch_size, 1]
    temp.view(-1)[i.view(-1)] = y
    return temp


def assign_multi_value_long(x, i, y):
    """

    :param x: [batch_size, seq_len]
    :param i: index [batch_size, < seq_len]
    :param y:
    :return:
    """
    temp = x.clone()
    b, l = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)
    i = i.view(-1)
    temp.view(-1)[i] = y.view(-1)[i]
    return temp


def assign_single_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y


def assign_multi_value_byte(x, i, y):
    x.view(-1)[i.view(-1).nonzero()] = y.view(-1)[i.view(-1).nonzero()]


def assign_multi_value(x, i, y):
    """

    :param x: [batch_size, seq_len, hidden_dim]
    :param i: index [batch_size, < seq_len]
    :param y: [batch_size, seq_len, hidden_dim]
    :return:
    """
    temp = x.clone()
    b, l, h = x.size()
    i = i + torch.arange(0, b * l, l, device=i.device).unsqueeze(1)
    temp.view(-1)[i.view(-1)] = y.view(-1)[i.view(-1)]
    return temp
