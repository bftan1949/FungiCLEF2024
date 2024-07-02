import os
import logging
import sys
from pathlib import Path
import numpy as np
import torch
import pandas as pd

poisonous_lvl = pd.read_csv(r'poison_status_list.csv')
POISONOUS_SPECIES = torch.tensor(poisonous_lvl[poisonous_lvl["poisonous"] == 1].class_id.unique())
EDIBLE_SPECIES = torch.tensor(poisonous_lvl[poisonous_lvl["poisonous"] == 0].class_id.unique())
NUM_CLASS = len(poisonous_lvl)


class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter:
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def get_logger(log_path):
    if not os.path.exists(os.path.dirname(log_path)):
        os.mkdir(os.path.dirname(log_path))

    # 步骤1
    logger = logging.getLogger()
    # logger会过滤掉所有比自己日志等级低的信息, 所以为了让所有信息都能
    # 顺利显示, 需要设置为所有handler中等级最低的
    logger.setLevel(logging.DEBUG)

    # 步骤2
    file_handler = logging.FileHandler(filename=log_path, mode='a')
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    # 步骤4
    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    return logger


def mixup_diff_class(x, y, mf=None, alpha=0.4):

    _x = x.clone()
    _y = y.clone()

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)

    # 创建一个与标签相同大小的布尔掩码，用于标记是否执行混合
    diff_class_mask = (y != y[index])

    x1 = x[diff_class_mask]
    x2 = x[index][diff_class_mask]
    mixed_x = lam * x1 + (1 - lam) * x2
    _x[diff_class_mask] = mixed_x

    _y[diff_class_mask] = -1

    if mf is not None:
        mf1 = mf[diff_class_mask]
        mf2 = mf[index][diff_class_mask]
        mixed_mf = lam * mf1 + (1 - lam) * mf2
        mf[diff_class_mask] = mixed_mf

        return _x, _y, mf

    return _x, _y


def mixup(x, y, alpha=0.4):
    """
    mixup: Beyond Empirical Risk Minimization.ICLR 2018
    https://arxiv.org/pdf/1710.09412.pdf
    https://github.com/facebookresearch/mixup-cifar10

    Args:
        Returns mixed inputs, pairs of targets, and lambda
    """
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)

    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam, index


def accuracy(output, target, topk=(1,)):
    # output: Nx201
    if output.size(1) == NUM_CLASS + 1:
        target[target == -1] = NUM_CLASS

    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].flatten().float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def cal_track1(output, target):
    (acc1,) = accuracy(output, target)
    acc1 = acc1 / 100
    track1 = 1 - acc1
    return track1


def cal_track2(output, target):
    # output: Nx1604
    # target: N
    pred = torch.argmax(output, dim=-1)
    wrong = target[pred != target]

    cost_psc = 100
    cost_esc = 1

    num_psc = sum(torch.isin(wrong, POISONOUS_SPECIES.to(wrong.device)))
    num_esc = sum(torch.isin(wrong, EDIBLE_SPECIES.to(wrong.device)))

    return (cost_psc * num_psc + cost_esc * num_esc) / len(output)


if __name__ == '__main__':
    a = torch.tensor([0, 1, 2, 3, 4, 5])
    b = torch.tensor([7, 6, 1, 2, 3])
    print(torch.isin(b, a))
