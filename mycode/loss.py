from torch import nn
import numpy as np
import torch
import pandas as pd
from torch.nn import functional as F


class ProserLoss(nn.Module):
    def __init__(self, beta=1):
        super().__init__()
        self.celoss = nn.CrossEntropyLoss()
        self.beta = beta

    def forward(self, logits, labels):
        # logits: Nx201
        # labels: N
        unknow_class = logits.size(1) - 1
        labels[labels == -1] = unknow_class
        loss1 = self.celoss(logits, labels)

        mask = (labels != unknow_class)
        if sum(mask) > 0:
            mask_logits = logits[mask]
            mask_labels = labels[mask]
            mask_logits[:, mask_labels] = 0
            open_set_labels = torch.zeros_like(mask_labels, dtype=torch.long).to(logits.device)
            open_set_labels = open_set_labels + unknow_class
            loss2 = self.celoss(mask_logits, open_set_labels)
            return loss1 + self.beta * loss2
        else:
            return loss1


class PoisonousCostLoss(nn.Module):
    def __init__(self, c_psc=100, c_esc=1):
        super().__init__()

        file_path = r'poison_status_list.csv'

        df = pd.read_csv(file_path)
        poi_class = np.asarray(df[df['poisonous'] == 1]['class_id'])
        edi_class = np.asarray(df[df['poisonous'] == 0]['class_id'])

        self.class_num = len(df)
        self.unknow_class = self.class_num

        # 多出来的一行和一列是给openset准备的，因为算track2的时候不考虑openset，所以设为0
        cost_matrix = torch.zeros(size=(self.class_num + 1, self.class_num + 1))
        mask = torch.eye(self.class_num + 1, dtype=torch.bool)
        cost_matrix[~mask] = 0.5
        cost_matrix[poi_class[:, None], edi_class] = c_esc
        cost_matrix[edi_class[:, None], poi_class] = c_psc

        col_sum = cost_matrix.sum(dim=0, keepdim=True)  # 1x201
        self.cost_matrix = cost_matrix / col_sum

    def forward(self, logits, labels):
        # logits: Nx1604
        # targets: N

        if logits.size(1) == self.class_num:
            padding = torch.zeros(size=(logits.size(0), 1)).to(logits.device)
            logits = torch.concatenate([logits, padding], dim=1)  # Nx1605

        labels[labels == -1] = self.unknow_class
        self.cost_matrix = self.cost_matrix.to(logits.device)

        log_soft = F.log_softmax(logits, dim=-1).T  # Nx1604 -> 1604xN
        cost = self.cost_matrix[:, labels]  # 1604xN
        loss = (log_soft * cost).sum(dim=0).mean()
        return loss


class SeesawLossWithLogits(nn.Module):
    """
    This is unofficial implementation for Seesaw loss,
    which is proposed in the techinical report for LVIS workshop at ECCV 2020.
    For more detail, please refer https://arxiv.org/pdf/2008.10032.pdf.
    Args:
    class_counts: The list which has number of samples for each class.
                  Should have same length as num_classes.
    p: Scale parameter which adjust the strength of panishment.
       Set to 0.8 as a default by following the original paper.
    """

    def __init__(self, p: float = 0.8):
        super().__init__()

        file_path = r'/data/dataset/fungi2024/train/FungiCLEF2023_train_metadata_PRODUCTION.csv'

        train_df = pd.read_csv(file_path)
        class_id = np.array(train_df['class_id'])
        class_counts = np.bincount(class_id)

        class_counts = torch.FloatTensor(class_counts)
        conditions = class_counts[:, None] > class_counts[None, :]
        trues = (class_counts[None, :] / class_counts[:, None]) ** p
        falses = torch.ones(len(class_counts), len(class_counts))
        self.s = torch.where(conditions, trues, falses)
        self.class_num = len(class_counts)
        self.eps = 1.0e-6

    def forward(self, logits, targets):
        targets = nn.functional.one_hot(targets, num_classes=self.class_num).float().to(targets.device)
        self.s = self.s.to(targets.device)
        max_element, _ = logits.max(axis=-1)
        logits = logits - max_element[:, None]  # to prevent overflow

        numerator = torch.exp(logits)
        denominator = ((1 - targets)[:, None, :]
                       * self.s[None, :, :]
                       * torch.exp(logits)[:, None, :]).sum(axis=-1) + torch.exp(logits)

        sigma = numerator / (denominator + self.eps)
        loss = (- targets * torch.log(sigma + self.eps)).sum(-1)
        return loss.mean()


def CrossEntropyLoss():
    return nn.CrossEntropyLoss()


if __name__ == '__main__':
    # params = {'a':1, 'b':2}
    loss_fn = ProserLoss(beta=1).cuda()
    logits = torch.rand(size=(3, 5)).cuda()
    labels = torch.zeros(3, dtype=torch.long).cuda() + 4
    print(labels)
    loss = loss_fn(logits, labels)
    # print(loss)
