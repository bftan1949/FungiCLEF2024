import numpy as np
import pandas as pd
import torch
from torch.nn import functional as F

# torch.manual_seed(100)
# logits = torch.randn((2, 3), requires_grad=True)
# a = F.gumbel_softmax(logits, dim=-1, hard=True)
# print(a)
# a = a.unsqueeze(-1)
# b = torch.rand(2, 3, 2, requires_grad=True)
# print(b)
# loss = (b * a).sum()
# loss.backward()
# print(logits.grad)

# #
# local_dir = "/data/tbf/fungi/pretrain/clip"
# train_meta_file = '/data/dataset/fungi2024/train/FungiCLEF2023_train_metadata_PRODUCTION.csv'
# val_meta_file = '/data/dataset/fungi2024/test/FungiCLEF2024_TestMetadata.csv'
# #
# meta_data = pd.read_csv(val_meta_file)
# #
# pois = meta_data['poisonous']
# print(len(pois))
# print(len(pois[pois==1]))

# val_meta_data = pd.read_csv(val_meta_file)
#
# na_index = np.asarray(train_meta_data['Substrate'].isnull())
# print(na_index)
#
# # todo
# attr = 'Substrate'
#
#
# tensor = torch.load('/data/tbf/fungi/mycode/cache/train/MetaSubstrate.pth')
# print(tensor[22])
# print(train_meta_data.iloc[22]['Substrate'])

if __name__ == '__main__':
    # 假设我们有一个batch的logits
    prob = torch.tensor([[0.5] + [0.5 / 1603 for _ in range(1603)]])
    print(prob)

    # 计算每个实例的softmax
    # probs = F.softmax(logits, dim=1)
    # print(probs)

    # 计算每个实例的熵
    entropy = -torch.sum(prob * torch.log(prob), dim=1)

    print(entropy)
