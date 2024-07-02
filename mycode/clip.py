import sys
import time
import numpy as np
import pandas as pd
from transformers import CLIPTokenizer, CLIPTextModel
import torch

# 指定本地模型的路径
local_dir = "/data/tbf/fungi/pretrain/clip"
train_meta_file = '/data/dataset/fungi2024/train/FungiCLEF2023_train_metadata_PRODUCTION.csv'
val_meta_file = '/data/dataset/fungi2024/val/FungiCLEF2023_val_metadata_PRODUCTION.csv'

train_meta_data = pd.read_csv(train_meta_file)
val_meta_data = pd.read_csv(val_meta_file)

train_labels = np.asarray(train_meta_data['class_id'])
val_labels = np.asarray(val_meta_data['class_id'])

# todo
cols = ['countryCode', 'Substrate', 'Habitat', 'MetaSubstrate']

# 从本地加载分词器和文本编码器
tokenizer = CLIPTokenizer.from_pretrained(local_dir)
text_model = CLIPTextModel.from_pretrained(local_dir).cuda()

chunk_size = 10000
for attr in cols:
    t1 = time.time()

    na_index = np.asarray(train_meta_data[attr].isnull())

    meta_data = [str(each) for each in list(train_meta_data[attr])]

    data = torch.zeros((len(meta_data), 512))

    for i in range(int(np.ceil(len(meta_data) / chunk_size))):
        inputs = tokenizer(meta_data[i * chunk_size:(i + 1) * chunk_size],
                           return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            text_features = text_model(**inputs).last_hidden_state[:, 0, :]
        data[i * chunk_size:(i + 1) * chunk_size] = text_features.detach().cpu()

    data[na_index] = 0

    torch.save(data, f'./cache/train/{attr}.pth')
    print(f'./cache/train/{attr}.pth saved, cost {time.time() - t1}')

    del inputs, text_features, data

chunk_size = 10000
for attr in cols:
    t1 = time.time()

    na_index = np.asarray(val_meta_data[attr].isnull())

    meta_data = [str(each) for each in list(val_meta_data[attr])]

    data = torch.zeros((len(val_meta_data), 512))

    for i in range(int(np.ceil(len(meta_data) / chunk_size))):
        inputs = tokenizer(meta_data[i * chunk_size:(i + 1) * chunk_size],
                           return_tensors="pt", padding=True, truncation=True)
        with torch.no_grad():
            inputs = {k: v.cuda() for k, v in inputs.items()}
            text_features = text_model(**inputs).last_hidden_state[:, 0, :]
        data[i * chunk_size:(i + 1) * chunk_size] = text_features.detach().cpu()

    data[na_index] = 0

    torch.save(data, f'./cache/val/{attr}.pth')

    print(f'./cache/val/{attr}.pth saved, cost {time.time() - t1}')

    del inputs, text_features, data
