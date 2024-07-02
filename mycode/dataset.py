import os
import sys
import math
import torch
from PIL import Image
import scipy.io
from torch.utils.data import Dataset
import random
from pathlib import Path
from PIL import ImageFilter
import numpy as np
import cv2
import pandas as pd

random.seed(100)


def get_img(path):
    im_bgr = cv2.imread(path)
    im_rgb = im_bgr[:, :, ::-1]
    return im_rgb


class FGVCMetaDataset(Dataset):
    def __init__(self, data_path, split, transform=None):

        super().__init__()

        self.data_path = Path(data_path)
        self.transform = transform

        assert split in ['train', 'val']

        if split == 'train':
            self.data_path = self.data_path / 'train'
            self.meta_data = pd.read_csv(self.data_path / 'FungiCLEF2023_train_metadata_PRODUCTION.csv')
            self.data_path = self.data_path / 'DF20'
        else:
            self.data_path = self.data_path / 'val'
            self.meta_data = pd.read_csv(self.data_path / 'FungiCLEF2023_val_metadata_PRODUCTION.csv')
            self.data_path = self.data_path / 'DF21'

        self.labels = self.meta_data['class_id'].values

        # meta_cols = ['month', 'day', 'Latitude', 'Longitude', 'countryCode', 'Substrate', 'Habitat', 'MetaSubstrate']
        meta_cols = ['month', 'day', 'Latitude', 'Longitude']
        self.missing_index = list(self.meta_data[self.meta_data[meta_cols].isnull().any(axis=1)].index)

        self.month = list(self.meta_data['month'])
        self.day = list(self.meta_data['day'])
        self.lon = list(self.meta_data['Longitude'])
        self.lat = list(self.meta_data['Latitude'])

        # if not os.path.exists('./cache'):
        #     raise '先用clip进行metadata预处理'
        #
        # self.country = torch.load(f'./cache/{split}/countryCode.pth')
        # self.substrate = torch.load(f'./cache/{split}/Substrate.pth')
        # self.habitat = torch.load(f'./cache/{split}/Habitat.pth')
        # self.meta_substrate = torch.load(f'./cache/{split}/MetaSubstrate.pth')

    def get_scaled_date_ratio(self, month, day):
        '''
        scale date to [-1,1]
        '''
        month = int(month)
        day = int(day)
        days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        total_days = 366

        assert day <= days[month - 1]
        sum_days = sum(days[:month - 1]) + day
        assert sum_days > 0 and sum_days <= total_days

        return (sum_days / total_days) * 2 - 1

    def encode_loc_time(self, lat, lon, time):
        # assumes inputs location and date features are in range -1 to 1
        # location is lon, lat
        loc_time = torch.tensor([lat, lon, time])
        loc_time = torch.concatenate([torch.sin(math.pi * loc_time), torch.cos(math.pi * loc_time)])
        return loc_time

    def __len__(self):
        return self.meta_data.shape[0]
        # return 1024

    def __getitem__(self, index: int):
        if index not in self.missing_index:
            month = self.month[index]
            day = self.day[index]
            scaled_date = self.get_scaled_date_ratio(month, day)
            lat = self.lat[index] / 90
            lon = self.lon[index] / 180
            loc_time = self.encode_loc_time(lat, lon, scaled_date)
            # country = self.country[index]  # 512
            # substrate = self.substrate[index]  # 512
            # habitat = self.habitat[index]  # 512
            # meta_substrate = self.meta_substrate[index]  # 512
            # meta_feat = torch.concatenate([loc_time, country, substrate, habitat, meta_substrate])  # 6+512*4
            meta_feat = loc_time
        else:
            meta_feat = torch.zeros(6 + 512 * 4, dtype=torch.float)

        image_path = str(self.data_path / self.meta_data.loc[index]['image_path'])
        img = get_img(image_path)
        label = self.labels[index]

        if self.transform:
            img = self.transform(image=img)['image']

        return (img, meta_feat), label


class FGVCDataset(Dataset):
    def __init__(self, data_path, split, transform=None):

        super().__init__()

        self.data_path = Path(data_path)
        self.transform = transform

        assert split in ['train', 'val']

        if split == 'train':
            self.data_path = self.data_path / 'train'
            self.meta_data = pd.read_csv(self.data_path / 'FungiCLEF2023_train_metadata_PRODUCTION.csv')
            self.data_path = self.data_path / 'DF20'

        else:
            self.data_path = self.data_path / 'val'
            self.meta_data = pd.read_csv(self.data_path / 'FungiCLEF2023_val_metadata_PRODUCTION.csv')
            self.data_path = self.data_path / 'DF21'

        self.labels = self.meta_data['class_id'].values

    def __len__(self):
        return self.meta_data.shape[0]

    def __getitem__(self, index: int):
        # get labels

        image_path = str(self.data_path / self.meta_data.loc[index]['image_path'])
        img = get_img(image_path)
        label = self.labels[index]

        if self.transform:
            img = self.transform(image=img)['image']

        return img, label


if __name__ == '__main__':
    dataset = FGVCDataset('/data/dataset/fungi2024', split='train')
    labels = dataset.labels
    print(set(labels))
    # df = pd.read_csv(r'/data/dataset/fungi2024/train/FungiCLEF2023_train_metadata_PRODUCTION.csv')
    # print(set(df['class_id']))
    # print(set(a))
