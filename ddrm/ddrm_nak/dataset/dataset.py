import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split


#加载数据集
class Dataset(Dataset):
    def __init__(self, start_samples, end_samples, seq_len=48, n_channels=2, n_classes=4):
        self.x = np.load(r'E:\LJN\bishe\ddrm_nak\dataset_uav_h_error\clean_waveform.npy')
        self.y = np.load(r'E:\LJN\bishe\ddrm_nak\dataset_uav_h_error\impaired_waveforms.npy')
        self.h = np.load(r'E:\LJN\bishe\ddrm_nak\dataset_uav_h_error\true_h.npy') 
        self.h = np.repeat(self.h[:, :, np.newaxis], 48, axis=2)
        self.x = self.x[start_samples:end_samples]
        self.y = self.y[start_samples:end_samples]
        self.h = self.h[start_samples:end_samples] 

        # 拼接 x 和 h
        self.train_input = np.concatenate([self.x, self.h], axis=1)  # 修改为 np.concatenate
        self.test_input = np.concatenate([self.y, self.h], axis=1)  # 修改为 np.concatenate

    def __len__(self):
        return len(self.train_input)

    def __getitem__(self, idx):
        # 返回拼接后的输入 x 和 h
        return self.train_input[idx], self.test_input[idx]


def get_QPSKdataloader(start = 0, end = 100000, batch_size=64, shuffle = True ):

    train_data = Dataset(start, end)

    train_loader = DataLoader(train_data, batch_size=64, shuffle = shuffle)
    return train_loader

def get_signal_shape():
    return (4, 48)  # 4个通道，长度为48的信号
