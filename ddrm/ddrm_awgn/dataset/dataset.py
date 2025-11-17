import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split


#加载数据集
class Dataset(Dataset):
    def __init__(self, start_samples, end_samples, seq_len=48, n_channels=2, n_classes=4):
        self.x = np.load(r'F:\LJN\bishe\bishe\data\awgn_data\qpsk_clean.npy')
        self.x = self.x[start_samples:end_samples]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx]

def get_QPSKdataloader(start = 0, end = 100000, batch_size=64, shuffle = True ):

    train_data = Dataset(start, end)

    train_loader = DataLoader(train_data, batch_size=64, shuffle = shuffle)
    return train_loader

def get_signal_shape():
    return (2, 48)  # 2个通道，长度为48的信号
