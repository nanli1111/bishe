from pathlib import Path
import sys
import os
# 添加父目录到Python路径
# 获取项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split


#加载数据集
class Dataset(Dataset):
    def __init__(self, start_samples, end_samples, seq_len=48, n_channels=2, n_classes=4):
        self.x = np.load(r'data\rayleigh_data\clean_waveforms.npy')
        self.y = np.load(r'data\rayleigh_data\true_h.npy')
        self.z = np.load(r'data\rayleigh_data\impaired_waveforms.npy')
        self.x = self.x[start_samples:end_samples]
        self.y = self.y[start_samples:end_samples]
        self.z = self.z[start_samples:end_samples]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

def get_QPSKdataloader(start = 0, end = 400000, batch_size=64, shuffle = True ):

    train_data = Dataset(start, end)

    train_loader = DataLoader(train_data, batch_size=64, shuffle = shuffle)
    return train_loader

def get_signal_shape():
    return (2, 48)  # 2个通道，长度为48的信号
