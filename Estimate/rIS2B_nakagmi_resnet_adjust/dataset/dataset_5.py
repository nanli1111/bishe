import torch
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader, random_split

# 数据集定义
class QPSKDataset(Dataset):
    def __init__(self, start_samples, end_samples, seq_len=48, n_channels=2, n_classes=4):
        # 假设数据存储在文件中，读取数据
        self.x = np.load(r'F:\LJN\bishe\bishe\Estimate\data\nakagmi_data_5\clean_waveforms.npy')
        self.y= np.load(r'F:\LJN\bishe\bishe\Estimate\data\nakagmi_data_5\impaired_waveforms.npy')
        self.z = np.load(r'F:\LJN\bishe\bishe\Estimate\data\nakagmi_data_5\estimated_h.npy')
        L = self.x.shape[2]
        if self.z.ndim == 2:
            self.z = self.z[:, :, np.newaxis]
            self.z = np.repeat(self.z, L, axis=-1)
        else:
            self.z = self.z
        self.x = self.x[start_samples:end_samples]  # 切割指定的样本范围
        self.y = self.y[start_samples:end_samples]  # 切割指定的样本范围
        self.z = self.z[start_samples:end_samples]  # 切割指定的样本范围
        
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx], self.z[idx]

# 获取训练集和验证集的加载器
def get_train_QPSKdataloader(start=0, end=100000, batch_size=64, shuffle=True, val_split=0.2):

    # 加载数据集
    dataset = QPSKDataset(start, end)

    # 计算训练集和验证集的大小
    total_size = len(dataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size

    # 使用random_split划分数据集
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 创建DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

# 获取训练集和验证集的加载器
def get_test_QPSKdataloader(start=0, end=100000, batch_size=64, shuffle=False):

    # 加载数据集    
    test_dataset = QPSKDataset(start, end)

    # 计算训练集和验证集的大小
    test_size = len(test_dataset)

    # 创建DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)

    return test_loader

# 获取信号的形状
def get_signal_shape():
    return (2, 80)  # 2个通道，长度为48的信号
