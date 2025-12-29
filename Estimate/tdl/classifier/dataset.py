import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import os

# --- 配置你的数据路径 ---
# 请修改为你实际存放 .npy 文件的文件夹路径
DATA_DIR = r'F:\LJN\bishe\bishe\Estimate\data\tdl_data'

class QPSKDataset(Dataset):
    def __init__(self, start_samples, end_samples):
        # 1. 加载数据
        # 这里的 mmap_mode='r' 可以在内存不足时只读取部分数据，如果内存够大可以去掉
        self.clean = np.load(os.path.join(DATA_DIR, 'clean_waveforms.npy'))
        self.impaired = np.load(os.path.join(DATA_DIR, 'impaired_waveforms.npy'))
        
        # 注意文件名：请确认文件夹里是 'estimate_h.npy' 还是 'estimated_h.npy'
        # 这里假设名为 'estimated_h.npy'
        h_path = os.path.join(DATA_DIR, 'estimated_h.npy')
        if not os.path.exists(h_path):
            h_path = os.path.join(DATA_DIR, 'estimate_h.npy') # 尝试另一个名字
        self.h = np.load(h_path)
        
        # 加载标签
        self.labels = np.load(os.path.join(DATA_DIR, 'labels.npy'))

        # 2. 处理 H 的维度 (保持你的逻辑)
        # 目标: 将 H 扩展为 (N, 2, 240) 以匹配波形维度
        L = self.impaired.shape[2] 
        if self.h.ndim == 2:
            # (N, Features) -> (N, Features, 1)
            self.h = self.h[:, :, np.newaxis]
            # (N, Features, 1) -> (N, Features, L)
            self.h = np.repeat(self.h, L, axis=-1)
        
        # 3. 切片 (选择训练集或测试集范围)
        self.clean = self.clean[start_samples:end_samples]
        self.impaired = self.impaired[start_samples:end_samples]
        self.h = self.h[start_samples:end_samples]
        self.labels = self.labels[start_samples:end_samples]
        
        # 4. 转 Tensor
        self.clean = torch.FloatTensor(self.clean)
        self.impaired = torch.FloatTensor(self.impaired)
        self.h = torch.FloatTensor(self.h)
        self.labels = torch.LongTensor(self.labels)

    def __len__(self):
        return len(self.impaired)

    def __getitem__(self, idx):
        # 返回: 含噪波形, 信道, 标签
        return self.impaired[idx], self.h[idx], self.labels[idx]

def get_dataloader(start, end, batch_size, shuffle=False):
    dataset = QPSKDataset(start, end)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return loader