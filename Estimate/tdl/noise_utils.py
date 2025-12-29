import numpy as np
import torch

def add_awgn_noise_np(clean_data, EbN0_db):
    """
    Numpy 版本加噪
    """
    noisy_data = np.zeros(clean_data.shape)
    # 计算信号功率 (Batch, 2, L) -> 全局标量
    signal_power = np.mean((np.abs(clean_data[:,0,:]) ** 2) + (np.abs(clean_data[:,1,:]) ** 2))
    
    EbN0_linear = 10**(EbN0_db/10)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0/2)
    
    # 生成I/Q两路独立高斯噪声
    noise_I = noise_std * np.random.randn(*clean_data[:,0,:].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:,1,:].shape)
    
    # 加噪
    noisy_data[:,0,:] = clean_data[:,0,:] + noise_I
    noisy_data[:,1,:] = clean_data[:,1,:] + noise_Q

    return noisy_data

def add_awgn_noise_torch(clean_data, EbN0_db):
    """
    PyTorch 版本加噪 (支持 GPU)
    输入: clean_data (Batch, 2, Length)
    输入: EbN0_db (float)
    """
    # 确保不修改原数据
    noisy_data = clean_data.clone()
    
    # 计算 Batch 平均功率
    # 注意：这里假设整个 Batch 的信号幅度水平一致，对于 QPSK 符号这是合理的
    signal_power = torch.mean((torch.abs(clean_data[:, 0, :]) ** 2) + (torch.abs(clean_data[:, 1, :]) ** 2))
    
    EbN0_linear = 10 ** (EbN0_db / 10.0)
    N0 = signal_power / EbN0_linear
    noise_std = torch.sqrt(N0 / 2.0)

    # 生成噪声
    noise_I = noise_std * torch.randn_like(clean_data[:, 0, :])
    noise_Q = noise_std * torch.randn_like(clean_data[:, 1, :])
    
    # 叠加
    noisy_data[:, 0, :] = clean_data[:, 0, :] + noise_I
    noisy_data[:, 1, :] = clean_data[:, 1, :] + noise_Q

    return noisy_data