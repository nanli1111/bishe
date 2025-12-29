import os
import numpy as np
from scipy.linalg import toeplitz

# =============================================================================
# 全局配置
# =============================================================================
CONFIG = {
    'data_dir': r'Estimate/data/tdl_data',
    # 中间数据保存路径
    'output_file': r'Estimate/tdl/baseline/results/estimation_data.npz',
    'save_dir': r'Estimate/tdl/baseline/results',
    'sps': 16,             
    'num_taps': 129,       
    'alpha': 0.25,         
    'snr_range': np.arange(-5, 26, 3), 
    'train_samples': 30000, 
    'test_samples': 20000,
    'channel_len': 15      
}

# =============================================================================
# LMMSE 估计器类
# =============================================================================
class LMMSE_Denoiser:
    def __init__(self):
        self.W = None 
        
    def fit(self, h_ls_train, h_true_train):
        y = h_ls_train   
        x = h_true_train 
        # Vector-wise LMMSE: Calculate weights for each tap position
        R_yy = np.mean(y * np.conj(y), axis=0)
        R_xy = np.mean(x * np.conj(y), axis=0)
        self.W = R_xy / (R_yy + 1e-12)
        
    def predict(self, h_ls_test):
        return self.W * h_ls_test

# =============================================================================
# 物理层函数
# =============================================================================
def add_awgn_noise_np(clean_data, EbN0_db):
    """ (Batch, 2, 240) -> (Batch, 2, 240) """
    noisy_data = np.zeros(clean_data.shape)
    signal_power = np.mean((np.abs(clean_data[:,0,:]) ** 2) + (np.abs(clean_data[:,1,:]) ** 2))
    EbN0_linear = 10**(EbN0_db/10)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0/2)
    noise_I = noise_std * np.random.randn(*clean_data[:,0,:].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:,1,:].shape)
    noisy_data[:,0,:] = clean_data[:,0,:] + noise_I
    noisy_data[:,1,:] = clean_data[:,1,:] + noise_Q
    return noisy_data, float(N0)

def generate_noisy_channel_vec(h_true_vec, noise_std):
    noise = noise_std * (np.random.randn(*h_true_vec.shape) + 1j * np.random.randn(*h_true_vec.shape))
    return h_true_vec + noise

def generate_rrc_filter(sps, num_taps, alpha):
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif abs(abs(4 * alpha * ti) - 1.0) < 1e-9:
            h[i] = (alpha / np.sqrt(2)) * ((1 + 2/np.pi)*np.sin(np.pi/(4*alpha)) + (1 - 2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            h[i] = (np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))) / (np.pi*ti*(1-(4*alpha*ti)**2))
    return h / np.sqrt(np.sum(h**2))

def apply_rrc_and_downsample(rx_data, rrc_filter, sps=16):
    batch_size = rx_data.shape[0]
    win_len_symbols = 15 
    rx_c = rx_data[:, 0, :] + 1j * rx_data[:, 1, :]
    rx_symbols = np.zeros((batch_size, win_len_symbols), dtype=complex)
    
    for i in range(batch_size):
        filtered = np.convolve(rx_c[i], rrc_filter, mode='full')
        center_idx = len(filtered) // 2
        offsets = np.arange(-(win_len_symbols//2), (win_len_symbols//2) + 1) * sps
        sample_indices = center_idx + offsets
        rx_symbols[i] = filtered[sample_indices]
    return rx_symbols

def qpsk_decision(symbols):
    preds = np.zeros(len(symbols), dtype=int)
    r, i = np.real(symbols), np.imag(symbols)
    preds[(r > 0) & (i > 0)] = 0
    preds[(r <= 0) & (i > 0)] = 1
    preds[(r <= 0) & (i <= 0)] = 2
    preds[(r > 0) & (i <= 0)] = 3
    return preds

# =============================================================================
# 数据加载器 (修改: 加载 estimated_h)
# =============================================================================
class TDLDatasetLoader:
    def __init__(self, data_dir, limit=None):
        print(f"Loading data from {data_dir}...")
        self.rx = np.load(os.path.join(data_dir, 'impaired_waveforms.npy'))
        h_vec_true = np.load(os.path.join(data_dir, 'true_h.npy'))
        h_vec_est = np.load(os.path.join(data_dir, 'estimated_h.npy')) # 新增：加载 estimated_h
        self.labels = np.load(os.path.join(data_dir, 'labels.npy'))
        
        # 处理 true_h
        if h_vec_true.ndim == 3:
            self.h_true = h_vec_true[:, 0, :] + 1j * h_vec_true[:, 1, :]
        else:
            self.h_true = h_vec_true

        # 处理 estimated_h
        if h_vec_est.ndim == 3: # 如果是 (N, 2, 15) 形式
            self.h_dataset_est = h_vec_est[:, 0, :] + 1j * h_vec_est[:, 1, :]
        elif h_vec_est.ndim == 2: # 如果已经是 (N, 15) 复数形式
            self.h_dataset_est = h_vec_est
        else:
            # 兼容处理，如果形状不符则报错或给出警告
            print(f"Warning: Unexpected estimated_h shape {h_vec_est.shape}. Attempting to adapt.")
            # 尝试根据原始代码的逻辑（只取中心点，并转复数）
            if h_vec_est.ndim == 1: # 如果是 (N,)，假设已经是中心点
                self.h_dataset_est = h_vec_est 
            elif h_vec_est.ndim == 2 and h_vec_est.shape[1] == 2: # (N, 2)，假设是 I/Q
                self.h_dataset_est = h_vec_est[:,0] + 1j * h_vec_est[:,1]
            else:
                raise ValueError(f"Unsupported estimated_h shape: {h_vec_est.shape}")


        if limit is not None:
            self.rx = self.rx[:limit]
            self.h_true = self.h_true[:limit]
            self.h_dataset_est = self.h_dataset_est[:limit] # 也要限制
            self.labels = self.labels[:limit]
            
        print(f"Dataset Ready. Rx Shape: {self.rx.shape}, H_true Shape: {self.h_true.shape}, H_dataset_est Shape: {self.h_dataset_est.shape}")

    def get_data(self, start, end):
        return self.rx[start:end], self.h_true[start:end], self.h_dataset_est[start:end], self.labels[start:end]