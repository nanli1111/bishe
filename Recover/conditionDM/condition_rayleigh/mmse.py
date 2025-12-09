import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset.dataset import QPSKDataset
from test_fig import add_awgn_noise_np

# 全局绘图配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_rrc_filter(sps, num_taps, alpha):
    """生成RRC滤波器系数"""
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

def matched_filter(signal_y, rrc_filter):
    """匹配滤波与下采样"""
    N, _, L = signal_y.shape
    sampling_idx = len(rrc_filter) // 2 
    sampled = np.zeros(N, dtype=complex)
    start = (L // 2)
    end = start + 16 

    for i in range(N):
        i_conv = np.convolve(signal_y[i, 0, start:end], rrc_filter, mode='full')
        q_conv = np.convolve(signal_y[i, 1, start:end], rrc_filter, mode='full')
        sampled[i] = i_conv[sampling_idx] + 1j * q_conv[sampling_idx]
    return sampled

def mmse_equalization(rx_complex, h_est, snr_db):
    """MMSE均衡"""
    h_c = h_est[:, 0] + 1j * h_est[:, 1]
    snr_linear = 10 ** (snr_db / 10.0)
    w = np.conj(h_c) / (np.abs(h_c)**2 + 1.0/snr_linear)
    return rx_complex * w

def decision_making(symbols):
    """硬判决 (Gray Mapping)"""
    r, i = np.real(symbols), np.imag(symbols)
    bits = np.zeros((len(symbols), 2), dtype=int)
    # 0:(+,+), 1:(-,+), 2:(-,-), 3:(+,-)
    bits[(r > 0) & (i > 0)] = (0, 0)
    bits[(r < 0) & (i > 0)] = (0, 1)
    bits[(r < 0) & (i < 0)] = (1, 1)
    bits[(r > 0) & (i < 0)] = (1, 0)
    return bits

def calculate_ber(true_bits, pred_bits):
    return np.sum(true_bits != pred_bits) / true_bits.size

def run_simulation(y_clean, h_est, true_bits, snr_db, rrc_filter):
    # 1. 加噪 (符号SNR -> 采样SNR)
    snr_sample = snr_db - 10 * math.log10(16)
    y_noisy = add_awgn_noise_np(y_clean, snr_sample)
    
    # 2. 匹配滤波
    y_filtered = matched_filter(y_noisy, rrc_filter)
    
    # 3. MMSE均衡 & 判决
    y_eq = mmse_equalization(y_filtered, h_est, snr_db)
    pred_bits = decision_making(y_eq)
    
    return calculate_ber(true_bits, pred_bits)

if __name__ == "__main__":
    # 配置
    start, end = 0, 100000
    label_path = r'F:\LJN\bishe\bishe\data\rayleigh_data\labels.npy'
    save_dir = 'conditionDM/condition_rayleigh/ber_result'
    
    # 1. 预加载数据 (移出循环以提升性能)
    print("Loading data...")
    dataset = QPSKDataset(start, end)
    y_all, z_all = dataset.y, dataset.z
    
    full_labels = np.load(label_path)
    map_dict = {0:(0,0), 1:(0,1), 2:(1,1), 3:(1,0)}
    true_bits = np.array([map_dict[int(l)] for l in full_labels[start:end]], dtype=int)

    # 2. 准备滤波器
    rrc = generate_rrc_filter(sps=16, num_taps=129, alpha=0.25)
    
    # 3. SNR 扫描
    snr_range = np.arange(2, 18, 1)
    bers = []
    
    print("Running simulation...")
    for snr in tqdm(snr_range):
        ber = run_simulation(y_all, z_all, true_bits, snr, rrc)
        bers.append(ber)

    # 4. 保存与绘图
    os.makedirs(save_dir, exist_ok=True)
    
    # 保存CSV
    with open(os.path.join(save_dir, 'baseline_ber.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr_db', 'baseline_ber'])
        writer.writerows(zip(snr_range, bers))
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, bers, 'v-', label='Baseline (MMSE)')
    plt.xticks(np.arange(min(snr_range), max(snr_range)+1, 2))
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Baseline Performance (Rayleigh)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'baseline.png'))
    print(f"Done. Results saved to {save_dir}")