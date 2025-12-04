import os
import math
import csv
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from dataset.dataset import QPSKDataset
from test_fig_x_pre import add_awgn_noise_np

# 全局绘图配置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
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

def mmse_pre_equalization_waveform(rx_signal, h_est_seq, snr_db):
    """采样级 MMSE 均衡"""
    y_complex = rx_signal[:, 0, :] + 1j * rx_signal[:, 1, :] 
    h_complex = h_est_seq[:, 0, :] + 1j * h_est_seq[:, 1, :]

    snr_linear = 10 ** (snr_db / 10.0)
    numerator = np.conj(h_complex)
    denominator = (np.abs(h_complex) ** 2 + 1.0 / snr_linear)
    w = numerator / denominator

    r_complex = y_complex * w
    
    r_i = np.real(r_complex)
    r_q = np.imag(r_complex)
    return np.stack([r_i, r_q], axis=1)

def matched_filter_and_downsample(signal_y, rrc_filter):
    """
    对均衡后的波形进行匹配滤波并直接下采样
    signal_y: [N, 2, L]
    return: [N] complex symbols
    """
    N, _, L = signal_y.shape
    filter_len = len(rrc_filter)
    
    # 假设波形中心对齐，取中间点作为最佳采样点
    sampling_idx = filter_len // 2 
    
    sampled_symbols = np.zeros(N, dtype=complex)
    start = (L // 2)
    end = start + 16 
    for i in range(N):
        # I路卷积
        conv_i = np.convolve(signal_y[i, 0, start: end], rrc_filter, mode='full')
        # Q路卷积
        conv_q = np.convolve(signal_y[i, 1, start: end], rrc_filter, mode='full')
        
        # 直接取出采样点
        i_val = conv_i[sampling_idx]
        q_val = conv_q[sampling_idx]
        
        sampled_symbols[i] = i_val + 1j * q_val
        
    return sampled_symbols

def decision_making(symbols):
    """硬判决"""
    r, i = np.real(symbols), np.imag(symbols)
    bits = np.zeros((len(symbols), 2), dtype=int)
    bits[(r > 0) & (i > 0)] = (0, 0)
    bits[(r < 0) & (i > 0)] = (0, 1)
    bits[(r < 0) & (i < 0)] = (1, 1)
    bits[(r > 0) & (i < 0)] = (1, 0)
    return bits

def calculate_ber(true_bits, pred_bits):
    return np.sum(true_bits != pred_bits) / true_bits.size

def run_simulation(y_clean, y_faded, h_est, true_bits, snr_db, rrc_filter):
    # 1. 加噪
    sps = 16
    snr_sample = snr_db - 10 * math.log10(sps)
    y_noisy = add_awgn_noise_np(y_faded, snr_sample)
    
    # 2. 扩展 h
    if h_est.ndim == 2:
        h_est = h_est[:, :, np.newaxis]
        h_est = np.repeat(h_est, y_clean.shape[2], axis=-1)
    
    # 3. MMSE 均衡
    y_eq = mmse_pre_equalization_waveform(y_noisy, h_est, snr_sample)
    
    # 4. 匹配滤波 & 下采样 (合并操作)
    sampled_complex = matched_filter_and_downsample(y_eq, rrc_filter)
    
    # 5. 判决
    pred_bits = decision_making(sampled_complex)
    
    return calculate_ber(true_bits, pred_bits)

if __name__ == "__main__":
    # 配置
    start, end = 0, 100000
    label_path = r'F:\LJN\bishe\bishe\data\rayleigh_data\labels.npy'
    save_dir = 'IS2B/IS2B_rayleigh/ber_result'
    
    print("Loading data...")
    dataset = QPSKDataset(start, end)
    
    y_clean = dataset.x
    y_faded = dataset.y
    h_all = dataset.z
    
    full_labels = np.load(label_path)
    map_dict = {0:(0,0), 1:(0,1), 2:(1,1), 3:(1,0)}
    true_bits = np.array([map_dict[int(l)] for l in full_labels[start:end]], dtype=int)

    rrc = generate_rrc_filter(sps=16, num_taps=129, alpha=0.25)
    
    snr_range = np.arange(2, 18, 1)
    bers = []
    
    print("Running simulation...")
    for snr in tqdm(snr_range):
        ber = run_simulation(
            y_clean=y_clean,
            y_faded=y_faded,
            h_est=h_all, 
            true_bits=true_bits, 
            snr_db=snr, 
            rrc_filter=rrc
        )
        bers.append(ber)

    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, 'baseline_ber.csv'), 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr_db', 'baseline_ber'])
        writer.writerows(zip(snr_range, bers))
    
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, bers, 'v-', label='Baseline (MMSE)')
    plt.grid(True, which='both', linestyle='--', alpha=0.6)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Baseline Performance (Rayleigh)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'baseline.png'))
    print(f"Done. Results saved to {save_dir}")