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

def matched_filter_full(signal_y, rrc_filter):
    """
    仅执行匹配滤波卷积，不进行下采样。
    返回完整的卷积后波形，用于可视化。
    signal_y: [N, 2, L]
    return: [N, 2, L_conv] (长度会变长)
    """
    N, _, L = signal_y.shape
    # 卷积后长度 = L + len(filter) - 1
    # 我们通常只关心和原信号对齐的中心部分
    filter_len = len(rrc_filter)
    output_len = L + filter_len - 1
    start_idx = (filter_len - 1) // 2
    end_idx = start_idx + L
    
    filtered_output = np.zeros((N, 2, L), dtype=float) # 截取回原长度
    
    for i in range(N):
        # I路
        conv_i = np.convolve(signal_y[i, 0], rrc_filter, mode='full')
        # Q路
        conv_q = np.convolve(signal_y[i, 1], rrc_filter, mode='full')
        
        # 截取中心部分以对齐原信号
        filtered_output[i, 0] = conv_i[start_idx:end_idx]
        filtered_output[i, 1] = conv_q[start_idx:end_idx]
        
    return filtered_output

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

# ================= 修改后的可视化函数 =================
def visualize_signals(y_clean, y_noisy, y_eq, y_mf, idx, snr_db, save_dir):
    """
    画出四个波形的对比：
    1. 原始干净信号 (Clean)
    2. 接收信号 (Faded+Noisy)
    3. MMSE均衡后 (Equalized)
    4. 匹配滤波后 (Matched Filtered) -> 应该是光滑的
    """
    t = np.arange(y_clean.shape[1])
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # I路
    axes[0].plot(t, y_clean[0], 'k-', linewidth=2, alpha=0.3, label='原始干净 (Clean)')
    axes[0].plot(t, y_noisy[0], 'r--', linewidth=1, alpha=0.4, label='接收含噪 (Noisy)')
    axes[0].plot(t, y_mf[0], 'b-', linewidth=2, label='匹配滤波后 (MF Output)') # 重点看这个
    axes[0].set_title(f"I路波形处理流程 (Sample {idx}, SNR={snr_db}dB)")
    axes[0].legend(loc='upper right')
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Q路
    axes[1].plot(t, y_clean[1], 'k-', linewidth=2, alpha=0.3, label='原始干净 (Clean)')
    axes[1].plot(t, y_noisy[1], 'b--', linewidth=1, alpha=0.4, label='接收含噪 (Noisy)')
    axes[1].plot(t, y_mf[1], 'm-', linewidth=2, label='匹配滤波后 (MF Output)') # 重点看这个
    axes[1].set_title(f"Q路波形处理流程 (Sample {idx}, SNR={snr_db}dB)")
    axes[1].legend(loc='upper right')
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    plt.tight_layout()
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'vis_baseline_snr{snr_db}_idx{idx}.png'))
    plt.close()

def run_simulation(y_clean, y_faded, h_est, true_bits, snr_db, rrc_filter, vis_indices=None, save_dir=None):
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
    
    # 4. 匹配滤波 (全长卷积)
    # 注意：这里我们为了可视化，先算出完整的滤波后波形
    y_mf_waveform = matched_filter_full(y_eq, rrc_filter)

    # --- 可视化 ---
    if vis_indices is not None and save_dir is not None:
        for idx in vis_indices:
            visualize_signals(
                y_clean[idx], 
                y_noisy[idx], 
                y_eq[idx], 
                y_mf_waveform[idx], # 传入匹配滤波后的波形
                idx, snr_db, save_dir
            )
    # -------------

    # 5. 下采样 (从匹配滤波后的波形中取样)
    # 假设中心点是对齐的
    L = y_clean.shape[2]
    sampling_idx = L // 2 
    
    # 取出复数符号
    sampled_complex = y_mf_waveform[:, 0, sampling_idx] + 1j * y_mf_waveform[:, 1, sampling_idx]
    
    # 6. 判决
    pred_bits = decision_making(sampled_complex)
    
    return calculate_ber(true_bits, pred_bits)

if __name__ == "__main__":
    # 配置
    start, end = 400000, 500000
    label_path = r'F:\LJN\bishe\bishe\data\nakagmi_data\labels.npy'
    save_dir = 'CNN/RESNET/ber_results'
    vis_dir = os.path.join(save_dir, 'vis_baseline')
    
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
    
    vis_indices = np.random.choice(len(y_clean), 3, replace=False)
    
    print("Running simulation...")
    for snr in tqdm(snr_range):
        current_vis_indices = vis_indices if snr in [5, 10, 15] else None
        
        ber = run_simulation(
            y_clean=y_clean,
            y_faded=y_faded,
            h_est=h_all, 
            true_bits=true_bits, 
            snr_db=snr, 
            rrc_filter=rrc,
            vis_indices=current_vis_indices,
            save_dir=vis_dir
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
    plt.title('Baseline Performance (nakagmi)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'baseline.png'))
    print(f"Done. Results saved to {save_dir}")