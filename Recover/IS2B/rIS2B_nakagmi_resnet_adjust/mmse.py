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
    """生成单位能量的 RRC 滤波器系数"""
    t = (np.arange(num_taps) - (num_taps - 1) / 2) / sps
    h = np.zeros_like(t)
    for i, ti in enumerate(t):
        if ti == 0:
            h[i] = 1.0 - alpha + (4 * alpha / np.pi)
        elif abs(abs(4 * alpha * ti) - 1.0) < 1e-9:
            h[i] = (alpha / np.sqrt(2)) * ((1 + 2/np.pi)*np.sin(np.pi/(4*alpha)) + (1 - 2/np.pi)*np.cos(np.pi/(4*alpha)))
        else:
            h[i] = (np.sin(np.pi*ti*(1-alpha)) + 4*alpha*ti*np.cos(np.pi*ti*(1+alpha))) / (np.pi*ti*(1-(4*alpha*ti)**2))
    
    # [关键] 保持与数据集生成器一致：单位能量归一化
    return h / np.sqrt(np.sum(h**2))

def mmse_pre_equalization_waveform(rx_signal, h_est_seq, snr_db):
    """采样级 MMSE 均衡"""
    y_complex = rx_signal[:, 0, :] + 1j * rx_signal[:, 1, :] 
    h_complex = h_est_seq[:, 0, :] + 1j * h_est_seq[:, 1, :]

    snr_linear = 10 ** (snr_db / 10.0)
    # MMSE Weight: W = h* / (|h|^2 + 1/SNR)
    # 这个公式是通用的，前提是 SNR 定义为 Ps/Pn
    numerator = np.conj(h_complex)
    denominator = (np.abs(h_complex) ** 2 + 1.0 / snr_linear)
    w = numerator / denominator

    r_complex = y_complex * w
    
    r_i = np.real(r_complex)
    r_q = np.imag(r_complex)
    return np.stack([r_i, r_q], axis=1)

def matched_filter_full(signal_y, rrc_filter):
    """仅执行匹配滤波卷积"""
    N, _, L = signal_y.shape
    filter_len = len(rrc_filter)
    start_idx = (filter_len - 1) // 2
    end_idx = start_idx + L
    
    filtered_output = np.zeros((N, 2, L), dtype=float)
    
    for i in range(N):
        conv_i = np.convolve(signal_y[i, 0], rrc_filter, mode='full')
        conv_q = np.convolve(signal_y[i, 1], rrc_filter, mode='full')
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

# ================= 新增：星座图可视化函数 =================
def plot_constellation(symbols_complex, snr_db, save_dir, num_points=2048):
    """
    绘制星座图 (适配新数据集幅度)
    """
    if len(symbols_complex) > num_points:
        indices = np.random.choice(len(symbols_complex), num_points, replace=False)
        pts = symbols_complex[indices]
    else:
        pts = symbols_complex

    plt.figure(figsize=(6, 6))
    plt.scatter(pts.real, pts.imag, s=5, c='blue', alpha=0.3, label='Received (Equalized)')
    
    # [修改] 适配新数据集：标准参考点改为 ±1 ±j
    ref_pts = [1+1j, -1+1j, -1-1j, 1-1j]
    ref_x = [p.real for p in ref_pts]
    ref_y = [p.imag for p in ref_pts]
    plt.scatter(ref_x, ref_y, s=50, c='red', marker='x', label='Ideal QPSK')

    plt.title(f"Baseline Constellation (MMSE+MF) @ SNR={snr_db}dB")
    plt.xlabel("In-Phase (I)")
    plt.ylabel("Quadrature (Q)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    # [修改] 扩大显示范围，适应可能的过冲
    plt.xlim(-3.0, 3.0)
    plt.ylim(-3.0, 3.0)
    
    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, f'constellation_snr{snr_db}.png'))
    plt.close()

def run_simulation(y_clean, y_faded, h_est, true_bits, snr_db, rrc_filter, 
                   vis_indices=None, save_dir=None, draw_constellation=False):
    # 1. 加噪
    sps = 16
    snr_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)
    y_noisy = add_awgn_noise_np(y_faded, snr_sample)
    
    # 2. 扩展 h
    if h_est.ndim == 2:
        h_est = h_est[:, :, np.newaxis]
        h_est = np.repeat(h_est, y_clean.shape[2], axis=-1)
    
    # 3. MMSE 均衡
    y_eq = mmse_pre_equalization_waveform(y_noisy, h_est, snr_sample)
    
    # 4. 匹配滤波 (全长)
    y_mf_waveform = matched_filter_full(y_eq, rrc_filter)

    # 5. 下采样 (取最佳采样点)
    L = y_clean.shape[2]
    sampling_idx = L // 2 
    sampled_complex = y_mf_waveform[:, 0, sampling_idx] + 1j * y_mf_waveform[:, 1, sampling_idx]
    
    # --- 绘制星座图 (如果需要) ---
    if draw_constellation and save_dir is not None:
        vis_con_dir = os.path.join(save_dir, 'vis_constellation')
        plot_constellation(sampled_complex, snr_db, vis_con_dir)
    # ---------------------------

    # 6. 判决与BER
    pred_bits = decision_making(sampled_complex)
    return calculate_ber(true_bits, pred_bits)

if __name__ == "__main__":
    # 配置
    start, end = 400000, 500000
    label_path = r'F:\LJN\bishe\bishe\data\nakagmi_data\labels.npy'
    save_dir = 'IS2B/rIS2B_nakagmi_resnet_adjust/ber_results'
    # 注意修正路径分隔符
    vis_dir = os.path.join(save_dir, 'vis_baseline')
    
    print("Loading data...")
    dataset = QPSKDataset(start, end)
    y_clean, y_faded, h_all = dataset.x, dataset.y, dataset.z
    
    full_labels = np.load(label_path)
    map_dict = {0:(0,0), 1:(0,1), 2:(1,1), 3:(1,0)}
    true_bits = np.array([map_dict[int(l)] for l in full_labels[start:end]], dtype=int)

    rrc = generate_rrc_filter(sps=16, num_taps=129, alpha=0.25)
    
    snr_range = np.arange(0, 19, 1)
    bers = []
    
    print("Running simulation...")
    for snr in tqdm(snr_range):
        # 仅在特定的 SNR 点绘制星座图
        need_constellation = (snr in [0, 5, 10, 15, 18])
        
        ber = run_simulation(
            y_clean=y_clean,
            y_faded=y_faded,
            h_est=h_all, 
            true_bits=true_bits, 
            snr_db=snr, 
            rrc_filter=rrc,
            vis_indices=None,
            save_dir=save_dir,
            draw_constellation=need_constellation
        )
        bers.append(ber)

    # 保存数据和BER图
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
    plt.title('Baseline Performance (Adjusted for New Dataset)')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'baseline.png'))
    print(f"Done. Results saved to {save_dir}")