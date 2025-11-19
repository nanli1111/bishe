import os
import sys
import math
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
from scipy.signal import fftconvolve
from torch.utils.data import Dataset, DataLoader, random_split

# ============ 环境与路径设置 ============
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from data.rayleigh_data.rayleigh_data_gengerate import UAVBaseDatasetGenerator
from dataset.dataset import QPSKDataset

# 中文显示设置
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False


# ============ AWGN 加噪 ============
def add_awgn_noise_np(clean_data, EbN0_db):
    """为 I/Q 信号添加 AWGN 噪声"""
    signal_power = np.mean(np.abs(clean_data[:, 0, :]) ** 2 + np.abs(clean_data[:, 1, :]) ** 2)
    EbN0_linear = 10 ** (EbN0_db / 10)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0 / 2)

    noisy_data = np.zeros_like(clean_data)
    noise_I = noise_std * np.random.randn(*clean_data[:, 0, :].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:, 1, :].shape)

    noisy_data[:, 0, :] = clean_data[:, 0, :] + noise_I
    noisy_data[:, 1, :] = clean_data[:, 1, :] + noise_Q

    return noisy_data


# ============ 匹配滤波 + 均衡 ============
def data_pre_with_mmse(snrDB, use_mmse=True):
    """匹配滤波并可选进行 MMSE 均衡"""
    generator_rrc = UAVBaseDatasetGenerator()
    rrc_filter = generator_rrc.create_rrc_filter()
    SAMPLES_PER_SYMBOL = generator_rrc.SAMPLES_PER_SYMBOL
    filter_delay = len(rrc_filter) // 2

    # 加载测试数据
    test_data = QPSKDataset(0, 500000)
    test_data.y = add_awgn_noise_np(test_data.y, snrDB - 10 * math.log10(16))

    # 构造信号与信道估计
    baseline_signal = np.zeros((len(test_data.y) * 16,), dtype=np.complex64)
    channel_estimates = np.zeros(len(test_data.y), dtype=np.complex64)

    for i in range(len(test_data.y)):
        baseline_signal[16 * i:16 * i + 16] = test_data.y[i, 0, 24:40] + 1j * test_data.y[i, 1, 24:40]
        if hasattr(test_data, 'z') and test_data.z is not None:
            channel_estimates[i] = test_data.z[i, 0] + 1j * test_data.z[i, 1]

    # 匹配滤波
    filtered_signal = np.convolve(baseline_signal, rrc_filter, mode='full')
    filtered_signal = filtered_signal[filter_delay:-filter_delay]
    downsampled_signal = filtered_signal[::SAMPLES_PER_SYMBOL]

    if use_mmse:
        equalized_signal = mmse_equalization(downsampled_signal, channel_estimates, snrDB)
        return equalized_signal
    else:
        return downsampled_signal


# ============ MMSE 均衡器 ============
def mmse_equalization(received_signal, channel_estimates, snr_db):
    """MMSE 均衡"""
    snr_linear = 10 ** (snr_db / 10)
    w = np.conj(channel_estimates) / (np.abs(channel_estimates) ** 2 + 1 / snr_linear)
    return received_signal * w


# ============ 判决与 BER ============
def decision_making(signal, threshold=0):
    """QPSK 硬判决"""
    real_part, imag_part = np.real(signal), np.imag(signal)
    decision = np.zeros((len(signal), 2), dtype=int)
    decision[(real_part > threshold) & (imag_part > threshold)] = [0, 0]
    decision[(real_part < threshold) & (imag_part > threshold)] = [0, 1]
    decision[(real_part < threshold) & (imag_part < threshold)] = [1, 1]
    decision[(real_part > threshold) & (imag_part < threshold)] = [1, 0]
    return decision


def calculate_ber(true_labels, pred_labels):
    """计算误码率"""
    errors = np.sum(true_labels != pred_labels)
    ber = errors / true_labels.size
    print(f"BER: {ber:.6e}")
    return ber


# ============ 接收机流程 ============
def matched_filter_decision(labels, snr_db, use_mmse=True, SAMPLES_PER_SYMBOL=16):
    """执行接收与检测"""
    if use_mmse:
        equalized_signal = data_pre_with_mmse(snr_db, use_mmse=True)
    else:
        equalized_signal = data_pre_with_mmse(snr_db, use_mmse=False)

    predicted_labels = decision_making(equalized_signal)
    ber = calculate_ber(labels, predicted_labels)
    return ber


# ============ 绘制 BER 曲线 ============
def plot_ber_curve( mmse_bers, snr_range, save_path):
    plt.figure(figsize=(10, 7))
    snr_array = np.array(snr_range)

    # 理论 Rayleigh BER
    snr_linear = 10 ** (snr_array / 10.0)
    ber_rayleigh = 0.5 * (1.0 - np.sqrt(snr_linear / (1.0 + snr_linear)))

    plt.semilogy(snr_array, ber_rayleigh, 'g-.', label='瑞利平坦衰落理论 QPSK')
    plt.semilogy(snr_array, mmse_bers, 's-', label='MMSE 均衡')
    print(mmse_bers)
    print(ber_rayleigh)
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('QPSK 在 Rayleigh 信道下的理论与仿真 BER 对比')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


# ============ 主程序入口 ============
if __name__ == "__main__":
    # 加载标签数据
    label = np.load(r'F:\LJN\bishe\bishe\data\rayleigh_data\labels.npy')
    label_data = label[:500000]

    label_data_IQ = np.zeros((len(label_data), 2), dtype=int)
    mapping = {0: [0, 0], 1: [0, 1], 2: [1, 1], 3: [1, 0]}
    for i, val in enumerate(label_data):
        label_data_IQ[i] = mapping[val]

    baseline_bers, mmse_bers = [], []
    snr_range = np.arange(-5, 12, 1)

    for snr_db in snr_range:
        print(f"\n当前 SNR: {snr_db} dB")


        print("MMSE 均衡：")
        mmse_ber = matched_filter_decision(label_data_IQ, snr_db, use_mmse=True)
        mmse_bers.append(mmse_ber)

        print("-" * 50)

    # 绘制 BER 曲线
    save_path = r'ddrm/ddrm_nak/ber_result/ber_result_with_mmse.png'
    plot_ber_curve(mmse_bers, snr_range, save_path)

