import os
import torch
import numpy as np
import math
from scipy.special import erfc
from torch import optim
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.utils import save_image

# 自定义模块
from model.unet import UNet, build_network
from ddrm_core import DDRM
from dataset.dataset import get_test_QPSKdataloader, QPSKDataset 
from test_fig import add_awgn_noise_np, add_awgn_noise_torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置支持中文的字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 选择一个支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 测试误码率

#模型去噪


# ============ 生成 RRC 滤波器 ============
def generate_rrc_filter(sps, num_taps, alpha):
    """生成单位能量的RRC滤波器"""
    # (此函数无需修改)
    t = np.arange(num_taps) - (num_taps - 1) / 2; t /= sps
    h = np.zeros_like(t); beta = alpha
    for i, ti in enumerate(t):
        if ti == 0: h[i] = 1.0 - beta + (4 * beta / np.pi)
        elif abs(abs(4 * beta * ti) - 1.0) < 1e-9: h[i] = (beta / np.sqrt(2)) * (((1 + 2 / np.pi) * np.sin(np.pi / (4 * beta))) + ((1 - 2 / np.pi) * np.cos(np.pi / (4 * beta))))
        else: h[i] = (np.sin(np.pi * ti * (1 - beta)) + 4 * beta * ti * np.cos(np.pi * ti * (1 + beta))) / (np.pi * ti * (1 - (4 * beta * ti)**2))
    return h / np.sqrt(np.sum(h**2))


# 匹配滤波+下采样
def matched_filter(signal, rrc_filter):
    
    filter_len = len(rrc_filter) 
    
    # 匹配滤波
    number, _, wave_len = signal.x.shape
    filtered_signal = np.zeros((number, 2, wave_len + filter_len - 1))
    for i in range(number):
        filtered_signal[i, 0, :] = np.convolve(signal.y[i, 0, :], rrc_filter, mode='full')
        filtered_signal[i, 1, :] = np.convolve(signal.y[i, 1, :], rrc_filter, mode='full')
    # 2. 采样
    sampling_idx = (wave_len // 2) + ((filter_len - 1) // 2)
    sampled_iq = filtered_signal[:, :, sampling_idx]
    sampled_complex = sampled_iq[:, 0] + 1j * sampled_iq[:, 1]
    return sampled_complex

# ============ MMSE 均衡器 ============
def mmse_equalization(received_signal, channel_estimates, snr_db):
    """MMSE 均衡"""

    channel_estimates_iq = channel_estimates[:,0] + 1j * channel_estimates[:,1]
    snr_linear = 10 ** (snr_db / 10)
    w = np.conj(channel_estimates_iq) / (np.abs(channel_estimates_iq) ** 2 + 1 / snr_linear)
    return received_signal * w


# 硬判决
def decision_making(downsampled_signal, threshold=0):

    real_part = np.real(downsampled_signal)
    imag_part = np.imag(downsampled_signal)

    decision = np.zeros((len(downsampled_signal),2), dtype=int)
    #print(real_part[:10])
    #print(imag_part[:10])
    for i in range(len(downsampled_signal)):
        if (real_part[i] > threshold) and (imag_part[i] > threshold):
            decision[i,0] = 0 
            decision[i,1] = 0
        elif (real_part[i] < threshold) and (imag_part[i] > threshold):
            decision[i,0] = 0 
            decision[i,1] = 1
        elif (real_part[i] < threshold) and (imag_part[i] < threshold):
            decision[i,0] = 1 
            decision[i,1] = 1
        elif (real_part[i] > threshold) and (imag_part[i] < threshold):
            decision[i,0] = 1
            decision[i,1] = 0
    #print(decision[:10])
    return decision

# 计算误码率
def calculate_ber(original_labels, predicted_labels):
    
    predicted_labels = predicted_labels.astype(int)

    
    #print(original_labels[:10])
    #print(predicted_labels[:10])
    error_num_i = 0
    error_num_q = 0
    error_num = 0
    
    error_num_i = np.sum(original_labels[:, 0] != predicted_labels[:, 0])
    error_num_q = np.sum(original_labels[:, 1] != predicted_labels[:, 1])

    error_num = error_num_i + error_num_q

    #print(error_num_i)
    #print(error_num_q)
    ber = error_num / (len(original_labels)*2)
    print(ber)
    return ber

# 接收器
def matched_filter_decision(labels, snr_db, SAMPLES_PER_SYMBOL=16):
    rrc_filter = generate_rrc_filter(sps = 16, num_taps = 129, alpha = 0.25)
    # 数据加载
    signal = QPSKDataset(0, 500000)

    # 添加噪声
    signal.y = add_awgn_noise_np(signal.y, snr_db)
    
    #  匹配滤波+下采样
    downsampled_baseline_signal = matched_filter(signal, rrc_filter) 

    # MMSE均衡
    channel_estimates = signal.z
    equalized_signal = mmse_equalization(downsampled_baseline_signal, channel_estimates, snr_db)

    # 硬判决
    predicted_baseline_labels = decision_making(equalized_signal)

    # 确保 labels 和 predicted_labels 都是扁平化的一维数组
    baseline_ber = calculate_ber(labels, predicted_baseline_labels)
    return baseline_ber

# 绘制BER曲线并保存图像
def plot_ber_curve( baseline_bers, snr_range, save_path='ber_result/ber_curve.png'):
    plt.figure(figsize=(12, 8))


    # 绘制baseline
    plt.semilogy(snr_range, baseline_bers, 'v-', label='Baseline')

    # 设置横轴仅显示偶数刻度
    xticks = np.arange(np.ceil(min(snr_range) / 2) * 2, np.floor(max(snr_range) / 2) * 2 + 2, 2)
    plt.xticks(xticks)

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Bit Error Rate vs. SNR (QPSK)')
    plt.legend()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    plt.savefig(save_path)
    plt.close()




if __name__ == "__main__":
    n_steps = 30  # 扩散步数
    #标签数据
    label = np.load(r'F:\LJN\bishe\bishe\data\rayleigh_data\labels.npy')  

    label_data = label[0:500000]

    # 创建一个空的数组label_data_IQ，形状为(20000, 2)
    label_data_IQ = np.zeros((len(label_data), 2), dtype=int)
    # 遍历label_data，根据每个标签值更新label_data_IQ
    for i in range(len(label_data)):
        if label_data[i] == 0:
            label_data_IQ[i][0] = 0
            label_data_IQ[i][1] = 0
        elif label_data[i] == 1:
            label_data_IQ[i][0] = 0
            label_data_IQ[i][1] = 1
        elif label_data[i] == 2:
            label_data_IQ[i][0] = 1
            label_data_IQ[i][1] = 1
        elif label_data[i] == 3:
            label_data_IQ[i][0] = 1
            label_data_IQ[i][1] = 0

    baseline_bers = []
    snr_range = np.arange(2, 10, 1)
    for snr_db in snr_range:
        print(f"当前SNR: {snr_db} dB")
        baseline_ber =matched_filter_decision(label_data_IQ, snr_db - 10*math.log(16,10), SAMPLES_PER_SYMBOL=16)

        baseline_bers.append(baseline_ber)

    #绘图
     # 绘制并保存BER曲线
    plot_ber_curve(baseline_bers, snr_range, save_path=f'ddrm/ddrm_rayleigh/ber_result/baseline.png')

