from pathlib import Path
import sys
import os
# 添加父目录到Python路径
# 获取项目根目录
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))
import torch
import numpy as np
import math
from scipy.special import erfc
from scipy.signal import fftconvolve
from torch.utils.data import Dataset, DataLoader, random_split
from data.rayleigh_data.rayleigh_data_gengerate import UAVBaseDatasetGenerator
import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置支持中文的字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 选择一个支持中文的字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 自定义模块
from dataset.dataset import Dataset  

# 测试误码率

# 加噪
def add_awgn_noise_np(clean_data, EbN0_db):
    noisy_data = np.zeros(clean_data.shape)
    # 计算信号功率
    signal_power = np.mean((np.abs(clean_data[:,0,:]) ** 2) + (np.abs(clean_data[:,1,:]) ** 2))
    EbN0_linear = 10**(EbN0_db/10)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0/2)
    # 生成I/Q两路独立高斯噪声
    noise_I = noise_std * np.random.randn(*clean_data[:,0,:].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:,1,:].shape)
    
    # 加噪（模拟实际信道）
    noisy_data[:,0,:] = clean_data[:,0,:] + noise_I;  # I路接收信号
    noisy_data[:,1,:] = clean_data[:,1,:] + noise_Q;  # Q路接收信号

    return noisy_data

# 数据重整_匹配滤波 + MMSE均衡
def data_pre_with_mmse(snrDB, use_mmse=True):
    generator_rrc = UAVBaseDatasetGenerator()
    rrc_filter = generator_rrc.create_rrc_filter()
    SAMPLES_PER_SYMBOL = generator_rrc.SAMPLES_PER_SYMBOL
    filter_delay = len(rrc_filter) // 2

    test_data = Dataset(0, 100000)
    test_data.x = add_awgn_noise_np(test_data.x, snrDB - 10*math.log(16,10))
    
    # 重构信号
    baseline_signal = np.zeros((len(test_data.x)*16,), dtype=np.complex64)
    channel_estimates = np.zeros(len(test_data.x), dtype=np.complex64)  # 存储信道估计

    for i in range(len(test_data.x)):
        # 重构信号
        baseline_signal[16*i:16*i+16] = test_data.x[i, 0, 24:40] + 1j * test_data.x[i, 1, 24:40]
        # 直接从test_data.y获取信道估计
        if hasattr(test_data, 'y') and test_data.y is not None:
            # 假设test_data.y的形状为 [batch_size, 2] 或 [batch_size, 2, something]
            if test_data.y.ndim == 2:  # [batch_size, 2]
                channel_estimates[i] = test_data.y[i, 0] + 1j * test_data.y[i, 1]
            elif test_data.y.ndim == 3:  # [batch_size, 2, length]
                # 取第一个时间点的信道估计
                channel_estimates[i] = test_data.y[i, 0, 0] + 1j * test_data.y[i, 1, 0]
            else:
                # 默认使用第一个元素
                channel_estimates[i] = test_data.y[i, 0] + 1j * test_data.y[i, 1]
        else:
            # 如果没有信道估计数据，使用默认值1 (AWGN信道)
            channel_estimates[i] = 1.0 + 0j
            print(f"警告: 样本 {i} 没有信道估计数据，使用默认值")

    # 匹配滤波
    filtered_signal = np.convolve(baseline_signal, rrc_filter, mode='full')
    filtered_signal = filtered_signal[filter_delay:-filter_delay]
    
    # 下采样
    downsampled_signal = filtered_signal[::SAMPLES_PER_SYMBOL]
    
    # MMSE均衡
    if use_mmse:
        equalized_signal = mmse_equalization(downsampled_signal, channel_estimates, snrDB)
        return equalized_signal
    else:
        return downsampled_signal

# MMSE均衡函数
def mmse_equalization(received_signal, channel_estimates, snr_db):
    """
    MMSE均衡
    参数:
    - received_signal: 接收到的复信号
    - channel_estimates: 信道估计值
    - snr_db: 信噪比(dB)
    """
    # 转换为线性信噪比
    snr_linear = 10**(snr_db  / 10)
    
    # MMSE均衡器系数: w = h* / (|h|² + 1/SNR)
    # equalizer_coeff = np.conj(channel_estimates) / (np.abs(channel_estimates)**2 + 1/snr_linear)

    equalizer_coeff = np.conj(channel_estimates) / (np.abs(channel_estimates)**2 + snr_linear)
    # 应用均衡

    equalized_signal = received_signal * equalizer_coeff
    
    return equalized_signal

# 硬判决
def decision_making(downsampled_signal, threshold=0):
    real_part = np.real(downsampled_signal)
    imag_part = np.imag(downsampled_signal)

    decision = np.zeros((len(downsampled_signal),2), dtype=int)
    
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
    return decision

# 计算误码率
def calculate_ber(original_labels, predicted_labels):
    
    predicted_labels = predicted_labels.astype(int)

    error_num_i = np.sum(original_labels[:, 0] != predicted_labels[:, 0])
    error_num_q = np.sum(original_labels[:, 1] != predicted_labels[:, 1])
    #print(error_num_i)
    #print(error_num_q)
    error_num = error_num_i + error_num_q

    ber = error_num / (len(original_labels)*2)
    print(f"BER: {ber:.6f}")
    return ber

# 接收器 - 支持MMSE均衡
def matched_filter_decision(labels, snr_db, use_mmse=True, SAMPLES_PER_SYMBOL=16):
    """
    接收器函数，支持MMSE均衡
    参数:
    - labels: 真实标签
    - snr_db: 信噪比
    - use_mmse: 是否使用MMSE均衡
    """
    if use_mmse:
        equalized_signal = data_pre_with_mmse(snr_db, use_mmse=True)
    else:
        baseline_signal = data_pre_with_mmse(snr_db, use_mmse=False)
        equalized_signal = baseline_signal
    
    # 硬判决
    predicted_labels = decision_making(equalized_signal)

    # 计算误码率
    ber = calculate_ber(labels, predicted_labels)
    return ber

# 绘制BER曲线并保存图像
def plot_ber_curve(baseline_bers, mmse_bers, snr_range, save_path='ber_result/ber_curve.png'):
    plt.figure(figsize=(12, 8))

    # 绘制理论BER曲线
    theoretical_bers = [0.5 * erfc(np.sqrt(10 ** (snr_db / 10) / 2)) for snr_db in snr_range]
    plt.semilogy(snr_range, theoretical_bers, 'r--', label='理论曲线')

    # 绘制baseline和MMSE
    plt.semilogy(snr_range, baseline_bers, 'v-', label='Baseline (无均衡)')
    plt.semilogy(snr_range, mmse_bers, 's-', label='MMSE均衡')

    # 设置横轴仅显示偶数刻度
    xticks = np.arange(np.ceil(min(snr_range) / 2) * 2, np.floor(max(snr_range) / 2) * 2 + 2, 2)
    plt.xticks(xticks)

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Bit Error Rate vs. SNR (QPSK) - 有无MMSE均衡对比')
    plt.legend()

    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 保存图像
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

if __name__ == "__main__":

    # 标签数据
    label = np.load(r'F:\LJN\bishe\bishe\data\rayleigh_data\labels.npy')  
    label_data = label[0:100000]

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
    mmse_bers = []
    snr_range = np.arange(-5, 10, 1)
    
    for snr_db in snr_range:
        print(f"当前SNR: {snr_db} dB")
        
        # 测试无均衡的情况
        print("无均衡:")
        baseline_ber = matched_filter_decision(label_data_IQ, snr_db , use_mmse=False)
        baseline_bers.append(baseline_ber)
        
        # 测试MMSE均衡的情况
        print("MMSE均衡:")
        mmse_ber = matched_filter_decision(label_data_IQ, snr_db , use_mmse=True)
        mmse_bers.append(mmse_ber)
        
        print("-" * 50)

    # 绘图
    plot_ber_curve(baseline_bers, mmse_bers, snr_range, 
                   save_path=r'CDDM\main\ber_result\ber_result_with_mmse.png')

    # 打印性能改善统计
    improvement = np.array(baseline_bers) - np.array(mmse_bers)
    print("\n性能改善统计:")
    for i, snr in enumerate(snr_range):
        print(f"SNR={snr}dB: 无均衡BER={baseline_bers[i]:.6f}, MMSE BER={mmse_bers[i]:.6f}, 改善={improvement[i]:.6f}")