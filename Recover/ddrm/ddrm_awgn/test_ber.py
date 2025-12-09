# 这个脚本用了和其他测试脚本不同的匹配滤波逻辑，这是一个遗留的“半错误”结构（在代码上做了繁琐而无用的操作，但最终得到了正确的结果，保留这个文件以作纪念。


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

def model_test(snrDB, n_steps):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======== 配置模型 ========
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(fr'ddrm/ddrm_awgn/results/best_model_epoch_with_n_steps{n_steps}.pth', map_location=device))

    # DDRM 对象
    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # 数据加载
    test_data = QPSKDataset(100000, 120000)
    test_data.x = add_awgn_noise_np(test_data.x, snrDB)
    # 确保数据类型为 Float
    test_data.x = torch.tensor(test_data.x, dtype=torch.float32)

    test_loader =DataLoader(test_data, batch_size=256, shuffle=False)

    # 用于存储所有生成的结果
    all_generated = []
    pbar = tqdm(test_loader, desc=f"snrdb: {snr_db}")
    
     # 遍历 test_loader 中的所有批次
    for x_sample in pbar:  # 根据你的数据结构调整这里
        x_sample = x_sample.to(device)  # 将批次数据移动到设备（GPU/CPU）

        # 添加高斯噪声
        x_denoised = ddrm.denoise(x_sample)

        all_generated.append(x_denoised.cpu().numpy())  # 将去噪后的信号存储到 CPU 上，方便后续处理


    # 合并所有结果 (20000, ...)
    all_generated = np.concatenate(all_generated, axis=0)

    # 重构信号
    model_signal = np.zeros((len(all_generated)*16,), dtype=np.complex64)
    baseline_signal = np.zeros((len(all_generated)*16,), dtype=np.complex64)

    for i in range(len(all_generated)):
        model_signal[16*i:16*i+16] = all_generated[i, 0, 24:40] + 1j*all_generated[i, 1, 24:40]  # 取中间16个点
        baseline_signal[16*i:16*i+16] = test_data.x[i, 0, 24:40] + 1j*test_data.x[i, 1, 24:40]  # 取中间16个点 

    return  model_signal, baseline_signal


# 下采样
def downsample(signal, SAMPLES_PER_SYMBOL=16):
    return signal[::SAMPLES_PER_SYMBOL]

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
def matched_filter_decision(labels, snr_db, n_steps, SAMPLES_PER_SYMBOL=16):

    recovered_signal, baseline_signal = model_test(snr_db, n_steps) 
    # 下采样
    #print(recovered_signal[:10])
    downsampled_model_signal = downsample(recovered_signal, SAMPLES_PER_SYMBOL)
    downsampled_baseline_signal = downsample(baseline_signal, SAMPLES_PER_SYMBOL)
    # 硬判决
    predicted_model_labels = decision_making(downsampled_model_signal)
    predicted_baseline_labels = decision_making(downsampled_baseline_signal)

    # 确保 labels 和 predicted_labels 都是扁平化的一维数组
    model_ber = calculate_ber(labels, predicted_model_labels)
    baseline_ber = calculate_ber(labels, predicted_baseline_labels)
    return model_ber, baseline_ber

# 绘制BER曲线并保存图像
def plot_ber_curve(output_re_bers, baseline_bers, snr_range, save_path='ber_result/ber_curve.png'):
    plt.figure(figsize=(12, 8))

    # 绘制模型BER曲线
    plt.semilogy(snr_range, output_re_bers, 'o-', label='Model')

    # 绘制理论BER曲线
    theoretical_bers = [0.5 * erfc(np.sqrt(10 ** (snr_db / 10) / 2)) for snr_db in snr_range]
    plt.semilogy(snr_range, theoretical_bers, 'r--', label='理论曲线')

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
    label = np.load(r'F:\LJN\bishe\bishe\data\awgn_data\qpsk_labels.npy')  

    label_data = label[100000:]

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

    model_bers = []
    baseline_bers = []
    snr_range = np.arange(-2, 8, 1)
    for snr_db in snr_range:
        print(f"当前SNR: {snr_db} dB")
        model_ber, baseline_ber =matched_filter_decision(label_data_IQ, snr_db - 10*math.log(16,10), n_steps, SAMPLES_PER_SYMBOL=16)
            
        model_bers.append(model_ber)
        baseline_bers.append(baseline_ber)

    #绘图
     # 绘制并保存BER曲线
    plot_ber_curve(model_bers, baseline_bers, snr_range, save_path=f'ddrm/ddrm_awgn/ber_result/ber_curve_nsetps{n_steps}.png')





        


    






