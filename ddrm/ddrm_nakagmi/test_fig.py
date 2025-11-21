# test_ddrm_qpsk.py
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
import math
from torchvision.utils import save_image
from model.unet import UNet, build_network
from ddrm_core import DDRM
from dataset.dataset import get_test_QPSKdataloader, get_signal_shape


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

def add_awgn_noise_torch(clean_data, EbN0_db):
    """
    clean_data: [batch, 2, length]  torch.Tensor (float, 可在 GPU 上)
    EbN0_db: 信噪比
    返回加噪后的信号，类型和设备与输入相同
    """
    # 计算信号功率 (I/Q 平均)
    signal_power = (clean_data[:,0,:].pow(2) + clean_data[:,1,:].pow(2)).mean()

    # 计算噪声功率
    EbN0_linear = 10 ** (EbN0_db / 10)
    N0 = signal_power / EbN0_linear
    noise_std = torch.sqrt(N0/2)

    # 生成 I/Q 两路独立高斯噪声
    noise_I = noise_std * torch.randn_like(clean_data[:,0,:])
    noise_Q = noise_std * torch.randn_like(clean_data[:,1,:])

    # 加噪
    noisy_data = clean_data.clone()
    noisy_data[:,0,:] += noise_I
    noisy_data[:,1,:] += noise_Q

    return noisy_data



def test_ddrm_snr(model, ddrm, dataloader, snr_list=[5, 10, 20, 30], device='cuda', save_dir='ddrm/ddrm_awgn/test_results'):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    ddrm.model.eval()

    # 选取一批样本
    x_sample = next(iter(dataloader))
    x_sample = x_sample[:8].to(device)  # 取前8个信号样本

    for snr_db in snr_list:
        # 添加高斯噪声
        x_noisy = add_awgn_noise_torch(x_sample, snr_db - 10*math.log(16,10))
        x_denoised = ddrm.denoise(x_noisy)

        # 创建一个 2x3 的子图布局
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        # 绘制原始信号的幅度图 (I 路)
        axs[0, 0].plot(x_sample[0][0].cpu().numpy(), label='Original I Channel', color='blue')
        axs[0, 0].set_title(f"Original Signal - I Channel - SNR {snr_db} dB")
        axs[0, 0].set_xlabel('Sample Index')
        axs[0, 0].set_ylabel('Magnitude')
        axs[0, 0].legend()

        # 绘制原始信号的幅度图 (Q 路)
        axs[0, 1].plot(x_sample[0][1].cpu().numpy(), label='Original Q Channel', color='purple')
        axs[0, 1].set_title(f"Original Signal - Q Channel - SNR {snr_db} dB")
        axs[0, 1].set_xlabel('Sample Index')
        axs[0, 1].set_ylabel('Magnitude')
        axs[0, 1].legend()

        # 绘制带噪信号的幅度图 (I 路)
        axs[0, 2].plot(x_noisy[0][0].cpu().numpy(), label='Noisy I Channel', color='red')
        axs[0, 2].set_title(f"Noisy Signal - I Channel - SNR {snr_db} dB")
        axs[0, 2].set_xlabel('Sample Index')
        axs[0, 2].set_ylabel('Magnitude')
        axs[0, 2].legend()

        # 绘制带噪信号的幅度图 (Q 路)
        axs[1, 0].plot(x_noisy[0][1].cpu().numpy(), label='Noisy Q Channel', color='blue')
        axs[1, 0].set_title(f"Noisy Signal - Q Channel - SNR {snr_db} dB")
        axs[1, 0].set_xlabel('Sample Index')
        axs[1, 0].set_ylabel('Magnitude')
        axs[1, 0].legend()

        # 绘制去噪信号的幅度图 (I 路)
        axs[1, 1].plot(x_denoised[0][0].cpu().numpy(), label='Denoised I Channel', color='green')
        axs[1, 1].set_title(f"Denoised Signal - I Channel - SNR {snr_db} dB")
        axs[1, 1].set_xlabel('Sample Index')
        axs[1, 1].set_ylabel('Magnitude')
        axs[1, 1].legend()

        # 绘制去噪信号的幅度图 (Q 路)
        axs[1, 2].plot(x_denoised[0][1].cpu().numpy(), label='Denoised Q Channel', color='orange')
        axs[1, 2].set_title(f"Denoised Signal - Q Channel - SNR {snr_db} dB")
        axs[1, 2].set_xlabel('Sample Index')
        axs[1, 2].set_ylabel('Magnitude')
        axs[1, 2].legend()

        # 调整布局，防止图像重叠
        plt.tight_layout()

        # 保存图像
        plt.savefig(os.path.join(save_dir, f"signal_comparison_snr{snr_db}.png"))
        plt.close()  # 关闭当前图像

        print(f"SNR={snr_db} dB 完成去噪，并保存结果。")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ======== 配置模型 ========
    n_steps = 50  # 扩散步数
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)

    # 加载训练好的模型权重
    model.load_state_dict(torch.load(r'ddrm/ddrm_awgn/results/best_model_epoch_with_n_steps50.pth', map_location=device))

    # DDRM 对象
    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # 数据加载
    dataloader = get_test_QPSKdataloader(start = 100000, end = 120000, batch_size=16)

    # 测试不同 SNR
    snr_list = np.arange(-2, 15, 5)  # 单位 dB
    test_ddrm_snr(model, ddrm, dataloader, snr_list=snr_list, device=device)