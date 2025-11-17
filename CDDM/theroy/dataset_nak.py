import torch
from torch.utils.data import Dataset
import numpy as np
import os
import matplotlib.pyplot as plt

# ==============================================================================
# 区域 1: 核心工具函数 - AWGN噪声生成器
# (此函数保持不变，因为它是一个通用的、与数据集无关的工具)
# ==============================================================================
def add_awgn_noise(signal_iq: np.ndarray, snr_db: float) -> np.ndarray:
    """
    向给定的I/Q信号（形状为 [2, N]）添加高斯白噪声(AWGN)。
    
    此函数精确地根据信噪比(SNR)计算噪声功率，并将其平均分配到I路和Q路。
    
    参数:
        signal_iq (np.ndarray): 输入的复数信号，以 [2, N] 的numpy数组表示，其中第一行为I路，第二行为Q路。
        snr_db (float): 目标信噪比，单位为分贝(dB)。

    返回:
        np.ndarray: 添加了AWGN噪声后的信号，形状与输入相同。
    """
    # 1. 计算信号功率 (Signal Power)
    #    功率定义为信号平方的均值。对于I/Q信号，我们先求和I^2 + Q^2，再对所有采样点取平均。
    signal_power = np.mean(np.sum(np.power(signal_iq, 2), axis=0))

    # 如果信号功率为0（例如全零输入），则不添加噪声，直接返回
    if signal_power < 1e-12: # 使用一个小的阈值来避免浮点数精度问题
        return signal_iq

    # 2. 将信噪比从dB转换为线性尺度
    snr_linear = 10 ** (snr_db / 10.0)
    
    # 3. 根据信噪比定义计算总噪声功率
    #    SNR = Signal Power / Noise Power  =>  Noise Power = Signal Power / SNR
    noise_power_total = signal_power / snr_linear
    
    # 4. 计算每条路的噪声功率和标准差
    #    总噪声功率平均分配到I路和Q路，因此每路的噪声功率是总功率的一半。
    #    噪声的标准差是其方差（功率）的平方根。
    noise_power_per_channel = noise_power_total / 2.0
    noise_std = np.sqrt(noise_power_per_channel)
    
    # 5. 生成符合该标准差的高斯噪声
    noise = np.random.normal(0, noise_std, signal_iq.shape)
    
    # 6. 返回加噪后的信号
    return (signal_iq + noise).astype(np.float32)

# ==============================================================================
# 区域 2: 数据集类 (已根据半盲GAN要求修改)
# ==============================================================================
class QPSKFadingDataset(Dataset):
    def __init__(self, data_dir: str):
        # --- [核心修改 1]: 添加信道估计文件的路径 ---
        self.impaired_path = os.path.join(data_dir, "qpsk_impaired.npy")
        self.clean_path = os.path.join(data_dir, "qpsk_clean.npy")
        self.labels_path = os.path.join(data_dir, "qpsk_labels.npy")
        self.h_estimated_path = os.path.join(data_dir, "qpsk_estimated_h.npy") # 新增

        # 检查所有必需文件是否存在
        for path in [self.impaired_path, self.clean_path, self.labels_path, self.h_estimated_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(f"数据集文件未找到: {path}")

        # --- [核心修改 2]: 加载所有数据，包括信道估计 ---
        # 使用内存映射模式(mmap_mode='r')可以高效处理无法完全加载到RAM中的大型数据集
        self.impaired_data = np.load(self.impaired_path, mmap_mode='r')
        self.clean_data = np.load(self.clean_path, mmap_mode='r')
        self.labels = np.load(self.labels_path, mmap_mode='r')
        self.h_estimated_data = np.load(self.h_estimated_path, mmap_mode='r') # 新增
        
        # 确保所有数据集组件的样本数量一致
        assert len(self.impaired_data) == len(self.clean_data) == len(self.labels) == len(self.h_estimated_data), \
            "所有数据文件 (impaired, clean, labels, h_estimated) 的样本数量必须一致！"

        # 动态添加噪声的SNR范围
        self.snr_range_db = (5.0, 15.0)

        print("QPSKFadingDataset 初始化完成:")
        print(f"  - 样本总数: {len(self)}")
        print(f"  - 信号波形形状: {self.impaired_data.shape}")
        print(f"  - 信道估计形状: {self.h_estimated_data.shape}")
        print(f"  - GAN的最终输入形状将为: [Batch, 4, {self.impaired_data.shape[2]}]") # 提示最终输入形状
        
    def __len__(self) -> int:
        return len(self.impaired_data)

    def set_snr_range(self, min_snr_db: float, max_snr_db: float):
        self.snr_range_db = (min_snr_db, max_snr_db)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # 1. 获取原始数据 (信号、目标、标签、信道估计)
        source_signal_np = self.impaired_data[idx].astype(np.float32)
        target_signal_np = self.clean_data[idx].astype(np.float32)
        label_np = self.labels[idx]
        h_estimated_np = self.h_estimated_data[idx] # 形状: (2,) -> [h_real, h_imag]

        # 2. 动态添加AWGN噪声
        min_snr, max_snr = self.snr_range_db
        random_snr_db = np.random.uniform(min_snr, max_snr)
        noisy_source_signal_np = add_awgn_noise(source_signal_np, random_snr_db)

        scaled_noisy_signal_np = noisy_source_signal_np  # 形状: [2, L]

        # --- [核心修改 3]: 构建GAN的4通道输入 ---
        # 目标: 将形状为[2, L]的信号 和 形状为[2]的信道估计ĥ 合并为 [4, L]的张量
        
        # 步骤 A: 获取信号长度 L
        signal_length = scaled_noisy_signal_np.shape[1]
        
        # 步骤 B: 将信道估计ĥ从 [2] 扩展到 [2, L]
        # 方法：创建一个新的 [2, L] 数组，并将ĥ的实部和虚部分别填充到两行
        h_expanded_np = np.zeros_like(noisy_source_signal_np, dtype=np.float32)
        h_expanded_np[0, :] = h_estimated_np[0]  # 实部填充到第一行
        h_expanded_np[1, :] = h_estimated_np[1]  # 虚部填充到第二行
        
        # 步骤 C: 沿着通道维度(axis=0)拼接信号和扩展后的ĥ
        gan_input_np = np.concatenate([scaled_noisy_signal_np, h_expanded_np], axis=0) # 形状: [4, L]

        # 4. 转换为PyTorch张量
        input_tensor = torch.from_numpy(gan_input_np)
        target_tensor = torch.from_numpy(target_signal_np.copy())
        label_tensor = torch.tensor(label_np, dtype=torch.long)

        # 返回 (GAN输入, GAN目标, 标签)
        return input_tensor, target_tensor, label_tensor
        
# ==============================================================================
# 区域 3: 验证与测试 (已更新以适配新的数据格式)
# ==============================================================================
def verify_snr_calculation(num_tests: int = 1000):
    """独立验证add_awgn_noise函数计算的准确性。"""
    print("\n" + "="*50)
    print("开始验证SNR计算的准确性...")
    
    # 创建一个简单的、功率已知的正弦/余弦测试信号
    L = 48
    t = np.linspace(0, 2*np.pi, L, endpoint=False)
    test_signal = np.vstack([np.cos(t), np.sin(t)]).astype(np.float32) # 振幅为1
    
    theoretical_power = np.mean(np.sum(np.power(test_signal, 2), axis=0))
    print(f"测试信号理论功率: {theoretical_power:.4f} (应为1.0)")

    test_snrs_db = [0, 5, 10, 20]
    for snr_db in test_snrs_db:
        measured_snrs = []
        for _ in range(num_tests):
            noisy_signal = add_awgn_noise(test_signal, snr_db)
            noise_realization = noisy_signal - test_signal
            measured_noise_power = np.mean(np.sum(np.power(noise_realization, 2), axis=0))
            # 防止噪声功率为0导致log10计算错误
            if measured_noise_power > 1e-12:
                measured_snr = 10 * np.log10(theoretical_power / measured_noise_power)
                measured_snrs.append(measured_snr)
        
        avg_measured = np.mean(measured_snrs)
        std_measured = np.std(measured_snrs)
        print(f"  - 目标SNR: {snr_db:2.1f} dB -> 测量SNR: {avg_measured:5.2f} ± {std_measured:.2f} dB")
    print("SNR验证完成。多次测量的平均值应非常接近目标值。")
    print("="*50 + "\n")


def test_dataset_loading_and_plotting(dataset: QPSKFadingDataset, num_samples_to_plot: int = 3):
    """测试数据集的加载、取样和可视化功能。"""
    print("\n" + "="*50)
    print("开始测试数据集加载与可视化 (4通道输入版本)...")
    
    # 固定SNR以进行可复现的可视化对比
    dataset.set_snr_range(10.0, 10.0)

    # 设置中文字体 (如果需要)
    try:
        plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文标签
        plt.rcParams['axes.unicode_minus'] = False # 用于正常显示负号
    except:
        print("注意: 未找到'SimHei'字体，中文可能无法正常显示。")

    
    fig, axes = plt.subplots(num_samples_to_plot, 2, figsize=(16, 4 * num_samples_to_plot))
    fig.suptitle("半盲GAN数据集样本可视化 (信号与信道信息)", fontsize=16)

    for i in range(num_samples_to_plot):
        # 从数据集中获取一个样本
        input_tensor, target_tensor, label = dataset[i]
        
        # --- [核心修改]: 从4通道输入中分离出信号和信道信息 ---
        # input_tensor 形状为 [4, L]
        noisy_signal_iq = input_tensor.numpy()[:2, :]  # 前2个通道是信号的I/Q
        h_est_real = input_tensor.numpy()[2, 0]      # 第3个通道是ĥ的实部 (所有值都一样)
        h_est_imag = input_tensor.numpy()[3, 0]      # 第4个通道是ĥ的虚部 (所有值都一样)
        
        # 获取用于对比的、未加噪的衰落信号和理想信号
        fading_signal = dataset.impaired_data[i] 
        clean_signal = target_tensor.numpy()
        
        # 构建图表标题，包含信道信息
        title_info = f"(标签: {label.item()}, ĥ={h_est_real:.2f}{h_est_imag:+.2f}j)"

        # 绘制I路
        axes[i, 0].plot(clean_signal[0, :], 'g-', label='I (Clean/理想目标)', linewidth=2.5, alpha=0.8)
        axes[i, 0].plot(fading_signal[0, :], 'b-', alpha=0.7, label='I (Fading/衰落后)')
        axes[i, 0].plot(noisy_signal_iq[0, :], 'r--', alpha=0.8, label='I (Noisy/归一化输入)')
        axes[i, 0].set_title(f"样本 {i+1} - I 路 {title_info}")
        axes[i, 0].grid(True, linestyle=':')
        axes[i, 0].legend()
        
        # 绘制Q路
        axes[i, 1].plot(clean_signal[1, :], 'g-', label='Q (Clean/理想目标)', linewidth=2.5, alpha=0.8)
        axes[i, 1].plot(fading_signal[1, :], 'b-', alpha=0.7, label='Q (Fading/衰落后)')
        axes[i, 1].plot(noisy_signal_iq[1, :], 'r--', alpha=0.8, label='Q (Noisy/归一化输入)')
        axes[i, 1].set_title(f"样本 {i+1} - Q 路 {title_info}")
        axes[i, 1].grid(True, linestyle=':')
        axes[i, 1].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("fading_dataset_4channel_test.png")
    print("可视化图像已保存到 fading_dataset_4channel_test.png")
    print("="*50 + "\n")

# ==============================================================================
# 区域 4: 主执行模块
# ==============================================================================
if __name__ == '__main__':
    # **重要**: 此目录应指向您运行数据生成脚本时创建的目录
    DATA_DIRECTORY = "./dataset_nak"

    # 步骤 1: 独立验证SNR计算逻辑
    verify_snr_calculation()

    # 步骤 2: 创建数据集实例
    try:
        # 使用新的类名实例化
        qpsk_dataset = QPSKFadingDataset(data_dir=DATA_DIRECTORY)
        
        # 步骤 3: 测试数据集加载和可视化
        test_dataset_loading_and_plotting(dataset=qpsk_dataset, num_samples_to_plot=3)

        # 额外演示: 在实际训练循环中如何动态调整SNR
        print("\n演示在训练循环中如何设置不同的SNR范围:")
        
        # 早期训练阶段，可以使用高SNR，让模型先学习基本恢复任务
        print("--- 早期训练阶段 ---")
        qpsk_dataset.set_snr_range(15.0, 20.0)
        input_sample, target_sample, label = qpsk_dataset[0]
        print(f"获取一个高信噪比样本，输入形状: {input_sample.shape} (验证为4通道)")
        # ... 此处可将样本送入模型进行训练 ...

        # 后期训练阶段，可以降低SNR，增加任务难度，提升模型鲁棒性
        print("\n--- 后期训练阶段 ---")
        qpsk_dataset.set_snr_range(-5.0, 5.0)
        input_sample, target_sample, label = qpsk_dataset[0]
        print(f"获取一个低信噪比样本，输入形状: {input_sample.shape} (验证为4通道)")
        # ... 此处可将样本送入模型进行训练 ...

    except FileNotFoundError as e:
        print(f"\n错误: {e}")
        print(f"请确保衰落数据集已在 '{DATA_DIRECTORY}' 目录下生成。")
        print("如果目录不同，请修改脚本中的 DATA_DIRECTORY 变量。")