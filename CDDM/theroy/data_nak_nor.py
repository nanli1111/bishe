import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import os

class QPSKDatasetGenerator:
    def __init__(self):
        # --- 基本参数 ---
        self.NUM_SYMBOLS = 120000  # 生成的符号数量
        self.SAMPLES_PER_SYMBOL = 16  # 每个符号的采样点数
        self.PULSE_SHAPING_FILTER_TAPS = 129  # 脉冲成形滤波器的阶数（抽头数）
        self.PULSE_SHAPING_ROLLOFF = 0.25  # 滚降系数
        self.WINDOW_SIZE_IN_SYMBOLS = 3  # 滑动窗口大小（以符号为单位）
        
        # --- 全局损伤参数 ---
        self.NAKAGAMI_M_VALUES = [0.8, 1.0, 2.0, 3.0]  # Nakagami-m 衰落的 m 值列表
        self.NAKAGAMI_OMEGA = 1.0  # 信道平均功率 E[|h|^2]
        self.NONLINEARITY_COEFF = -0.2  # 三次非线性系数
        self.CHANNEL_ESTIMATION_ERROR_SNR_DB = 20  # 信道估计误差的信噪比 (dB)

        # --- 自动计算的参数 ---
        self.WINDOW_LENGTH_IN_SAMPLES = self.WINDOW_SIZE_IN_SYMBOLS * self.SAMPLES_PER_SYMBOL  # 窗口长度（以采样点为单位）
        self.OUTPUT_DIR = "./dataset_nak"  # 数据集输出目录
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        print("="*60)
        print("QPSK 数据集生成器已初始化 (无噪声版本)")
        print(f"每轮符号数: {self.NUM_SYMBOLS}")
        print(f"每符号采样数: {self.SAMPLES_PER_SYMBOL}")
        print(f"窗口长度: {self.WINDOW_LENGTH_IN_SAMPLES} 采样点")
        print(f"Nakagami m 值: {self.NAKAGAMI_M_VALUES}")
        print(f"非线性系数: {self.NONLINEARITY_COEFF}")
        print(f"信道估计信噪比 (dB): {self.CHANNEL_ESTIMATION_ERROR_SNR_DB}")
        print(f"输出目录: {self.OUTPUT_DIR}")
        print("="*60)

    def generate_qpsk_symbols(self):
        """生成 QPSK 符号和对应的标签。"""
        labels = np.random.randint(0, 4, self.NUM_SYMBOLS)
        qpsk_mapping = {0: 1 + 1j, 1: -1 + 1j, 2: 1 - 1j, 3: -1 - 1j}
        symbols = np.array([qpsk_mapping[label] for label in labels], dtype=np.complex64)
        symbols /= np.sqrt(2)  # 归一化符号，使平均功率为 1
        return symbols, labels

    def create_rrc_filter(self):
        """创建根升余弦 (RRC) 滤波器。"""
        t = np.arange(self.PULSE_SHAPING_FILTER_TAPS) - (self.PULSE_SHAPING_FILTER_TAPS - 1) / 2.0
        t /= self.SAMPLES_PER_SYMBOL
        beta = self.PULSE_SHAPING_ROLLOFF
        h = np.zeros_like(t, dtype=float)
        
        # 处理 t = 0 的特殊情况
        idx_zero = np.where(t == 0)[0]
        if idx_zero.size > 0:
            h[idx_zero] = 1 + beta * (4/np.pi - 1)
            
        # 处理 |t| = 1/(4*beta) 的特殊情况
        idx_beta = np.where(np.abs(np.abs(t) - 1/(4*beta)) < 1e-9)[0]
        if idx_beta.size > 0:
            val = (beta / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) + 
                                         (1 - 2/np.pi) * np.cos(np.pi / (4 * beta)))
            h[idx_beta] = val
        
        # 处理一般情况
        idx_general = np.where((t != 0) & (np.abs(np.abs(t) - 1/(4*beta)) > 1e-9))[0]
        t_general = t[idx_general]
        num = np.sin(np.pi * t_general * (1 - beta)) + 4 * beta * t_general * np.cos(np.pi * t_general * (1 + beta))
        den = np.pi * t_general * (1 - (4 * beta * t_general)**2)
        h[idx_general] = num / den
        
        return h

    def pulse_shaping(self, symbols):
        """对符号进行上采样并应用脉冲成形。"""
        # 对符号进行上采样，在符号之间插入 SAMPLES_PER_SYMBOL-1 个零
        upsampled = np.zeros(len(symbols) * self.SAMPLES_PER_SYMBOL, dtype=np.complex64)
        upsampled[::self.SAMPLES_PER_SYMBOL] = symbols
        
        rrc_filter = self.create_rrc_filter()
        # 通过卷积实现脉冲成形
        ideal_baseband_stream = np.convolve(upsampled, rrc_filter, mode='full')
        
        # --- 修改部分开始 ---
        # 找到生成波形自身的实际峰值 (使用 np.abs() 处理复数信号)
        actual_peak = np.max(np.abs(ideal_baseband_stream))
        
        # 用实际峰值进行归一化，确保 clean 信号的峰值幅度为 1
        if actual_peak > 1e-9:
            ideal_baseband_stream /= actual_peak
        # --- 修改部分结束 ---
        
        # 计算滤波器引入的延迟
        filter_delay = len(rrc_filter) // 2
        return ideal_baseband_stream, filter_delay

    def _generate_nakagami_h(self, m, omega, num_samples):
        """生成复数 Nakagami-m 衰落信道系数。"""
        # 从 Gamma 分布生成幅度平方
        gamma_dist = gamma.rvs(a=m, scale=omega/m, size=num_samples)
        h_magnitude = np.sqrt(gamma_dist)
        # 生成均匀分布的随机相位
        h_phase = np.random.uniform(0, 2 * np.pi, num_samples)
        complex_h = h_magnitude * np.exp(1j * h_phase)
        return complex_h.astype(np.complex64)

    def apply_nakagami_channel(self, ideal_baseband_stream, m_value, omega_value):
        """应用时变的 Nakagami-m 衰落信道。"""
        # 计算符号数量，为每个符号生成一个信道系数
        num_symbols = len(ideal_baseband_stream) // self.SAMPLES_PER_SYMBOL
        h_sequence_per_symbol = self._generate_nakagami_h(m_value, omega_value, num_symbols)
        
        # 定义符号和采样点的时间轴
        time_points_symbols = np.arange(num_symbols) * self.SAMPLES_PER_SYMBOL
        time_points_stream = np.arange(len(ideal_baseband_stream))
        
        # 通过线性插值，将信道系数从符号速率上采样到采样点速率
        h_magnitude_upsampled = np.interp(time_points_stream, time_points_symbols, np.abs(h_sequence_per_symbol))
        # 对相位进行解缠绕后再插值，避免相位跳变
        h_phase_upsampled = np.interp(time_points_stream, time_points_symbols, np.unwrap(np.angle(h_sequence_per_symbol)))
        
        # 合成上采样后的复数信道系数
        upsampled_h = h_magnitude_upsampled * np.exp(1j * h_phase_upsampled)
        # 将信号与信道系数相乘，模拟衰落
        faded_stream = ideal_baseband_stream * upsampled_h
        return faded_stream.astype(np.complex64), upsampled_h.astype(np.complex64)

    def apply_nonlinearity(self, faded_signal_stream):
        """应用三次非线性损伤。"""
        nonlinear_signal_stream = faded_signal_stream - self.NONLINEARITY_COEFF * (faded_signal_stream**3)
        return nonlinear_signal_stream.astype(np.complex64)

    def add_awgn_noise(self, signal_stream, snr_db):
        """添加高斯白噪声（此函数在当前流程中未被调用）。"""
        signal_power = np.mean(np.abs(signal_stream)**2)
        snr_linear = 10**(snr_db / 10.0)
        noise_power = signal_power / snr_linear
        noise_std = np.sqrt(noise_power / 2)  # I/Q 两路，所以除以2
        noise = noise_std * (np.random.randn(len(signal_stream)) + 1j * np.random.randn(len(signal_stream)))
        noisy_signal_stream = signal_stream + noise
        return noisy_signal_stream.astype(np.complex64)

    def generate_channel_estimate(self, true_h_stream, error_snr_db):
        """通过向真实信道添加噪声来生成不完美的信道估计。"""
        h_power = np.mean(np.abs(true_h_stream)**2)
        error_snr_linear = 10**(error_snr_db / 10.0)
        error_power = h_power / error_snr_linear
        error_std = np.sqrt(error_power / 2)
        error_noise = error_std * (np.random.randn(len(true_h_stream)) + 1j * np.random.randn(len(true_h_stream)))
        estimated_h_stream = true_h_stream + error_noise
        return estimated_h_stream.astype(np.complex64)

    def sliding_window_slicing(self, ideal_baseband_stream, impaired_full_stream, 
                                 filter_delay, labels, 
                                 full_stream_true_h, full_stream_estimated_h):
        """使用滑动窗口将数据切片，用于后续处理。"""
        # 计算每个符号峰值（最佳采样点）的位置
        symbol_peak_positions = [filter_delay + i * self.SAMPLES_PER_SYMBOL for i in range(self.NUM_SYMBOLS)]
        
        impaired_windowed_data, clean_windowed_data, valid_labels = [], [], []
        h_true_for_windows, h_estimated_for_windows = [], []
        
        half_window = self.WINDOW_LENGTH_IN_SAMPLES // 2
        
        # 以每个符号的最佳采样点为中心进行切片
        for i, peak_pos in enumerate(symbol_peak_positions):
            window_start = peak_pos - half_window
            window_end = peak_pos + half_window
            
            # 确保窗口不会超出信号流的边界
            if window_start >= 0 and window_end < len(impaired_full_stream):
                window_impaired = impaired_full_stream[window_start:window_end]
                window_clean = ideal_baseband_stream[window_start:window_end]
                # 提取窗口中心的信道信息
                h_true_at_center = full_stream_true_h[peak_pos]
                h_estimated_at_center = full_stream_estimated_h[peak_pos]
                
                # 再次确认窗口长度是否正确
                if len(window_impaired) == self.WINDOW_LENGTH_IN_SAMPLES:
                    impaired_windowed_data.append(window_impaired)
                    clean_windowed_data.append(window_clean)
                    valid_labels.append(labels[i])
                    h_true_for_windows.append(h_true_at_center)
                    h_estimated_for_windows.append(h_estimated_at_center)

        # 将列表转换为 NumPy 数组
        impaired_windowed_data = np.array(impaired_windowed_data, dtype=np.complex64)
        clean_windowed_data = np.array(clean_windowed_data, dtype=np.complex64)
        valid_labels = np.array(valid_labels)
        h_true_for_windows = np.array(h_true_for_windows, dtype=np.complex64)
        h_estimated_for_windows = np.array(h_estimated_for_windows, dtype=np.complex64)
        
        # 将复数数据分离为 I/Q 两路，以满足模型输入格式
        impaired_data_iq = np.zeros((len(impaired_windowed_data), 2, self.WINDOW_LENGTH_IN_SAMPLES), dtype=np.float32)
        impaired_data_iq[:, 0, :] = np.real(impaired_windowed_data)
        impaired_data_iq[:, 1, :] = np.imag(impaired_windowed_data)

        clean_data_iq = np.zeros((len(clean_windowed_data), 2, self.WINDOW_LENGTH_IN_SAMPLES), dtype=np.float32)
        clean_data_iq[:, 0, :] = np.real(clean_windowed_data)
        clean_data_iq[:, 1, :] = np.imag(clean_windowed_data)
        
        h_true_iq = np.zeros((len(h_true_for_windows), 2), dtype=np.float32)
        h_true_iq[:, 0] = np.real(h_true_for_windows)
        h_true_iq[:, 1] = np.imag(h_true_for_windows)

        h_estimated_iq = np.zeros((len(h_estimated_for_windows), 2), dtype=np.float32)
        h_estimated_iq[:, 0] = np.real(h_estimated_for_windows)
        h_estimated_iq[:, 1] = np.imag(h_estimated_for_windows)
        
        return impaired_data_iq, clean_data_iq, valid_labels, h_true_iq, h_estimated_iq

    def create_and_save_dataset(self, impaired_data_iq, clean_data_iq, labels, 
                                  h_true_iq, h_estimated_iq):
        """打乱并保存所有数据集组件。"""
        num_samples = len(impaired_data_iq)
        random_indices = np.random.permutation(num_samples)
        
        # 使用相同的随机索引打乱所有数据，以保持对应关系
        impaired_data_iq = impaired_data_iq[random_indices]
        clean_data_iq = clean_data_iq[random_indices]
        labels = labels[random_indices]
        h_true_iq = h_true_iq[random_indices]
        h_estimated_iq = h_estimated_iq[random_indices]

        files_to_save = {
            "qpsk_impaired.npy": impaired_data_iq,
            "qpsk_clean.npy": clean_data_iq,
            "qpsk_labels.npy": labels,
            "qpsk_true_h.npy": h_true_iq,
            "qpsk_estimated_h.npy": h_estimated_iq
        }
        
        print("\n" + "-"*20 + " 保存最终数据集 " + "-"*20)
        for filename, data in files_to_save.items():
            path = os.path.join(self.OUTPUT_DIR, filename)
            np.save(path, data)
            print(f"  已保存: {path} (形状: {data.shape})")
        print("-" * 60)
        
        return impaired_data_iq, clean_data_iq, labels, h_true_iq, h_estimated_iq

    def generate_dataset(self):
        """完整的端到端数据集生成流程。"""
        print("\n开始完整的数据集生成流程 (无噪声版本)...")
        
        # 1. 生成原始 QPSK 符号
        symbols, labels = self.generate_qpsk_symbols()
        # 2. 脉冲成形，得到理想的 clean 信号
        ideal_baseband_stream, filter_delay = self.pulse_shaping(symbols)
        
        all_impaired_windows, all_clean_windows, all_labels = [], [], []
        all_true_h, all_estimated_h = [], []

        # 核心循环: 遍历不同的信道条件（仅衰落）
        for m_val in self.NAKAGAMI_M_VALUES:
            print(f"\n  -> 正在为 Nakagami m={m_val} 生成数据...")
            
            # 3. 应用 Nakagami 衰落信道
            faded_stream, true_h_stream = self.apply_nakagami_channel(
                ideal_baseband_stream, m_val, self.NAKAGAMI_OMEGA
            )
            
            # 4. 应用非线性
            nonlinear_stream = self.apply_nonlinearity(faded_stream)
            
            # 5. 添加高斯白噪声 - 【此步骤已按要求跳过】
            # 传递给切片函数的受损信号是非线性步骤的输出
            impaired_stream_for_slicing = nonlinear_stream
            
            # 6. 生成不完美的信道估计
            estimated_h_stream = self.generate_channel_estimate(
                true_h_stream, self.CHANNEL_ESTIMATION_ERROR_SNR_DB
            )
            
            # 7. 切片成窗口
            impaired_w, clean_w, labels_w, true_h_w, est_h_w = self.sliding_window_slicing(
                ideal_baseband_stream, impaired_stream_for_slicing, filter_delay, labels, 
                true_h_stream, estimated_h_stream
            )
            
            # 收集当前条件下的所有数据
            all_impaired_windows.append(impaired_w)
            all_clean_windows.append(clean_w)
            all_labels.append(labels_w)
            all_true_h.append(true_h_w)
            all_estimated_h.append(est_h_w)
        
        print("\n正在合并所有场景下的数据...")
        # 8. 将所有场景的数据拼接在一起
        final_impaired_data = np.concatenate(all_impaired_windows, axis=0)
        final_clean_data = np.concatenate(all_clean_windows, axis=0)
        final_labels = np.concatenate(all_labels, axis=0)
        final_true_h = np.concatenate(all_true_h, axis=0)
        final_estimated_h = np.concatenate(all_estimated_h, axis=0)

        # 9. 打乱并保存最终的数据集
        impaired_data, clean_data, labels, true_h, est_h = self.create_and_save_dataset(
            final_impaired_data, final_clean_data, final_labels, final_true_h, final_estimated_h
        )
        
        print("\n" + "="*60)
        print("无噪声数据集生成完毕！")
        print("="*60)
        
        return impaired_data, clean_data, labels, true_h, est_h

    def visualize_sample(self, impaired_data, clean_data, labels, num_samples=3):
        """可视化几个样本，对比 clean 和 impaired 波形。"""
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples))
        fig.suptitle("样本波形可视化 (Clean vs. Impaired - 无噪声)", fontsize=16)

        for i in range(num_samples):
            ax_i0, ax_i1 = axes[i, 0], axes[i, 1]
            
            ax_i0.plot(clean_data[i, 0, :], 'b-', label='I (Clean)', alpha=0.9)
            ax_i0.plot(impaired_data[i, 0, :], 'r-', label='I (Impaired)', alpha=0.7)
            ax_i0.set_title(f'样本 #{i} - I 路 (标签: {labels[i]})')
            ax_i0.legend()
            ax_i0.grid(True, linestyle=':')
            
            ax_i1.plot(clean_data[i, 1, :], 'b-', label='Q (Clean)', alpha=0.9)
            ax_i1.plot(impaired_data[i, 1, :], 'r-', label='Q (Impaired)', alpha=0.7)
            ax_i1.set_title(f'样本 #{i} - Q 路 (标签: {labels[i]})')
            ax_i1.legend()
            ax_i1.grid(True, linestyle=':')
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        save_path = os.path.join(self.OUTPUT_DIR, "sample_visualization.png")
        plt.savefig(save_path)
        plt.show()
        print(f"样本可视化图像已保存至: {save_path}")


def main():
    """主执行函数。"""
    generator = QPSKDatasetGenerator()
    impaired_data, clean_data, labels, true_h, estimated_h = generator.generate_dataset()
    
    print("\n最终数据集统计:")
    print(f"受损数据形状: {impaired_data.shape}")
    print(f"干净数据形状: {clean_data.shape}")
    print(f"标签形状:     {labels.shape}")
    print(f"真实H形状:    {true_h.shape}")
    print(f"估计H形状:    {estimated_h.shape}")
    
    unique_labels, counts = np.unique(labels, return_counts=True)
    print("\n标签分布:")
    for label, count in zip(unique_labels, counts):
        print(f"  标签 {label}: {count} 个样本 ({count/len(labels)*100:.2f}%)")

    generator.visualize_sample(impaired_data, clean_data, labels)

if __name__ == "__main__":
    main()