import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import os
import random

class UAVBaseDatasetGenerator:
    """
    生成一个“基础”数据集，仅包含期望信号及其核心损伤（衰落、PA非线性）。
    这个数据集将作为后续训练中动态添加干扰和噪声的“原料”。
    """
    def __init__(self):
        # --- 基本参数 ---
        self.NUM_SYMBOLS = 500000  # 增加符号数量，因为只生成一次
        #self.NUM_SYMBOLS = 100000 #测试集不用那么大
        self.SAMPLES_PER_SYMBOL = 16
        self.PULSE_SHAPING_FILTER_TAPS = 129
        self.PULSE_SHAPING_ROLLOFF = 0.25
        self.WINDOW_SIZE_IN_SYMBOLS = 3
        
        # --- 核心损伤参数 (与论文设想一致) ---
        self.NAKAGAMI_M_VALUES = [1.5, 1.8, 2, 2.5, 3]
        self.NAKAGAMI_OMEGA = 1.0
        self.PA_COEFFS = {'a3': -0.5, 'a5': 0.26}
        #self.PA_COEFFS = {'a3': 0, 'a5': 0}
        #self.CHANNEL_ESTIMATION_NMSE_DB = -20
        self.CHANNEL_ESTIMATION_NMSE_DB_RANGE = [-20, -10] # 定义NMSE的随机范围
        self.CHUNK_SIZE_IN_SYMBOLS = 5000 # 每个小块处理5000个符号

        # --- 自动计算的参数 ---
        self.WINDOW_LENGTH_IN_SAMPLES = self.WINDOW_SIZE_IN_SYMBOLS * self.SAMPLES_PER_SYMBOL
        self.OUTPUT_DIR = r"data/nakagmi_data" # 新的输出目录
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        self.print_summary()

    def print_summary(self):
        print("="*60)
        print("城市UAV集群通信 - 基础数据集生成器")
        print("此版本仅生成期望信号，并施加衰落和PA非线性损伤。")
        print("干扰和噪声将在训练时动态添加。")
        print("="*60)
        print(f"总符号数: {self.NUM_SYMBOLS}")
        print(f"每符号采样数: {self.SAMPLES_PER_SYMBOL}")
        print(f"处理块大小 (符号数): {self.CHUNK_SIZE_IN_SYMBOLS}") # 打印新参数
        print(f"Nakagami m 值列表: {self.NAKAGAMI_M_VALUES}")
        print(f"PA非线性系数: a3={self.PA_COEFFS['a3']}, a5={self.PA_COEFFS['a5']}")
        #print(f"信道估计NMSE (dB): {self.CHANNEL_ESTIMATION_NMSE_DB}")
        print(f"信道估计NMSE范围 (dB): {self.CHANNEL_ESTIMATION_NMSE_DB_RANGE}")
        print(f"输出目录: {self.OUTPUT_DIR}")
        print("="*60)
    
    # --- 以下方法与上一版代码大部分相同 ---

    def generate_qpsk_symbols(self, num_symbols):
        labels = np.random.randint(0, 4, num_symbols)
        qpsk_mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: +1 - 1j}
        symbols = np.array([qpsk_mapping[label] for label in labels], dtype=np.complex64)
        symbols /= np.sqrt(2)
        return symbols, labels

    def create_rrc_filter(self):
        t = np.arange(self.PULSE_SHAPING_FILTER_TAPS) - (self.PULSE_SHAPING_FILTER_TAPS - 1) / 2.0
        t /= self.SAMPLES_PER_SYMBOL
        beta = self.PULSE_SHAPING_ROLLOFF
        h = np.zeros_like(t, dtype=float)
        idx_zero = np.where(t == 0)[0]; 
        if idx_zero.size > 0: h[idx_zero] = 1 + beta * (4/np.pi - 1)
        idx_beta = np.where(np.abs(np.abs(t) - 1/(4*beta)) < 1e-9)[0]
        if idx_beta.size > 0: h[idx_beta] = (beta / np.sqrt(2)) * ((1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) + (1 - 2/np.pi) * np.cos(np.pi / (4 * beta)))
        idx_general = np.where((t != 0) & (np.abs(np.abs(t) - 1/(4*beta)) > 1e-9))[0]
        t_general = t[idx_general]
        num = np.sin(np.pi * t_general * (1 - beta)) + 4 * beta * t_general * np.cos(np.pi * t_general * (1 + beta))
        den = np.pi * t_general * (1 - (4 * beta * t_general)**2)
        h[idx_general] = num / den
        return h

    def pulse_shaping(self, symbols):
        upsampled = np.zeros(len(symbols) * self.SAMPLES_PER_SYMBOL, dtype=np.complex64)
        upsampled[::self.SAMPLES_PER_SYMBOL] = symbols
        rrc_filter = self.create_rrc_filter()
        baseband_stream = np.convolve(upsampled, rrc_filter, mode='full')
        actual_peak = np.max(np.abs(baseband_stream))
        print(f"脉冲成形后信号峰值: {actual_peak:.4f}")
        if actual_peak > 1e-9: baseband_stream /= actual_peak
        filter_delay = len(rrc_filter) // 2
        return baseband_stream.astype(np.complex64), filter_delay

    def _generate_nakagami_h(self, m, omega, num_samples):
        gamma_dist = gamma.rvs(a=m, scale=omega/m, size=num_samples)
        h_magnitude = np.sqrt(gamma_dist)
        h_phase = np.random.uniform(0, 2 * np.pi, num_samples)
        complex_h = h_magnitude * np.exp(1j * h_phase)
        return complex_h.astype(np.complex64)

    def apply_nakagami_channel(self, baseband_stream, m_value, omega_value):
        # #不添加nakagami h=1+0j
        # upsampled_h = np.ones_like(baseband_stream, dtype=np.complex64)
        # faded_stream = baseband_stream * upsampled_h


        num_symbols = len(baseband_stream) // self.SAMPLES_PER_SYMBOL
        h_sequence_per_symbol = self._generate_nakagami_h(m_value, omega_value, num_symbols)
        time_points_symbols = np.arange(num_symbols) * self.SAMPLES_PER_SYMBOL
        time_points_stream = np.arange(len(baseband_stream))
        h_magnitude_upsampled = np.interp(time_points_stream, time_points_symbols, np.abs(h_sequence_per_symbol))
        h_phase_upsampled = np.interp(time_points_stream, time_points_symbols, np.unwrap(np.angle(h_sequence_per_symbol)))
        upsampled_h = h_magnitude_upsampled * np.exp(1j * h_phase_upsampled)
        faded_stream = baseband_stream * upsampled_h
        return faded_stream, upsampled_h

    def apply_pa_nonlinearity(self, signal_stream):
        magnitude = np.abs(signal_stream)
        original_phase_component = np.divide(signal_stream, magnitude, out=np.ones_like(signal_stream, dtype=np.complex64), where=magnitude!=0)
        a3, a5 = self.PA_COEFFS['a3'], self.PA_COEFFS['a5']
        new_magnitude = magnitude + a3 * (magnitude**3) + a5 * (magnitude**5)
        nonlinear_signal_stream = new_magnitude * original_phase_component
        return nonlinear_signal_stream.astype(np.complex64)

    def generate_channel_estimate(self, true_h_stream, nmse_db):
        h_power = np.mean(np.abs(true_h_stream)**2)
        nmse_linear = 10**(nmse_db / 10.0)
        error_power = h_power * nmse_linear
        error_std = np.sqrt(error_power / 2)
        error_noise = error_std * (np.random.randn(len(true_h_stream)) + 1j * np.random.randn(len(true_h_stream)))
        estimated_h_stream = true_h_stream + error_noise
        return estimated_h_stream.astype(np.complex64)

    def sliding_window_slicing(self, ideal_baseband_stream, impaired_stream, filter_delay, labels, true_h_stream, estimated_h_stream):
        symbol_peak_positions = [filter_delay + i * self.SAMPLES_PER_SYMBOL for i in range(len(labels))]
        impaired_windows, clean_windows, valid_labels = [], [], []
        h_true_windows, h_estimated_windows = [], []
        half_window = self.WINDOW_LENGTH_IN_SAMPLES // 2

        for i, peak_pos in enumerate(symbol_peak_positions):
            start, end = peak_pos - half_window, peak_pos + half_window
            if start >= 0 and end < len(impaired_stream):
                # 切片
                impaired_windows.append(impaired_stream[start:end])
                clean_windows.append(ideal_baseband_stream[start:end])
                valid_labels.append(labels[i])
                # 注意：这里我们保存窗口中心点的信道信息
                h_true_windows.append(true_h_stream[peak_pos])
                h_estimated_windows.append(estimated_h_stream[peak_pos])
        
        # 转换为复数numpy数组
        impaired_windows = np.array(impaired_windows, dtype=np.complex64)
        clean_windows = np.array(clean_windows, dtype=np.complex64)
        valid_labels = np.array(valid_labels)
        h_true_windows = np.array(h_true_windows, dtype=np.complex64)
        h_estimated_windows = np.array(h_estimated_windows, dtype=np.complex64)

        return impaired_windows, clean_windows, valid_labels, h_true_windows, h_estimated_windows

    def create_and_save_dataset(self, data_dict):
        """保存所有数据集组件。"""
        num_samples = len(data_dict["labels"])
        indices = np.random.permutation(num_samples)
        
        print("\n" + "-"*20 + " 保存基础数据集 " + "-"*20)
        for name, data in data_dict.items():
            shuffled_data = data[indices]
            # 复数数据需要特殊处理，保存为IQ两路
            if np.iscomplexobj(shuffled_data):
                # 对于波形数据 (N, L) -> (N, 2, L)
                if shuffled_data.ndim == 2: 
                    data_to_save = np.stack([np.real(shuffled_data), np.imag(shuffled_data)], axis=1)
                # 对于信道数据 (N,) -> (N, 2)
                else:
                    data_to_save = np.stack([np.real(shuffled_data), np.imag(shuffled_data)], axis=1)
            else:
                data_to_save = shuffled_data
            
            path = os.path.join(self.OUTPUT_DIR, f"{name}.npy")
            np.save(path, data_to_save.astype(np.float32))
            print(f"  已保存: {path} (形状: {data_to_save.shape})")
        print("-" * 60)
        
    def generate(self):
        """
        核心生成流程：生成包含所有Nakagami衰落情况的基础数据集。
        """
        print("\n开始生成基础数据集...")
        
        # 1. 生成一次性的QPSK符号和理想波形
        symbols, labels = self.generate_qpsk_symbols(self.NUM_SYMBOLS)
        #clean_baseband_stream, filter_delay = self.pulse_shaping(symbols)
        
        all_impaired_windows, all_clean_windows, all_labels = [], [], []
        all_true_h, all_estimated_h = [], []

        # 2. 遍历不同的信道条件，生成对应的受损数据
        num_symbols_per_m = self.NUM_SYMBOLS // len(self.NAKAGAMI_M_VALUES)
        # <--- 修改 2: 引入内层循环来处理小块数据 ---
        for i, m_val in enumerate(self.NAKAGAMI_M_VALUES):
            print(f"  -> 开始处理 Nakagami m={m_val}...")
            
            # 计算当前m值对应的数据块的起始和结束索引
            m_start_symbol_idx = i * num_symbols_per_m
            m_end_symbol_idx = (i + 1) * num_symbols_per_m
            
            # 计算这个m值下有多少个小块
            num_chunks = (m_end_symbol_idx - m_start_symbol_idx) // self.CHUNK_SIZE_IN_SYMBOLS

            for j in range(num_chunks):
                # 计算当前小块的符号索引
                chunk_start_idx = m_start_symbol_idx + j * self.CHUNK_SIZE_IN_SYMBOLS
                chunk_end_idx = chunk_start_idx + self.CHUNK_SIZE_IN_SYMBOLS
                
                # 提取当前小块的符号和标签
                symbols_chunk = symbols[chunk_start_idx:chunk_end_idx]
                labels_chunk = labels[chunk_start_idx:chunk_end_idx]
                
                # 1. 脉冲成形 (只对当前小块)
                clean_stream_chunk, delay_chunk = self.pulse_shaping(symbols_chunk)
                
                # 2. 应用衰落
                faded_stream, true_h_stream = self.apply_nakagami_channel(
                    clean_stream_chunk, m_val, self.NAKAGAMI_OMEGA
                )
                
                # 3. 应用PA非线性
                nonlinear_stream = self.apply_pa_nonlinearity(faded_stream)
                
                # 4. 为当前小块生成一个随机的NMSE值
                nmse_db_current = np.random.uniform(
                    self.CHANNEL_ESTIMATION_NMSE_DB_RANGE[0], 
                    self.CHANNEL_ESTIMATION_NMSE_DB_RANGE[1]
                )
                #nmse_db_current = -20.0 # 我们先生成-10dB误差的数据集
                # 打印信息，可以看到NMSE在变化
                if j % 5 == 0 or j == num_chunks - 1: # 每隔5个块打印一次，避免刷屏
                    print(f"     块 {j+1}/{num_chunks}, NMSE (dB): {nmse_db_current:.2f}")

                # 5. 生成信道估计
                estimated_h_stream = self.generate_channel_estimate(
                    true_h_stream, nmse_db_current
                )
                
                # 6. 切片成窗口
                impaired_w, clean_w, labels_w, true_h_w, est_h_w = self.sliding_window_slicing(
                    clean_stream_chunk, nonlinear_stream, delay_chunk, labels_chunk,
                    true_h_stream, estimated_h_stream
                )
                
                # 7. 将当前小块的结果追加到总列表中
                all_impaired_windows.append(impaired_w)
                all_clean_windows.append(clean_w)
                all_labels.append(labels_w)
                all_true_h.append(true_h_w)
                all_estimated_h.append(est_h_w)
        
        # <--- 修改 3: 合并所有小块的结果 ---
        # 这个合并逻辑本身不需要改变，因为它只是简单地将列表中的所有numpy数组连接起来
        print("\n正在合并所有场景和小块的数据...")
        final_data = {
            "impaired_waveforms": np.concatenate(all_impaired_windows, axis=0),
            "clean_waveforms": np.concatenate(all_clean_windows, axis=0),
            "labels": np.concatenate(all_labels, axis=0),
            "true_h": np.concatenate(all_true_h, axis=0),
            "estimated_h": np.concatenate(all_estimated_h, axis=0)
        }

        self.create_and_save_dataset(final_data)
        
        print("\n" + "="*60)
        print("基础数据集生成完毕！")
        print("="*60)
        
        return final_data


def main():
    generator = UAVBaseDatasetGenerator()
    generator.generate()

if __name__ == "__main__":
    main()