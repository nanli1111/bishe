import numpy as np
import matplotlib.pyplot as plt
import os

class UAVBaseDatasetGenerator:
    """
    生成一个“基础”数据集，仅包含期望信号及其核心损伤（瑞利衰落）。
    这里采用 **符号域乘 h 再成型** 的方式：
        symbols -> h_per_symbol * symbols -> pulse shaping
    这样经过理想匹配滤波 + 抽样后，更接近教科书里的
        y_k = h_k x_k + n_k
    模型，便于和 Rayleigh 理论 BER 对齐。
    """
    def __init__(self):
        # --- 基本参数 ---
        self.NUM_SYMBOLS = 500000  # 符号总数
        self.SAMPLES_PER_SYMBOL = 16  # 每个符号采样点数
        self.PULSE_SHAPING_FILTER_TAPS = 129  # 滤波器长度
        self.PULSE_SHAPING_ROLLOFF = 0.25  # 滤波器滚降因子
        self.WINDOW_SIZE_IN_SYMBOLS = 3  # 窗口大小（符号数）
        
        # --- 瑞利衰落参数 ---
        self.CHUNK_SIZE_IN_SYMBOLS = self.NUM_SYMBOLS  # 处理整个数据集，无分块

        # --- 自动计算的参数 ---
        self.WINDOW_LENGTH_IN_SAMPLES = self.WINDOW_SIZE_IN_SYMBOLS * self.SAMPLES_PER_SYMBOL
        self.OUTPUT_DIR = r"data/rayleigh_data_withoutisi"  # 输出目录
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        # self.print_summary()

    def print_summary(self):
        print("="*60)
        print("城市UAV集群通信 - 基础数据集生成器")
        print("此版本仅生成期望信号，并施加瑞利衰落损伤。")
        print("干扰和噪声将在训练时动态添加。")
        print("="*60)
        print(f"总符号数: {self.NUM_SYMBOLS}")
        print(f"每符号采样数: {self.SAMPLES_PER_SYMBOL}")
        print(f"输出目录: {self.OUTPUT_DIR}")
        print("="*60)
    
    def generate_qpsk_symbols(self, num_symbols):
        labels = np.random.randint(0, 4, num_symbols)  # 随机生成QPSK符号标签
        qpsk_mapping = {
            0: 1 + 1j,
            1: -1 + 1j,
            2: -1 - 1j,
            3: +1 - 1j
        }
        symbols = np.array([qpsk_mapping[label] for label in labels], dtype=np.complex64)
        symbols /= np.sqrt(2)  # 归一化到单位平均功率
        return symbols, labels

    def create_rrc_filter(self):
        t = np.arange(self.PULSE_SHAPING_FILTER_TAPS) - (self.PULSE_SHAPING_FILTER_TAPS - 1) / 2.0
        t /= self.SAMPLES_PER_SYMBOL
        beta = self.PULSE_SHAPING_ROLLOFF
        h = np.zeros_like(t, dtype=float)

        idx_zero = np.where(t == 0)[0]
        if idx_zero.size > 0:
            h[idx_zero] = 1 + beta * (4/np.pi - 1)

        idx_beta = np.where(np.abs(np.abs(t) - 1/(4*beta)) < 1e-9)[0]
        if idx_beta.size > 0:
            h[idx_beta] = (beta / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi / (4 * beta)) +
                (1 - 2/np.pi) * np.cos(np.pi / (4 * beta))
            )

        idx_general = np.where((t != 0) & (np.abs(np.abs(t) - 1/(4*beta)) > 1e-9))[0]
        t_general = t[idx_general]
        num = np.sin(np.pi * t_general * (1 - beta)) + \
              4 * beta * t_general * np.cos(np.pi * t_general * (1 + beta))
        den = np.pi * t_general * (1 - (4 * beta * t_general)**2)
        h[idx_general] = num / den

        return h

    def pulse_shaping(self, symbols):
        """
        RRC 成型：上采样 + 卷积 + 归一化到峰值 1，并返回滤波器时延。
        """
        upsampled = np.zeros(len(symbols) * self.SAMPLES_PER_SYMBOL, dtype=np.complex64)
        upsampled[::self.SAMPLES_PER_SYMBOL] = symbols

        rrc_filter = self.create_rrc_filter()
        baseband_stream = np.convolve(upsampled, rrc_filter, mode='full')

        actual_peak = np.max(np.abs(baseband_stream))
        if actual_peak > 1e-9:
            baseband_stream /= actual_peak

        filter_delay = len(rrc_filter) // 2
        return baseband_stream.astype(np.complex64), filter_delay

    def _generate_rayleigh_h(self, num_samples):
        """
        CN(0,1) => |h| 瑞利，E[|h|^2] = 1
        每个符号一个独立的 h_k。
        """
        h = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        return h.astype(np.complex64)

    # ================= 核心修改点 1：符号域乘 h =================
    def apply_rayleigh_channel(self, symbols):
        """
        在符号域施加单径平坦瑞利衰落：
            s_k = x_k       （原始符号）
            h_k ~ CN(0,1)   （每个符号一个系数）
            r_k = h_k * s_k （衰落后的符号）

        返回:
            faded_symbols : 衰落后的符号序列
            h_per_symbol  : 对应的每符号信道系数（用于 CSI / 保存 true_h）
        """
        num_symbols = len(symbols)
        h_per_symbol = self._generate_rayleigh_h(num_symbols)   # [num_symbols]
        faded_symbols = symbols * h_per_symbol                  # [num_symbols]
        return faded_symbols.astype(np.complex64), h_per_symbol

    # ============= 核心修改点 2：这里用的是 h_per_symbol[i] ============
    def sliding_window_slicing(self,
                               ideal_baseband_stream,
                               impaired_stream,
                               filter_delay,
                               labels,
                               h_per_symbol):
        """
        按照“符号峰值位置”为中心，切 3 符号长度的窗口。
        h_per_symbol: 每个符号的 Rayleigh 系数 h_k
        """
        symbol_peak_positions = [
            filter_delay + i * self.SAMPLES_PER_SYMBOL
            for i in range(len(labels))
        ]

        impaired_windows, clean_windows, valid_labels = [], [], []
        h_true_windows, h_estimated_windows = [], []
        half_window = self.WINDOW_LENGTH_IN_SAMPLES // 2  # 3 符号窗口 -> 一侧 24 点

        for i, peak_pos in enumerate(symbol_peak_positions):
            start, end = peak_pos - half_window, peak_pos + half_window

            if start >= 0 and end < len(impaired_stream):
                # 切片 3 符号长度窗口
                impaired_windows.append(impaired_stream[start:end])
                clean_windows.append(ideal_baseband_stream[start:end])
                valid_labels.append(labels[i])

                # 对应符号的信道系数：直接用 h_per_symbol[i]
                h_true_windows.append(h_per_symbol[i])
                h_estimated_windows.append(h_per_symbol[i])  # 完美 CSI 情况

        impaired_windows = np.array(impaired_windows, dtype=np.complex64)
        clean_windows = np.array(clean_windows, dtype=np.complex64)
        valid_labels = np.array(valid_labels)
        h_true_windows = np.array(h_true_windows, dtype=np.complex64)
        h_estimated_windows = np.array(h_estimated_windows, dtype=np.complex64)

        return impaired_windows, clean_windows, valid_labels, h_true_windows, h_estimated_windows
    
    def visualize_waveforms(self, impaired_waveforms, clean_waveforms, labels, true_h):
        # 取前六个窗口
        selected_indices = range(6)

        fig, axes = plt.subplots(3, 2, figsize=(12, 10))

        for i, idx in enumerate(selected_indices):
            ax = axes[i // 2, i % 2]
            ax.plot(np.real(impaired_waveforms[idx]), label="Impaired Signal", color="r")
            ax.plot(np.real(clean_waveforms[idx]), label="Clean Signal", color="b")
            ax.set_title(
                f"Window {idx+1} - h_c: {np.real(true_h[idx]):.2f} + j{np.imag(true_h[idx]):.2f}"
            )
            ax.legend()
            ax.grid(True)

        plt.tight_layout()
        plt.show()

    def create_and_save_dataset(self, data_dict):
        """保存所有数据集组件。"""
        num_samples = len(data_dict["labels"])
        
        print("\n" + "-"*20 + " 保存基础数据集 " + "-"*20)
        for name, data in data_dict.items():
            # 复数数据：拆成 I/Q 两路保存
            if np.iscomplexobj(data):
                if data.ndim == 2:
                    # [N, L] -> [N, 2, L]
                    data_to_save = np.stack([np.real(data), np.imag(data)], axis=1)
                else:
                    # 其他维度，按你现在的策略统一处理
                    data_to_save = np.stack([np.real(data), np.imag(data)], axis=1)
            else:
                data_to_save = data

            path = os.path.join(self.OUTPUT_DIR, f"{name}.npy")
            np.save(path, data_to_save.astype(np.float32))
            print(f"  已保存: {path} (形状: {data_to_save.shape})")
        print("-" * 60)

    # ============= 核心修改点 3：generate 里改成“符号域乘 h 再成型” =============
    def generate(self):
        """
        核心生成流程：生成包含“符号域 Rayleigh 衰落”的基础数据集。
        流程：
            1) 生成 QPSK 符号序列 symbols
            2) 生成每符号信道 h_per_symbol，得到 faded_symbols = h * symbols
            3) 对 symbols / faded_symbols 分别进行 RRC 成型
            4) 以符号峰值为中心切 3 符号窗口
            5) 保存 clean/impaired/labels/true_h/estimated_h
        """
        print("\n开始生成基础数据集...")

        # 1. 生成 QPSK 符号和标签
        symbols, labels = self.generate_qpsk_symbols(self.NUM_SYMBOLS)

        all_impaired_windows, all_clean_windows, all_labels = [], [], []
        all_true_h, all_estimated_h = [], []

        # 2. 符号域施加瑞利衰落
        faded_symbols, h_per_symbol = self.apply_rayleigh_channel(symbols)

        # 3. 分别对“干净符号”和“衰落符号”做成型
        clean_stream, delay_chunk = self.pulse_shaping(symbols)
        faded_stream, _           = self.pulse_shaping(faded_symbols)

        # 4. 切片成窗口（使用符号级 h_per_symbol）
        impaired_w, clean_w, labels_w, true_h_w, est_h_w = self.sliding_window_slicing(
            clean_stream, faded_stream, delay_chunk, labels, h_per_symbol
        )
        
        all_impaired_windows.append(impaired_w)
        all_clean_windows.append(clean_w)
        all_labels.append(labels_w)
        all_true_h.append(true_h_w)
        all_estimated_h.append(est_h_w)

        # 5. 汇总 + 保存
        final_data = {
            "impaired_waveforms": np.concatenate(all_impaired_windows, axis=0),
            "clean_waveforms":    np.concatenate(all_clean_windows, axis=0),
            "labels":             np.concatenate(all_labels, axis=0),
            "true_h":             np.concatenate(all_true_h, axis=0),
            "estimated_h":        np.concatenate(all_estimated_h, axis=0)
        }

        # 可视化随机挑选的六个窗口
        self.visualize_waveforms(
            final_data["impaired_waveforms"],
            final_data["clean_waveforms"],
            final_data["labels"],
            final_data["true_h"]
        )

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
