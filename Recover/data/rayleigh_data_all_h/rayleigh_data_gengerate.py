import numpy as np
import matplotlib.pyplot as plt
import os

class UAVBaseDatasetGenerator:
    """
    生成一个“基础”数据集，仅包含期望信号及其核心损伤（瑞利衰落）。
    这个数据集将作为后续训练中动态添加干扰和噪声的“原料”。
    """
    def __init__(self):
        # --- 基本参数 ---
        self.NUM_SYMBOLS = 500000  # 符号总数
        self.SAMPLES_PER_SYMBOL = 16  # 每个符号采样点数
        self.PULSE_SHAPING_FILTER_TAPS = 129  # 滤波器长度
        self.PULSE_SHAPING_ROLLOFF = 0.25  # 滤波器滚降因子
        self.WINDOW_SIZE_IN_SYMBOLS = 3  # 窗口大小（符号数）
        
        # --- 自动计算的参数 ---
        self.WINDOW_LENGTH_IN_SAMPLES = self.WINDOW_SIZE_IN_SYMBOLS * self.SAMPLES_PER_SYMBOL
        self.OUTPUT_DIR = r"data/rayleigh_data_all_h"  # 输出目录
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        self.print_summary()

    def print_summary(self):
        print("="*60)
        print("城市UAV集群通信 - 基础数据集生成器")
        print(f"总符号数: {self.NUM_SYMBOLS}")
        print(f"每符号采样数: {self.SAMPLES_PER_SYMBOL}")
        print(f"输出目录: {self.OUTPUT_DIR}")
        print("="*60)
    
    def generate_qpsk_symbols(self, num_symbols):
        """生成 QPSK 符号序列和标签"""
        labels = np.random.randint(0, 4, num_symbols)
        # 映射表: 0->(1+j), 1->(-1+j), 2->(-1-j), 3->(1-j)
        # 归一化能量
        qpsk_mapping = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        symbols = qpsk_mapping[labels]
        return symbols.astype(np.complex64), labels

    def create_rrc_filter(self):
        """生成 RRC 滤波器系数"""
        t = np.arange(self.PULSE_SHAPING_FILTER_TAPS) - (self.PULSE_SHAPING_FILTER_TAPS - 1) / 2.0
        t /= self.SAMPLES_PER_SYMBOL
        beta = self.PULSE_SHAPING_ROLLOFF
        
        h = np.zeros_like(t, dtype=float)
        
        # t = 0
        idx_zero = np.where(t == 0)[0]
        if idx_zero.size > 0:
            h[idx_zero] = 1.0 - beta + (4 * beta / np.pi)
            
        # t = +/- 1/(4*beta)
        idx_beta = np.where(np.abs(np.abs(t) - 1/(4*beta)) < 1e-9)[0]
        if idx_beta.size > 0:
            h[idx_beta] = (beta / np.sqrt(2)) * (
                (1 + 2/np.pi) * np.sin(np.pi/(4*beta)) + (1 - 2/np.pi) * np.cos(np.pi/(4*beta))
            )
            
        # t = other
        idx_general = np.where((t != 0) & (np.abs(np.abs(t) - 1/(4*beta)) > 1e-9))[0]
        t_gen = t[idx_general]
        num = np.sin(np.pi * t_gen * (1 - beta)) + 4 * beta * t_gen * np.cos(np.pi * t_gen * (1 + beta))
        den = np.pi * t_gen * (1 - (4 * beta * t_gen)**2)
        h[idx_general] = num / den
        
        return h

    def pulse_shaping(self, symbols):
        """RRC 成型滤波"""
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
        """生成瑞利信道系数"""
        h = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        return h.astype(np.complex64)

    def apply_rayleigh_channel(self, baseband_stream):
        """应用平坦瑞利衰落"""
        num_symbols = int(np.ceil(len(baseband_stream) / self.SAMPLES_PER_SYMBOL))
        
        # 1. 符号级信道
        h_symbol_rate = self._generate_rayleigh_h(num_symbols)
        
        # 2. 采样级信道 (Block Fading)
        h_sample_rate = np.repeat(h_symbol_rate, self.SAMPLES_PER_SYMBOL)
        h_sample_rate = h_sample_rate[:len(baseband_stream)]
        
        # 3. 衰落
        faded_stream = baseband_stream * h_sample_rate
        
        return faded_stream.astype(np.complex64), h_sample_rate

    def sliding_window_slicing(self, clean_stream, impaired_stream, filter_delay, labels, true_h_stream, estimated_h_stream):
        """切片，同时保存 true_h 和 estimated_h"""
        peak_positions = [filter_delay + i * self.SAMPLES_PER_SYMBOL for i in range(len(labels))]
        half_win = self.WINDOW_LENGTH_IN_SAMPLES // 2
        
        impaired_wins = []
        clean_wins = []
        valid_labels = []
        true_h_wins = []
        est_h_wins = [] # 新增：保存估计信道
        
        for i, peak in enumerate(peak_positions):
            start = peak - half_win
            end = peak + half_win
            
            if start >= 0 and end <= len(impaired_stream):
                impaired_wins.append(impaired_stream[start:end])
                clean_wins.append(clean_stream[start:end])
                valid_labels.append(labels[i])
                # 保存整段信道
                true_h_wins.append(true_h_stream[start:end])
                # 保存整段估计信道
                est_h_wins.append(estimated_h_stream[start:end])
        
        return (np.array(impaired_wins, dtype=np.complex64),
                np.array(clean_wins, dtype=np.complex64),
                np.array(valid_labels, dtype=np.int32),
                np.array(true_h_wins, dtype=np.complex64),
                np.array(est_h_wins, dtype=np.complex64))

    def visualize_waveforms(self, impaired_waveforms, clean_waveforms, labels, true_h):
        """可视化前 6 个样本"""
        indices = range(min(6, len(impaired_waveforms)))
        fig, axes = plt.subplots(3, 2, figsize=(12, 10))
        
        for i, idx in enumerate(indices):
            ax = axes[i // 2, i % 2]
            
            ax.plot(np.real(impaired_waveforms[idx]), label="Faded (I)", color="r", alpha=0.7)
            ax.plot(np.real(clean_waveforms[idx]), label="Clean (I)", color="b", linestyle='--')
            
            # 修复：格式化 h 的显示
            h_val = true_h[idx]
            if h_val.ndim > 0:
                center_h = h_val[len(h_val)//2]
            else:
                center_h = h_val
                
            ax.set_title(f"Win {idx} (Label {labels[idx]}) | h ~ {center_h.real:.2f}+{center_h.imag:.2f}j")
            ax.legend(loc='upper right', fontsize='small')
            ax.grid(True, linestyle=':', alpha=0.6)

        plt.tight_layout()
        plt.show()

    def create_and_save_dataset(self, data_dict):
        """保存为 .npy"""
        print("\n" + "-"*20 + " 保存数据集 " + "-"*20)
        
        for name, data in data_dict.items():
            path = os.path.join(self.OUTPUT_DIR, f"{name}.npy")
            
            if np.iscomplexobj(data):
                if data.ndim == 2: 
                    data_save = np.stack([np.real(data), np.imag(data)], axis=1)
                elif data.ndim == 1: 
                    data_save = np.stack([np.real(data), np.imag(data)], axis=1)
                else:
                    data_save = data
            else:
                data_save = data
            
            np.save(path, data_save)
            print(f"  Saved: {name:<20} shape: {data_save.shape}")
            
        print("-" * 60)

    def generate(self):
        print("\n开始生成基础数据集...")

        # 1. 生成符号
        symbols, labels = self.generate_qpsk_symbols(self.NUM_SYMBOLS)

        # 2. 成型滤波
        clean_stream, delay = self.pulse_shaping(symbols)

        # 3. 衰落信道
        faded_stream, true_h_stream = self.apply_rayleigh_channel(clean_stream)
        
        # 在这里定义估计信道 (目前假设完美估计，日后可在此处加估计误差)
        estimated_h_stream = true_h_stream.copy()

        # 4. 切片
        impaired_w, clean_w, labels_w, true_h_w, est_h_w = self.sliding_window_slicing(
            clean_stream, faded_stream, delay, labels, true_h_stream, estimated_h_stream
        )

        final_data = {
            "impaired_waveforms": impaired_w,
            "clean_waveforms": clean_w,
            "labels": labels_w,
            "true_h": true_h_w,       # [N, L]
            "estimated_h": est_h_w,   # [N, L]
        }

        # 5. 可视化
        self.visualize_waveforms(
            final_data["impaired_waveforms"], 
            final_data["clean_waveforms"], 
            final_data["labels"], 
            final_data["true_h"]
        )

        # 6. 保存
        self.create_and_save_dataset(final_data)
        
        print("\n生成完毕！")

def main():
    gen = UAVBaseDatasetGenerator()
    gen.generate()

if __name__ == "__main__":
    main()