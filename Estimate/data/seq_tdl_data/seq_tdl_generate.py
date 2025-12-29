import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import scipy.signal as signal
import os
import random

class UAVPilotDatasetGenerator:
    """
    【采样级真值版】UAV 导频插值数据集生成器
    特性：
    1. 物理参数: 1Mbaud, 600ns TDL-D, 50m/s (快衰落 + 强ISI)
    2. True H: 【采样点级】分辨率 (Sequence_Len * SPS)，捕捉符号内的微细变化
    3. Estimated H: 【采样点级稀疏】仅在头尾符号的中心采样位置有值，其余为0
    """
    def __init__(self):
        # --- 基础配置 ---
        self.NUM_SYMBOLS = int(4e5) 
        self.SAMPLES_PER_SYMBOL = 16 # SPS
        self.SEQUENCE_LENGTH = 128   # 符号数
        
        # 自动计算总采样点长度: 128 * 16 = 2048
        self.TOTAL_SAMPLES_PER_SEQ = self.SEQUENCE_LENGTH * self.SAMPLES_PER_SYMBOL
        
        # --- 物理层参数 ---
        self.PULSE_SHAPING_FILTER_TAPS = 129
        self.PULSE_SHAPING_ROLLOFF = 0.25
        
        # --- 3GPP TDL-D (恶劣 LoS) ---
        self.TDL_D_NORM_DELAYS = np.array([0.0000, 0.0000, 0.0350, 0.6120, 1.3630])
        self.TDL_D_POWERS_DB = np.array([-0.2, -5.5, -10.8, -13.0, -15.8])
        self.DELAY_SPREAD_SEC = 1500e-9 
        self.RICIAN_K_VALUES = [0, 0.5, 1, 1.5, 3] 
        
        # --- 场景参数 ---
        self.CARRIER_FREQ = 28e9 
        self.UAV_SPEED_MPS = 50.0 
        self.SYMBOL_RATE = 4e5     
        self.FS = self.SYMBOL_RATE * self.SAMPLES_PER_SYMBOL
        
        # 导频处的估计误差
        self.PILOT_NMSE_DB_RANGE = [-25, -15] 
        
        self.OUTPUT_DIR = r"Estimate/data/seq_tdl_data"
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        self.print_summary()

    def print_summary(self):
        print("="*60)
        print("UAV 采样级导频数据集 (Sample-Level True H)")
        print("="*60)
        print(f"序列符号数: {self.SEQUENCE_LENGTH} symbols")
        print(f"序列采样数: {self.TOTAL_SAMPLES_PER_SEQ} samples (128 * 16)")
        print(f"-"*20)
        print(f"1. Rx Waveform: (Batch, {self.TOTAL_SAMPLES_PER_SEQ}) - 接收波形")
        print(f"2. True H:      (Batch, {self.TOTAL_SAMPLES_PER_SEQ}) - 【完整】采样级信道")
        print(f"3. Sparse H:    (Batch, {self.TOTAL_SAMPLES_PER_SEQ}) - 【稀疏】仅头尾有值的含噪导频")
        print(f"-"*20)
        print(f"输出目录: {self.OUTPUT_DIR}")
        print("="*60)
    
    def calc_max_doppler(self):
        return self.UAV_SPEED_MPS * self.CARRIER_FREQ / 3e8

    def generate_qpsk_symbols(self, num_symbols):
        num_symbols = int(num_symbols)
        labels = np.random.randint(0, 4, num_symbols)
        qpsk_mapping = {0: 1 + 1j, 1: -1 + 1j, 2: -1 - 1j, 3: +1 - 1j}
        symbols = np.array([qpsk_mapping[label] for label in labels], dtype=np.complex64)
        return symbols / 1.0, labels

    def create_rrc_filter(self):
        t = np.arange(self.PULSE_SHAPING_FILTER_TAPS) - (self.PULSE_SHAPING_FILTER_TAPS - 1) / 2.0
        t /= self.SAMPLES_PER_SYMBOL
        beta = self.PULSE_SHAPING_ROLLOFF
        h = np.zeros_like(t, dtype=float)
        idx_zero = np.where(t == 0)[0]
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
        filter_delay = len(rrc_filter) // 2
        return baseband_stream.astype(np.complex64), filter_delay

    def _generate_jakes_noise(self, num_samples, fd, fs):
        noise = (np.random.randn(num_samples) + 1j * np.random.randn(num_samples)) / np.sqrt(2)
        f = np.fft.fftfreq(num_samples, 1/fs)
        with np.errstate(divide='ignore', invalid='ignore'):
            psd = 1 / np.sqrt(1 - (f / fd)**2)
        psd[np.abs(f) >= fd] = 0.0; psd[np.isinf(psd)] = 0.0; psd[np.isnan(psd)] = 0.0; psd[0] = 0.0
        psd = psd / np.sum(psd) * num_samples if np.sum(psd) > 0 else psd
        colored_noise = np.fft.ifft(np.fft.fft(noise) * np.sqrt(psd))
        return colored_noise / np.std(colored_noise)

    def apply_tdl_channel(self, baseband_stream, k_factor_db):
        num_samples = len(baseband_stream)
        fd = self.calc_max_doppler()
        delays_sec = self.TDL_D_NORM_DELAYS * self.DELAY_SPREAD_SEC
        tap_indices = np.round(delays_sec * self.FS).astype(int)
        tap_powers = 10**(self.TDL_D_POWERS_DB / 10.0); tap_powers /= np.sum(tap_powers)
        
        output_signal = np.zeros(num_samples, dtype=np.complex64)
        main_path_h = np.zeros(num_samples, dtype=np.complex64) 
        
        for i in range(len(tap_indices)):
            h_diffuse = self._generate_jakes_noise(num_samples, fd, self.FS)
            p_curr = np.sqrt(tap_powers[i])
            if i == 0: 
                k_lin = 10**(k_factor_db/10.0)
                t = np.arange(num_samples)/self.FS
                h_los = np.exp(1j*2*np.pi*fd*0.7*t)
                h_curr = p_curr*(np.sqrt(k_lin/(k_lin+1))*h_los + np.sqrt(1/(k_lin+1))*h_diffuse)
            else:
                h_curr = p_curr * h_diffuse
            
            delay = tap_indices[i]
            if delay == 0: 
                output_signal += h_curr * baseband_stream
                main_path_h += h_curr # 叠加所有零延迟径到 True H
            elif delay < num_samples:
                delayed = np.zeros_like(baseband_stream)
                delayed[delay:] = baseband_stream[:-delay]
                output_signal += h_curr * delayed
        
        return output_signal, main_path_h

    def apply_pa_nonlinearity(self, signal_stream):
        mag = np.abs(signal_stream)
        phase = np.divide(signal_stream, mag, out=np.zeros_like(signal_stream), where=mag!=0)
        new_mag = mag / (1 + (mag/2.0)**(6.0))**(1/6.0) 
        return (new_mag * phase).astype(np.complex64)

    def sequence_slicing_sample_level(self, impaired_stream, labels, true_h_stream, filter_delay, nmse_db):
        """
        【核心修改】生成采样点级的序列 (Sample-Level Sequences)
        """
        total_symbols = len(labels)
        num_seqs = total_symbols // self.SEQUENCE_LENGTH
        
        rx_seqs = [] 
        h_true_seqs = [] 
        h_est_seqs = [] 
        label_seqs = [] 
        
        # 导频在采样点层面的相对位置 (假设在符号中心)
        # 第一个符号中心
        pilot_idx_1 = self.SAMPLES_PER_SYMBOL // 2 
        # 最后一个符号中心
        pilot_idx_2 = (self.SEQUENCE_LENGTH - 1) * self.SAMPLES_PER_SYMBOL + (self.SAMPLES_PER_SYMBOL // 2)
        
        for i in range(num_seqs):
            sym_start = i * self.SEQUENCE_LENGTH
            sym_end = sym_start + self.SEQUENCE_LENGTH
            
            # --- 1. 确定采样点范围 ---
            sample_start = sym_start * self.SAMPLES_PER_SYMBOL + filter_delay
            sample_end = sym_end * self.SAMPLES_PER_SYMBOL + filter_delay
            
            # 边界检查
            if sample_end > len(impaired_stream): break
            if sample_end > len(true_h_stream): break
            
            # --- 2. 提取数据 ---
            
            # Rx 波形 (完整连续切片) -> Shape (2048,)
            rx_chunk = impaired_stream[sample_start : sample_end]
            
            # True H (完整连续切片, 采样级) -> Shape (2048,)
            h_true_chunk = true_h_stream[sample_start : sample_end]
            
            # --- 3. 生成稀疏导频 (Sparse Pilots) ---
            
            # 初始化全 0 的稀疏向量，长度与 True H 一致 (2048,)
            h_sparse = np.zeros_like(h_true_chunk)
            
            # 提取这两个特定采样点的真实 H
            val_p1 = h_true_chunk[pilot_idx_1]
            val_p2 = h_true_chunk[pilot_idx_2]
            
            # 计算噪声标准差 (基于该片段的平均能量)
            h_power = np.mean(np.abs(h_true_chunk)**2)
            error_std = np.sqrt(h_power * (10**(nmse_db / 10.0)) / 2)
            
            # 添加高斯噪声
            noise_p1 = error_std * (np.random.randn() + 1j * np.random.randn())
            noise_p2 = error_std * (np.random.randn() + 1j * np.random.randn())
            
            # 填入稀疏向量
            h_sparse[pilot_idx_1] = val_p1 + noise_p1
            h_sparse[pilot_idx_2] = val_p2 + noise_p2
            
            # Tx Labels (辅助信息)
            l_chunk = labels[sym_start:sym_end]
            
            rx_seqs.append(rx_chunk)
            h_true_seqs.append(h_true_chunk)
            h_est_seqs.append(h_sparse)
            label_seqs.append(l_chunk)
            
        return (np.array(rx_seqs, dtype=np.complex64), 
                np.array(h_true_seqs, dtype=np.complex64),
                np.array(h_est_seqs, dtype=np.complex64),
                np.array(label_seqs))

    def create_and_save_dataset(self, data_dict):
        total_samples = len(data_dict["true_h"])
        split_idx = int(total_samples * 0.8)
        
        indices = np.arange(total_samples)
        train_indices = np.random.permutation(indices[:split_idx])
        test_indices = indices[split_idx:]
        
        print("\n" + "-"*20 + " 保存数据集 " + "-"*20)
        
        def save_part(indices, prefix):
            for name, data in data_dict.items():
                part_data = data[indices]
                # 复数转实数: (N, 2048) -> (N, 2048, 2)
                if np.iscomplexobj(part_data):
                    part_data = np.stack([np.real(part_data), np.imag(part_data)], axis=-1)
                
                path = os.path.join(self.OUTPUT_DIR, f"{prefix}_{name}.npy")
                np.save(path, part_data.astype(np.float32))
                print(f"  [{prefix}] {name}: {part_data.shape}")

        save_part(train_indices, "train")
        save_part(test_indices, "test")
        print("-" * 60)

    def generate(self):
        print("\n开始生成采样级导频数据集...")
        symbols, labels = self.generate_qpsk_symbols(self.NUM_SYMBOLS)
        
        all_rx, all_h_true, all_h_est, all_lbl = [], [], [], []
        
        for k_val in self.RICIAN_K_VALUES:
            print(f"  -> 处理 K={k_val} dB ...")
            
            num_sym_k = self.NUM_SYMBOLS // len(self.RICIAN_K_VALUES)
            idx_start = int(self.RICIAN_K_VALUES.index(k_val) * num_sym_k)
            idx_end = idx_start + num_sym_k
            
            s_chunk = symbols[idx_start:idx_end]
            l_chunk = labels[idx_start:idx_end]
            
            clean, delay = self.pulse_shaping(s_chunk)
            faded, true_h = self.apply_tdl_channel(clean, k_val)
            nonlinear = self.apply_pa_nonlinearity(faded)
            
            # 随机导频质量
            nmse_db = np.random.uniform(self.PILOT_NMSE_DB_RANGE[0], self.PILOT_NMSE_DB_RANGE[1])
            
            # 使用新的采样级切片函数
            rx, h_t, h_e, lbl = self.sequence_slicing_sample_level(nonlinear, l_chunk, true_h, delay, nmse_db)
            
            all_rx.append(rx)
            all_h_true.append(h_t)
            all_h_est.append(h_e)
            all_lbl.append(lbl)
            
        final_data = {
            "rx_waveforms": np.concatenate(all_rx, axis=0), 
            "true_h": np.concatenate(all_h_true, axis=0),     # (N, 2048) 完整
            "sparse_h": np.concatenate(all_h_est, axis=0),    # (N, 2048) 稀疏
            "labels": np.concatenate(all_lbl, axis=0)         # (N, 128)
        }
        
        self.create_and_save_dataset(final_data)
        return final_data

def visualize_sample_level(data_dict, num_seqs=1, save_dir="Estimate/data/seq_tdl_data/vis"):
    """
    可视化采样级 H：
    1. 左图：复平面轨迹 (IQ Trajectory) - 展示相位旋转
    2. 右图：时域幅度 (Magnitude) - 展示衰落深度
    """
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    h_true = data_dict['true_h']
    h_sparse = data_dict['sparse_h']
    
    indices = np.random.choice(len(h_true), num_seqs, replace=False)
    
    print(f"\n可视化采样级信道 (IQ轨迹 + 幅度)...")
    
    for i, idx in enumerate(indices):
        # 创建画布：1行2列
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        plt.suptitle(f"Sample-Level Channel Analysis (Batch Index {idx})", fontsize=14, y=0.98)
        
        # 准备数据
        ht = h_true[idx]   # 完整真实信道 (Complex)
        hs = h_sparse[idx] # 稀疏导频 (Complex)
        
        # 找到导频的非零索引
        nz_idx = np.where(np.abs(hs) > 1e-9)[0]
        
        # ==========================================
        # 子图 1: 复平面轨迹 (Constellation / IQ)
        # ==========================================
        # 绘制完整轨迹
        ax1.plot(ht.real, ht.imag, 'b-', linewidth=1, alpha=0.6, label='True H Trajectory')
        # 绘制起点 (辅助判断旋转方向)
        ax1.plot(ht.real[0], ht.imag[0], 'kD', markersize=6, label='Start Point')
        # 绘制导频位置
        ax1.plot(hs.real[nz_idx], hs.imag[nz_idx], 'r*', markersize=12, zorder=5, label='Noisy Pilots')
        
        ax1.set_title("Channel H Trajectory (IQ Plane)")
        ax1.set_xlabel("Real (In-Phase)")
        ax1.set_ylabel("Imag (Quadrature)")
        ax1.grid(True, linestyle=':', alpha=0.6)
        ax1.axis('equal') # 保持圆形比例，这对于观察旋转非常重要
        ax1.legend(loc='best')

        # ==========================================
        # 子图 2: 时域幅度 (Magnitude)
        # ==========================================
        ht_mag = np.abs(ht)
        hs_mag = np.abs(hs)
        
        # 绘制幅度曲线
        ax2.plot(ht_mag, 'b-', linewidth=1.5, alpha=0.8, label='True |h|')
        # 绘制导频幅度
        ax2.plot(nz_idx, hs_mag[nz_idx], 'r*', markersize=12, zorder=5, label='Pilots')
        
        ax2.set_title("Channel Magnitude over Time")
        ax2.set_xlabel("Sample Index (0-2047)")
        ax2.set_ylabel("Magnitude |h|")
        ax2.grid(True, linestyle=':', alpha=0.6)
        ax2.legend(loc='best')
        
        # 保存
        plt.tight_layout()
        save_path = os.path.join(save_dir, f"sample_level_traj_idx{idx}.png")
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  -> 图表已保存: {save_path}")

if __name__ == "__main__":
    gen = UAVPilotDatasetGenerator()
    data = gen.generate()
    # 临时修正 visualize 适配 complex 输入
    visualize_sample_level(data)