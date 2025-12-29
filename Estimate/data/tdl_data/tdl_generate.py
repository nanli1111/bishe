import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma
import scipy.signal as signal
import os
import random

class UAVBaseDatasetGenerator:
    """
    生成一个“基础”数据集，用于 CNN 密集信道估计任务 (Vector Output)。
    特性：
    1. 3GPP TDL-D (LoS) 恶劣环境魔改版
    2. 28GHz + 50m/s (快衰落)
    3. 1MHz + 600ns (ISI)
    4. True H: 向量 (9) -> 学习轨迹
    5. Est H : 稀疏向量 (15, 仅中心有值) -> 辅助输入
    6. Labels: 标量 (1, 仅中心符号) -> 【本次修改】
    """
    def __init__(self):
        # --- 基本参数 ---
        self.NUM_SYMBOLS = int(5e5)
        self.SAMPLES_PER_SYMBOL = 16
        self.PULSE_SHAPING_FILTER_TAPS = 129
        self.PULSE_SHAPING_ROLLOFF = 0.25
        self.WINDOW_SIZE_IN_SYMBOLS = 5  # 每个样本窗口大小（符号数）
        
        # --- 3GPP TDL-D 信道参数 (恶劣版) ---
        self.TDL_D_NORM_DELAYS = np.array([0.0000, 0.0000, 0.0350, 0.6120, 1.3630])
        self.TDL_D_POWERS_DB = np.array([-0.2, -3.5, -7.8, -11.0, -15.8])
        self.DELAY_SPREAD_SEC = 1500e-9 
        self.RICIAN_K_VALUES = [1, 1.5, 3, 6, 10]  # dB
        
        # --- 场景参数 ---
        self.CARRIER_FREQ = 28e9  
        self.UAV_SPEED_MPS = 25.0 
        self.SYMBOL_RATE = 4e5    
        self.FS = self.SYMBOL_RATE * self.SAMPLES_PER_SYMBOL
        
        # 信道估计误差范围
        self.CHANNEL_ESTIMATION_NMSE_DB_RANGE = [-10, -20] 
        self.CHUNK_SIZE_IN_SYMBOLS = 5000 

        # --- 自动计算 ---
        self.WINDOW_LENGTH_IN_SAMPLES = self.WINDOW_SIZE_IN_SYMBOLS * self.SAMPLES_PER_SYMBOL
        self.OUTPUT_DIR = r"Estimate/data/tdl_data" 
        os.makedirs(self.OUTPUT_DIR, exist_ok=True)
        
        self.print_summary()

    def print_summary(self):
        print("="*60)
        print("UAV 稀疏导频数据集 (Center Label Only)")
        print("="*60)
        print(f"窗口大小: {self.WINDOW_SIZE_IN_SYMBOLS} symbols")
        print(f"True H: 完整的 9 点向量 (Dense Label)")
        print(f"Est H : 仅保留中心的稀疏向量 (Sparse Input)")
        print(f"Labels: 仅保留中心符号索引 (Scalar Label)")
        print(f"信道环境: TDL-D Strong Reflection (DS=600ns)")
        print("="*60)
    
    def calc_max_doppler(self):
        return self.UAV_SPEED_MPS * self.CARRIER_FREQ / 3e8

    def generate_qpsk_symbols(self, num_symbols):
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
                k_lin = 10**(k_factor_db / 10.0)
                t = np.arange(num_samples) / self.FS
                h_los = np.exp(1j * 2 * np.pi * fd * 0.7 * t) 
                h_curr = p_curr * (np.sqrt(k_lin/(k_lin+1))*h_los + np.sqrt(1/(k_lin+1))*h_diffuse)
            else:
                h_curr = p_curr * h_diffuse

            delay = tap_indices[i]
            if delay == 0:
                output_signal += h_curr * baseband_stream
                main_path_h += h_curr 
            elif delay < num_samples:
                delayed = np.zeros_like(baseband_stream)
                delayed[delay:] = baseband_stream[:-delay]
                output_signal += h_curr * delayed
            
        return output_signal, main_path_h

    def apply_pa_nonlinearity(self, signal_stream):
        mag = np.abs(signal_stream)
        phase = np.divide(signal_stream, mag, out=np.zeros_like(signal_stream), where=mag!=0)
        new_mag = mag / (1 + (mag / 2.0)**(6.0))**(1 / 6.0)
        return (new_mag * phase).astype(np.complex64)

    def generate_channel_estimate(self, true_h_stream, nmse_db):
        h_power = np.mean(np.abs(true_h_stream)**2)
        nmse_linear = 10**(nmse_db / 10.0)
        error_std = np.sqrt(h_power * nmse_linear / 2)
        error_noise = error_std * (np.random.randn(len(true_h_stream)) + 1j * np.random.randn(len(true_h_stream)))
        return (true_h_stream + error_noise).astype(np.complex64)

    def sliding_window_slicing(self, ideal_baseband_stream, impaired_stream, filter_delay, labels, true_h_stream, estimated_h_stream):
        """
        【修改点】
        1. h_true_vectors: 保留窗口内完整的 15 个点 (Dense Label)
        2. h_est_vectors : 截取后进行 Masking，只保留中心点 (Sparse Input)
        3. valid_labels  : 【修改】只取中心点符号索引 (Scalar Label)
        """
        symbol_peak_positions = [filter_delay + i * self.SAMPLES_PER_SYMBOL for i in range(len(labels))]
        
        impaired_windows, clean_windows = [], []
        h_true_vectors, h_est_vectors = [], []
        valid_labels_scalar = [] # 修改：保存标量标签
        
        half_window_sym = self.WINDOW_SIZE_IN_SYMBOLS // 2
        half_window_sample = self.WINDOW_LENGTH_IN_SAMPLES // 2

        for i, peak_pos in enumerate(symbol_peak_positions):
            sample_start = peak_pos - half_window_sample
            sample_end = peak_pos + half_window_sample
            sym_start_idx = i - half_window_sym
            sym_end_idx = i + half_window_sym + 1 
            
            if (sample_start >= 0 and sample_end < len(impaired_stream) and 
                sym_start_idx >= 0 and sym_end_idx <= len(symbol_peak_positions)):
                
                # 1. 波形切片
                impaired_windows.append(impaired_stream[sample_start:sample_end])
                clean_windows.append(ideal_baseband_stream[sample_start:sample_end])
                
                # 2. 截取 Labels (Center Only)
                # i 就是当前窗口中心的符号索引
                valid_labels_scalar.append(labels[i])
                
                # 3. 截取 H 向量
                indices = symbol_peak_positions[sym_start_idx : sym_end_idx]
                
                # 真实信道 (完整)
                h_true_dense = true_h_stream[indices]
                h_true_vectors.append(h_true_dense)
                
                # 估计信道 (Masking)
                h_est_chunk = estimated_h_stream[indices].copy()
                
                # 构建稀疏导频 (只保留中心点, index=7)
                h_est_sparse = np.zeros_like(h_est_chunk)
                center_idx = self.WINDOW_SIZE_IN_SYMBOLS // 2
                h_est_sparse[center_idx] = h_est_chunk[center_idx]
                
                h_est_vectors.append(h_est_sparse)
        
        return (np.array(impaired_windows, dtype=np.complex64), 
                np.array(clean_windows, dtype=np.complex64), 
                np.array(valid_labels_scalar, dtype=int),     # (N,) int
                np.array(h_true_vectors, dtype=np.complex64), # (N, 15) Dense
                np.array(h_est_vectors, dtype=np.complex64))  # (N, 15) Sparse

    def create_and_save_dataset(self, data_dict):
        num_samples = len(data_dict["impaired_waveforms"])
        indices = np.random.permutation(num_samples)
        
        print("\n" + "-"*20 + " 保存基础数据集 " + "-"*20)
        for name, data in data_dict.items():
            shuffled_data = data[indices]
            
            if np.iscomplexobj(shuffled_data):
                # 统一转为 (N, 2, L)
                if shuffled_data.ndim == 2: 
                    data_to_save = np.stack([np.real(shuffled_data), np.imag(shuffled_data)], axis=1)
                else:
                    data_to_save = shuffled_data
            else:
                # Labels (N,) 直接保存
                data_to_save = shuffled_data
            
            path = os.path.join(self.OUTPUT_DIR, f"{name}.npy")
            np.save(path, data_to_save) 
            print(f"  已保存: {path} (Shape: {data_to_save.shape})")
        print("-" * 60)
        
    def generate(self):
        print("\n开始生成密集向量数据集...")
        symbols, labels = self.generate_qpsk_symbols(self.NUM_SYMBOLS)
        all_imp, all_clean, all_lbl, all_h_true, all_h_est = [], [], [], [], []

        num_symbols_per_condition = self.NUM_SYMBOLS // len(self.RICIAN_K_VALUES)
        
        for i, k_val in enumerate(self.RICIAN_K_VALUES):
            print(f"  -> 处理 K={k_val} dB ...")
            start_idx = i * num_symbols_per_condition
            end_idx = start_idx + num_symbols_per_condition
            num_chunks = (end_idx - start_idx) // self.CHUNK_SIZE_IN_SYMBOLS

            for j in range(num_chunks):
                c_start = start_idx + j * self.CHUNK_SIZE_IN_SYMBOLS
                c_end = c_start + self.CHUNK_SIZE_IN_SYMBOLS
                
                s_chunk = symbols[c_start:c_end]
                l_chunk = labels[c_start:c_end]
                
                clean_s, delay = self.pulse_shaping(s_chunk)
                faded_s, true_h = self.apply_tdl_channel(clean_s, k_val)
                nonlinear_s = self.apply_pa_nonlinearity(faded_s)
                
                nmse = np.random.uniform(*self.CHANNEL_ESTIMATION_NMSE_DB_RANGE)
                est_h = self.generate_channel_estimate(true_h, nmse)
                
                imp_w, cln_w, lbl_s, h_t_vec, h_e_vec = self.sliding_window_slicing(
                    clean_s, nonlinear_s, delay, l_chunk, true_h, est_h
                )
                
                all_imp.append(imp_w)
                all_clean.append(cln_w)
                all_lbl.append(lbl_s)
                all_h_true.append(h_t_vec)
                all_h_est.append(h_e_vec)
        
        final_data = {
            "impaired_waveforms": np.concatenate(all_imp, axis=0),
            "clean_waveforms": np.concatenate(all_clean, axis=0),
            "labels": np.concatenate(all_lbl, axis=0), # (N,)
            "true_h": np.concatenate(all_h_true, axis=0),
            "estimated_h": np.concatenate(all_h_est, axis=0)
        }

        self.create_and_save_dataset(final_data)
        return final_data
    

def visualize_combined(data_dict, num_samples=3, save_dir="Estimate/data/tdl_data/vis"):
    """
    可视化：波形 + H 轨迹 (显示导频点)
    """
    import matplotlib.pyplot as plt
    os.makedirs(save_dir, exist_ok=True)
    
    y_imp = data_dict['impaired_waveforms']
    y_cln = data_dict['clean_waveforms']
    h_vec = data_dict['true_h']
    h_est = data_dict['estimated_h'] # Sparse
    labels = data_dict['labels'] # (N,)
    
    total = len(h_vec)
    indices = np.random.choice(total, num_samples, replace=False)
    
    fig, axes = plt.subplots(num_samples, 4, figsize=(24, 4 * num_samples))
    plt.subplots_adjust(wspace=0.25, hspace=0.4)
    
    if num_samples == 1: axes = np.expand_dims(axes, axis=0)
    
    for i, idx in enumerate(indices):
        sig_i = y_imp[idx]
        sig_c = y_cln[idx]
        h = h_vec[idx]
        h_e = h_est[idx]
        lbl = labels[idx] # Scalar
        
        # 还原复数
        if sig_i.ndim == 2:
            real_i, imag_i = sig_i[0], sig_i[1]
            real_c, imag_c = sig_c[0], sig_c[1]
        else:
            real_i, imag_i = np.real(sig_i), np.imag(sig_i)
            real_c, imag_c = np.real(sig_c), np.imag(sig_c)
            
        # Waveforms
        axes[i, 0].plot(real_c, 'g--', alpha=0.5); axes[i, 0].plot(real_i, 'b-')
        axes[i, 0].set_title(f"I-Channel (L={lbl})")
        
        axes[i, 1].plot(imag_c, 'g--', alpha=0.5); axes[i, 1].plot(imag_i, 'r-')
        axes[i, 1].set_title(f"Q-Channel")
        
        # H Trajectory
        axes[i, 2].plot(np.real(h), np.imag(h), 'b.-', alpha=0.4, label='True H')
        # 绘制导频点 (非零点)
        nz = np.where(np.abs(h_e) > 1e-9)[0]
        axes[i, 2].scatter(np.real(h_e[nz]), np.imag(h_e[nz]), c='r', s=100, marker='*', zorder=5, label='Center Pilot')
        axes[i, 2].set_title("H Trajectory & Center Pilot")
        axes[i, 2].axis('equal'); axes[i, 2].legend()
        
        # H Magnitude
        axes[i, 3].plot(np.abs(h), 'k-', label='True |h|')
        axes[i, 3].scatter(nz, np.abs(h_e[nz]), c='r', s=80, marker='*', label='Pilot')
        axes[i, 3].set_title("H Magnitude")
        axes[i, 3].legend()
        
    plt.tight_layout()
    save_path = os.path.join(save_dir, "sparse_pilot_vis.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"可视化已保存至: {save_path}")

if __name__ == "__main__":
    gen = UAVBaseDatasetGenerator()
    data = gen.generate()
    visualize_combined(data, num_samples=4)