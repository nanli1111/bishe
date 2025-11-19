import os
import math
import torch
import numpy as np
from tqdm import tqdm

# 自定义模块
from model.unet import build_network
from ddrm_core import DDRM
from dataset.dataset import QPSKDataset
from test_fig import add_awgn_noise_np   # 使用你已有的加噪函数

import matplotlib.pyplot as plt
from matplotlib import rcParams
# 设置支持中文的字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


# ========= 工具函数 =========

def downsample(signal, SAMPLES_PER_SYMBOL=16):
    """从逐采样基带流中抽取每符号一个采样点"""
    return signal[::SAMPLES_PER_SYMBOL]


def decision_making(downsampled_signal, threshold=0):
    """QPSK 硬判决到 2bit (I,Q)"""
    real_part = np.real(downsampled_signal)
    imag_part = np.imag(downsampled_signal)

    decision = np.zeros((len(downsampled_signal), 2), dtype=int)
    for i in range(len(downsampled_signal)):
        if (real_part[i] > threshold) and (imag_part[i] > threshold):
            decision[i, 0] = 0
            decision[i, 1] = 0
        elif (real_part[i] < threshold) and (imag_part[i] > threshold):
            decision[i, 0] = 0
            decision[i, 1] = 1
        elif (real_part[i] < threshold) and (imag_part[i] < threshold):
            decision[i, 0] = 1
            decision[i, 1] = 1
        elif (real_part[i] > threshold) and (imag_part[i] < threshold):
            decision[i, 0] = 1
            decision[i, 1] = 0
    return decision


def calculate_ber(original_labels, predicted_labels):
    """按 bit 计算 BER"""
    predicted_labels = predicted_labels.astype(int)
    error_num_i = np.sum(original_labels[:, 0] != predicted_labels[:, 0])
    error_num_q = np.sum(original_labels[:, 1] != predicted_labels[:, 1])
    error_num = error_num_i + error_num_q
    ber = error_num / (len(original_labels) * 2)
    print(f"BER: {ber:.6e}")
    return ber


# ========= 核心：仅 DDRM 的信号恢复 =========

def model_test(snrDB_sample, n_steps, ddrm, rx_clean, h_np, batch_size=256, symbol_span=(24, 40)):
    """
    snrDB_sample: 采样域 SNR（外面已经减掉 10log10(16) 再传进来）
    ddrm: 已构建好的 DDRM 对象
    rx_clean: Rayleigh-only 接收波形 [N, 2, L] (numpy)
    h_np:     信道系数 [N, 2] (numpy)
    返回:
        model_signal: DDRM 恢复后的基带流（complex, 逐采样，一符号 16 点）
    """
    device = ddrm.device
    N, C, L = rx_clean.shape
    assert C == 2

    # ====== 1. 在 Rayleigh 波形上叠加 AWGN ======
    rx_noisy = add_awgn_noise_np(rx_clean, snrDB_sample)   # [N,2,L]

    # 为 DDRM.restore 计算噪声 std（和 add_awgn_noise_np 定义一致）
    signal_power = np.mean(rx_clean[:, 0, :] ** 2 + rx_clean[:, 1, :] ** 2)
    EbN0_linear = 10 ** (snrDB_sample / 10.0)
    N0 = signal_power / EbN0_linear
    noise_std = math.sqrt(N0 / 2.0)
    sigma_y = noise_std

    # ====== 2. 转成 torch Tensor，分 batch 喂给 DDRM.restore ======
    y_all = torch.from_numpy(rx_noisy).float().to(device)   # [N,2,L]
    h_all = torch.from_numpy(h_np).float().to(device)       # [N,2]

    all_recovered = []

    start_idx, end_idx = symbol_span
    L_symbol = end_idx - start_idx  # 这里应该是 16

    pbar = tqdm(range(0, N, batch_size), desc=f"SNR(sample)={snrDB_sample:.2f} dB")
    for i in pbar:
        y_batch = y_all[i:i + batch_size]   # [B,2,L]
        h_batch = h_all[i:i + batch_size]   # [B,2]

        x_rec = ddrm.restore(
            y=y_batch,
            h=h_batch,
            sigma_y=sigma_y,
            sigma_p=1,
            eta=0.0,
            init_from='y'   # 从观测 y 开始逆扩散
        )
        all_recovered.append(x_rec.cpu().numpy())

    all_recovered = np.concatenate(all_recovered, axis=0)   # [N,2,L]

    # ====== 3. 重构“逐采样”的复基带流，取中间 L_symbol 个采样点 ======
    model_signal = np.zeros((N * L_symbol,), dtype=np.complex64)

    for i in range(N):
        model_symbol = all_recovered[i, 0, start_idx:end_idx] + 1j * all_recovered[i, 1, start_idx:end_idx]
        model_signal[i * L_symbol:(i + 1) * L_symbol] = model_symbol

    return model_signal


# ========= 接收机（只有 DDRM 一条链路） =========

def matched_filter_decision(labels, snr_db_sample, n_steps, ddrm, rx_clean, h_np,
                            SAMPLES_PER_SYMBOL=16, batch_size=256, symbol_span=(24, 40)):
    """
    labels: [N_symbols, 2]，I/Q 比特标签
    snr_db_sample: 采样域 SNR（外部已减去 10log10(16)）
    ddrm: 已构建好的 DDRM
    rx_clean, h_np: 来自 QPSKDataset 的 Rayleigh-only 数据
    """
    recovered_signal = model_test(
        snrDB_sample=snr_db_sample,
        n_steps=n_steps,
        ddrm=ddrm,
        rx_clean=rx_clean,
        h_np=h_np,
        batch_size=batch_size,
        symbol_span=symbol_span
    )

    # 下采样：从逐采样基带流抽成逐符号
    downsampled_model_signal = downsample(recovered_signal, SAMPLES_PER_SYMBOL)

    # 硬判决
    predicted_model_labels = decision_making(downsampled_model_signal)

    # 计算 BER
    model_ber = calculate_ber(labels, predicted_model_labels)
    return model_ber


# ========= 画 BER 曲线 =========

def plot_ber_curve(model_bers, ref_bers, snr_range, save_path='ber_result/ber_curve.png'):
    plt.figure(figsize=(12, 8))

    snr_array = np.array(snr_range)

    # DDRM BER 曲线
    plt.semilogy(snr_array, model_bers, 'o-', label='DDRM 恢复')

    # 参考曲线（你给的那条）
    plt.semilogy(snr_array, ref_bers, 's-', label='参考曲线')

    # X 轴刻度
    xticks = np.arange(np.ceil(min(snr_array) / 2) * 2,
                       np.floor(max(snr_array) / 2) * 2 + 2, 2)
    plt.xticks(xticks)

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('QPSK 在 Rayleigh+AWGN 下：DDRM 与参考曲线对比')
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


# ========= 主程序 =========

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 50              # 扩散步数
    batch_size = 1024          # 可以根据显存调大一点提速
    symbol_span = (24, 40)    # 中间 16 个点
    SAMPLES_PER_SYMBOL = 16

    # ====== 1. 构建并加载 DDRM 模型（只做一次） ======
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)

    ckpt_path = fr'ddrm/ddrm_nak/results/best_model_epoch_with_n_steps{n_steps}.pth'
    print(f"加载模型权重: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # ====== 2. 加载 Rayleigh 数据集（只做一次） ======
    test_data = QPSKDataset(400000, 500000)
    rx_clean = test_data.y    # [N,2,L] numpy
    h_np = test_data.z        # [N,2]   numpy
    N_windows = rx_clean.shape[0]

    # ====== 3. 标签数据（和 QPSKDataset 使用同一段符号） ======
    label = np.load(r'F:\LJN\bishe\bishe\data\rayleigh_data\labels.npy')
    # 保证长度对齐：用 [400000 : 400000 + N_windows]
    label_data = label[400000:400000 + N_windows]

    # 转成 I/Q 比特标签
    label_data_IQ = np.zeros((len(label_data), 2), dtype=int)
    for i in range(len(label_data)):
        if label_data[i] == 0:
            label_data_IQ[i] = [0, 0]
        elif label_data[i] == 1:
            label_data_IQ[i] = [0, 1]
        elif label_data[i] == 2:
            label_data_IQ[i] = [1, 1]
        elif label_data[i] == 3:
            label_data_IQ[i] = [1, 0]

    # ====== 4. SNR 扫描 ======
    model_bers = []

    # 对应 SNR = -5, -4, ..., 11 dB，共 17 个点
    snr_range = np.arange(0, 12, 1)

    for snr_db in snr_range:
        print(f"\n========== 当前 SNR (per symbol): {snr_db} dB ==========")
        # 采样域 SNR，要减去 10log10(16)
        snr_db_sample = snr_db - 10 * math.log10(SAMPLES_PER_SYMBOL)

        model_ber = matched_filter_decision(
            labels=label_data_IQ,
            snr_db_sample=snr_db_sample,
            n_steps=n_steps,
            ddrm=ddrm,
            rx_clean=rx_clean,
            h_np=h_np,
            SAMPLES_PER_SYMBOL=SAMPLES_PER_SYMBOL,
            batch_size=batch_size,
            symbol_span=symbol_span
        )

        model_bers.append(model_ber)

    # ====== 5. 参考曲线（长度 17，对应 snr_range 0 ~ 11 dB） ======
    ref_bers = [
        0.349084, 0.337354, 0.325297, 0.312177, 0.300510,
        0.290758, 0.281636, 0.270774, 0.262986, 0.256229,
        0.249686, 0.244446
    ]

    # ====== 6. 绘制并保存 BER 曲线 ======
    plot_ber_curve(
        model_bers, ref_bers, snr_range,
        save_path=f'ddrm/ddrm_nak/ber_result/ber_curve_nsteps{n_steps}.png'
    )
