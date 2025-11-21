import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams

from model.unet import build_network
from ddrm_core import DDRM
from dataset.dataset import QPSKDataset
from test_fig import add_awgn_noise_np

# 中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


# ===== 一些基础工具 =====
def decision_making(symbols, threshold=0.0):
    """
    QPSK 硬判决：复数符号 -> 2bit (I,Q)
    symbols: [N,] complex
    return: [N,2] 0/1 比特
    """
    real_part = np.real(symbols)
    imag_part = np.imag(symbols)

    decisions = np.zeros((len(symbols), 2), dtype=int)

    m00 = (real_part > threshold) & (imag_part > threshold)
    m01 = (real_part < threshold) & (imag_part > threshold)
    m11 = (real_part < threshold) & (imag_part < threshold)
    m10 = (real_part > threshold) & (imag_part < threshold)

    decisions[m00] = (0, 0)
    decisions[m01] = (0, 1)
    decisions[m11] = (1, 1)
    decisions[m10] = (1, 0)

    return decisions


def calculate_ber(labels_true, labels_pred):
    """
    逐 bit 计算 BER
    labels_*: [N,2] 0/1
    """
    labels_pred = labels_pred.astype(int)
    err_i = np.sum(labels_true[:, 0] != labels_pred[:, 0])
    err_q = np.sum(labels_true[:, 1] != labels_pred[:, 1])
    ber = (err_i + err_q) / (len(labels_true) * 2)
    print(f"BER: {ber:.6e}")
    return ber


# ===== DDRM 恢复 + 抽样 =====
def ddrm_restore_symbol(snr_db_sample, ddrm, rx_clean, h_np, batch_size=256):
    """
    在给定采样域 SNR 下，用 DDRM 对 Rayleigh+AWGN 波形做恢复，
    再在时间中点抽一个采样作为“符号”。
    """
    device = ddrm.device
    n, c, L = rx_clean.shape
    assert c == 2

    # Rayleigh-only 上叠加 AWGN
    rx_noisy = add_awgn_noise_np(rx_clean, snr_db_sample)

    # 用与 add_awgn_noise_np 一致的方式估计噪声方差
    signal_power = np.mean(rx_clean[:, 0, :] ** 2 + rx_clean[:, 1, :] ** 2)
    ebn0_linear = 10 ** (snr_db_sample / 10.0)
    n0 = signal_power / ebn0_linear
    sigma_y = math.sqrt(n0 / 2.0)

    # 转成 tensor 后分 batch 调用 ddrm.restore
    y_all = torch.from_numpy(rx_noisy).float().to(device)
    h_all = torch.from_numpy(h_np).float().to(device)

    recovered = []

    for start in tqdm(range(0, n, batch_size),
                      desc=f"SNR(sample)={snr_db_sample:.2f} dB"):
        end = min(start + batch_size, n)
        x_rec = ddrm.restore(
            y=y_all[start:end],
            h=h_all[start:end],
            sigma_y=sigma_y,
            sigma_p=1.0,
            eta=0.0,
            init_from='y',  # 从观测 y 初始化逆扩散
        )
        recovered.append(x_rec.cpu().numpy())

    recovered = np.concatenate(recovered, axis=0)  # [N,2,L]
    _, _, wave_len = recovered.shape

    # 很粗暴：直接取时间长度的中点作为符号采样点
    idx = wave_len // 2
    sampled_iq = recovered[:, :, idx]              # [N,2]
    sampled_complex = sampled_iq[:, 0] + 1j * sampled_iq[:, 1]
    return sampled_complex


def run_ddrm_chain(labels_iq, snr_db_sample, ddrm, rx_clean, h_np,
                   batch_size=256):
    """
    完整一条 DDRM 链路：恢复 -> 抽样 -> 判决 -> BER
    """
    symbols = ddrm_restore_symbol(
        snr_db_sample=snr_db_sample,
        ddrm=ddrm,
        rx_clean=rx_clean,
        h_np=h_np,
        batch_size=batch_size,
    )
    labels_pred = decision_making(symbols)
    return calculate_ber(labels_iq, labels_pred)


# ===== 画 BER 曲线 =====
def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(12, 8))

    snr_array = np.array(snr_range)
    plt.semilogy(snr_array, model_bers, 'o-', label='DDRM 恢复')
    plt.semilogy(snr_array, ref_bers, 's-', label='参考曲线')

    xticks = np.arange(
        np.ceil(min(snr_array) / 2) * 2,
        np.floor(max(snr_array) / 2) * 2 + 2,
        2,
    )
    plt.xticks(xticks)

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('QPSK 在 nakagmi+AWGN 下：DDRM 与参考曲线对比')
    plt.legend()

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # ----- 全局配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 80          # 扩散步数
    batch_size = 4096     # 测试 batch 大小
    sps = 16              # 每符号采样点数

    # ----- 1. 加载训练好的 DDRM 模型 -----
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)

    ckpt_path = fr'ddrm/ddrm_nakagmi/results/best_model_epoch_with_n_steps{n_steps}.pth'
    print(f"加载模型权重: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    ddrm = DDRM(model, n_steps=n_steps,
                min_beta=1e-4, max_beta=0.02, device=device)

    # ----- 2. 加载 Rayleigh 数据窗口 -----
    test_data = QPSKDataset(400000, 500000)
    rx_clean = test_data.y   # [N,2,L] 仅含瑞利
    h_np = test_data.z       # [N,2]    信道系数
    n_win = rx_clean.shape[0]

    # ----- 3. 加载符号标签并转为 IQ-bit -----
    label = np.load(r'F:\LJN\bishe\bishe\data\nakagmi_data\labels.npy')
    label_seg = label[400000:400000 + n_win]

    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    labels_iq = np.array([map_label[int(v)] for v in label_seg], dtype=int)

    # ----- 4. SNR 扫描 -----
    snr_range = np.arange(2, 18, 1)  # per-symbol SNR(dB)
    model_bers = []

    for snr_db in snr_range:
        print(f"\n=== SNR (per symbol): {snr_db} dB ===")
        # 换算到“采样域 SNR”：Es,Ts -> Es/样本
        snr_db_sample = snr_db - 10 * math.log10(sps)

        ber = run_ddrm_chain(
            labels_iq=labels_iq,
            snr_db_sample=snr_db_sample,
            ddrm=ddrm,
            rx_clean=rx_clean,
            h_np=h_np,
            batch_size=batch_size,
        )
        model_bers.append(ber)

    # ----- 5. 参考 BER 序列（预先算好的值） -----
    ref_bers = [
        0.325059, 0.313570, 0.301576, 0.290163,
        0.281074, 0.271785, 0.262691, 0.255602,
        0.249684, 0.244232, 0.239477, 0.235854,
        0.232577, 0.229836, 0.227602, 0.226255,
    ]

    # ----- 6. 画 BER 曲线 -----
    plot_ber(
        model_bers,
        ref_bers,
        snr_range,
        save_path=f'ddrm/ddrm_nakagmi/ber_result/ber_curve_nsteps{n_steps}.png'
    )
