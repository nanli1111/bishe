import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv  # 新增：引入 csv 模块

from model.unet import UNet
from conditionDM.condition_rayleigh.ddpm import CDDM
from dataset.dataset import QPSKDataset
from test_fig import add_awgn_noise_np

# ===== 画图中文支持 =====
rcParams['font.sans-serif'] = ['Microsoft YaHei']
rcParams['axes.unicode_minus'] = False


# ===== 判决 & BER =====
def decision_making(symbols):
    real_part = np.real(symbols)
    imag_part = np.imag(symbols)
    bits = np.zeros((len(symbols), 2), dtype=int)

    bits[(real_part > 0) & (imag_part > 0)] = (0, 0)
    bits[(real_part < 0) & (imag_part > 0)] = (0, 1)
    bits[(real_part < 0) & (imag_part < 0)] = (1, 1)
    bits[(real_part > 0) & (imag_part < 0)] = (1, 0)
    return bits


def calculate_ber(true_bits, pred_bits):
    err = np.sum(true_bits != pred_bits)
    ber = err / true_bits.size
    print(f"BER: {ber:.6e}")
    return ber


# ============ MMSE 均衡器 ============
def mmse_equalization(received_signal, channel_estimates, snr_db):
    """
    MMSE 均衡
    received_signal:   [N,2,L] 实数 I/Q
    channel_estimates: [N,2]   实部/虚部
    snr_db:            采样域 SNR(dB)
    """
    # I/Q -> 复数
    y_complex = received_signal[:, 0, :] + 1j * received_signal[:, 1, :]  # [N,L]

    h_complex = channel_estimates[:, 0, :] + 1j * channel_estimates[:, 1, :]    # [N]

    snr_linear = 10 ** (snr_db / 10.0)
    w = np.conj(h_complex) / (np.abs(h_complex) ** 2 + 1.0 / snr_linear)  # [N]

    # 每个样本整段平衰落：一条波形上用同一个 w
    r_complex = y_complex * w  # [N,L]

    r_i = np.real(r_complex)
    r_q = np.imag(r_complex)
    r = np.stack([r_i, r_q], axis=1)    # [N,2,L]
    return r


# ===== 整条链路：加噪 -> MMSE 均衡 -> CDDM 逆扩散 -> 抽样判决 -> BER =====
def cddm_run_chain(snr_db_sample, cddm, model, rx_faded, h_np, true_bits, batch_size=256):
    """
    snr_db_sample：采样域 SNR(dB)，和 add_awgn_noise_np 一致
    rx_faded：     [N,2,L]，已经包含瑞利衰落的基带信号
    h_np：         [N,2]，真实信道系数（实部、虚部）
    """
    device = cddm.device
    n, _, L = rx_faded.shape

    seq_len = rx_faded.shape[2]
    h_np = h_np[:, :, np.newaxis]         # (N,2,1)
    # 将最后一维从 1 repeat 到 48
    h_np = np.repeat(h_np, seq_len, axis=-1)  # (N, 2, 48）
    # 在“已有衰落”的波形上叠加 AWGN
    rx_noisy = add_awgn_noise_np(rx_faded, snr_db_sample)

    # 噪声方差估算（和 DDRM 代码保持一致）
    sig_power = np.mean(rx_faded[:, 0, :] ** 2 + rx_faded[:, 1, :] ** 2)
    snr_linear = 10 ** (snr_db_sample / 10.0)
    n0 = sig_power / snr_linear
    sigma_y = math.sqrt(n0 / 2.0)

    # MMSE 均衡：得到等效接收 r，作为 y_r
    rx_eq = mmse_equalization(rx_noisy, h_np, snr_db_sample)  # [N,2,L]

    y_all = torch.from_numpy(rx_eq).float().to(device)   # [N,2,L] 均衡后信号
    h_all = torch.from_numpy(h_np).float().to(device)    # [N,2]

    recovered = []
    model.eval()
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size),
                          desc=f"SNR(sample)={snr_db_sample:.2f} dB"):
            end = min(start + batch_size, n)
            y = y_all[start:end]  # [B,2,L]  已均衡
            h = h_all[start:end]  # [B,2]
            B = y.size(0)

            # 构造复信道向量 [B,k]，k=L，平衰落：整段用同一个 h
            h_real = h[:, 0, :]      # [B,L]
            h_imag = h[:, 1, :]      # [B,L]
            h_c = torch.complex(h_real, h_imag)   # [B,L]

            # CDDM 逆向扩散：
            # net 接口是 net(x_t, t, h_r)，用 lambda 把 UNet 包一下
            x_rec = cddm.sample_backward(
                y_r=y,
                net=lambda x_t, t_cond, h_r: model(
                    torch.cat([x_t, h_r.view(B, 2, L)], dim=1),  # [B,4,L]
                    t_cond
                ).view(B, -1),   # 展平到 [B,2L]
                h_c=h_c,
                sigma=sigma_y,
                t_max=None,
                clip_x0=True,
            )  # [B,2,L]

            recovered.append(x_rec.cpu().numpy())

    recovered = np.concatenate(recovered, axis=0)  # [N,2,L]

    # 和 DDRM 一样：取中间一个采样点作为符号
    mid = L // 2
    sym_i = recovered[:, 0, mid]
    sym_q = recovered[:, 1, mid]
    symbols = sym_i + 1j * sym_q

    pred_bits = decision_making(symbols)
    return calculate_ber(true_bits, pred_bits)


# ===== 画 BER 曲线 =====
def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, model_bers, 'o-', label='CDDM 恢复')
    plt.semilogy(snr_range, ref_bers, 's--', label='参考曲线')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('QPSK 在 Rayleigh+AWGN 下：CDDM 性能')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 150
    batch_size = 4096
    sps = 16  # 每符号采样点数

    # ===== 1. 加载训练好的 CDDM UNet =====
    model = UNet(
        n_steps=n_steps,
        channels=[10, 20, 40, 80],
        pe_dim=128,
        residual=False,
        in_channels=4,         # 2 路 x_t + 2 路 h_r
    ).to(device)

    ckpt_path = f'cddm/cddm_rayleigh/results/best_model_epoch_with_n_steps{n_steps}.pth'
    print(f"加载模型权重: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    cddm = CDDM(
        device=device,
        n_steps=n_steps,
        channel_type='rayleigh',
        min_beta=1e-4,
        max_beta=0.02
    )

    # ===== 2. 加载数据：这里 y 本身已经包含瑞利衰落 =====
    data = QPSKDataset(400000, 500000)
    rx_faded = data.y   # [N,2,L]：已经过瑞利衰落
    h_np = data.z       # [N,2]：真实信道系数（实部/虚部）
    n_win = rx_faded.shape[0]

    # ===== 3. 加载符号标签并转成比特 =====
    label = np.load(r'F:\LJN\bishe\bishe\data\rayleigh_data\labels.npy')
    label_seg = label[400000:400000 + n_win]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    true_bits = np.array([map_label[int(v)] for v in label_seg], dtype=int)

    # ===== 4. SNR 扫描 =====
    snr_range = np.arange(2, 18, 1)  # per-symbol SNR(dB)
    model_bers = []

    for snr_db in snr_range:
        print(f"\n=== SNR (per symbol): {snr_db} dB ===")
        # 从符号域 SNR 换到采样域 SNR（和 DDRM 代码保持一致）
        snr_db_sample = snr_db - 10 * math.log10(sps)

        ber = cddm_run_chain(
            snr_db_sample=snr_db_sample,
            cddm=cddm,
            model=model,
            rx_faded=rx_faded,
            h_np=h_np,
            true_bits=true_bits,
            batch_size=batch_size,
        )
        model_bers.append(ber)

    # ===== 5. 参考 BER：从 baseline_ber.csv 读取 (修改部分) =====
    # 确保路径正确，之前生成的 CSV 应该在 'CDDM/cddm_rayleigh/ber_result/baseline_ber.csv'
    csv_path = 'CDDM/cddm_rayleigh/ber_result/baseline_ber.csv'
    ref_snrs = []
    ref_bers = []
    
    if os.path.exists(csv_path):
        print(f"Loading Baseline BER from {csv_path}...")
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)   # 要求表头里有 snr_db, baseline_ber
            for row in reader:
                ref_snrs.append(float(row['snr_db']))
                ref_bers.append(float(row['baseline_ber']))
    else:
        print(f"Warning: CSV file {csv_path} not found! Using empty reference.")
        # 如果找不到文件，ref_snrs 和 ref_bers 为空，画图时可能需要处理

    # ===== 6. 画 BER 曲线 =====
    save_path = f'cddm/cddm_rayleigh/ber_result/ber_curve_nsteps{n_steps}.png'
    
    # 注意：这里使用 ref_snrs 作为 x 轴参考，确保你的仿真循环 snr_range 与 CSV 中的一致
    # 如果 CSV 数据与当前扫描范围不一致，画图可能会有视觉上的错位，建议 ref_snrs 和 snr_range 保持一致
    plot_x_axis = ref_snrs if len(ref_snrs) == len(model_bers) else snr_range
    
    plot_ber(
        model_bers,
        ref_bers,
        plot_x_axis,  # 使用 CSV 的 SNR 或者 当前扫描的 SNR
        save_path=save_path
    )
    print(f"✅ BER 曲线已保存到: {save_path}")