import os
import math
import csv
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# 项目模块
from dataset.dataset import QPSKDataset
from model.resnet import RESNET_model

# ==========================================
# 1. 噪声 & 标签映射等工具函数
# ==========================================
def add_awgn_noise_np(clean_data, EbN0_db):
    """
    给信号添加 AWGN 噪声（numpy 版）
    clean_data: [N, 2, L]，I/Q 两路
    EbN0_db: 每比特 SNR (dB)（这里已经减去了 sps 的那一步）
    """
    noisy_data = np.zeros(clean_data.shape, dtype=np.float32)

    # 信号功率：两路 I/Q 叠加后求平均
    signal_power = np.mean(
        (np.abs(clean_data[:, 0, :]) ** 2) +
        (np.abs(clean_data[:, 1, :]) ** 2)
    )

    EbN0_linear = 10 ** (EbN0_db / 10.0)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0 / 2.0)

    noise_I = noise_std * np.random.randn(*clean_data[:, 0, :].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:, 1, :].shape)

    noisy_data[:, 0, :] = clean_data[:, 0, :] + noise_I
    noisy_data[:, 1, :] = clean_data[:, 1, :] + noise_Q

    return noisy_data


def create_targets(y, device):
    """
    y: [B]，符号标签 0,1,2,3

    映射规则（与你训练时保持一致）：
      0 -> (I=-1, Q=-1)
      1 -> (I=-1, Q=+1)
      2 -> (I=+1, Q=+1)
      3 -> (I=+1, Q=-1)

    使用 0/1 表示 I/Q 两路 bit：
      I 分支 (y0):
        符号 2,3 为 +1 -> 0
        符号 0,1 为 -1 -> 1

      Q 分支 (y1):
        符号 1,2 为 +1 -> 0
        符号 0,3 为 -1 -> 1
    """
    y = y.to(device)
    y0 = torch.zeros_like(y, dtype=torch.float32, device=device)
    y1 = torch.zeros_like(y, dtype=torch.float32, device=device)

    mask_0 = (y == 0)
    mask_1 = (y == 1)
    mask_2 = (y == 2)
    mask_3 = (y == 3)

    # I 分支
    y0[mask_1 | mask_2] = 0
    y0[mask_0 | mask_3] = 1

    # Q 分支
    y1[mask_2 | mask_3] = 0
    y1[mask_0 | mask_1] = 1

    return y0, y1


# ==========================================
# 2. 单个 SNR 下评估 BER
# ==========================================
def eval_resnet_ber_for_loader(model, test_loader, device):
    """
    给一个 test_loader，跑完返回：
      - ber: bit error rate
      - avg_loss: 平均 loss（两支 BCE 之和）
    """
    criterion_i = nn.BCEWithLogitsLoss()
    criterion_q = nn.BCEWithLogitsLoss()

    model.eval()
    total_loss = 0.0
    correct0, correct1, total_bits = 0, 0, 0

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", ncols=110, leave=False):
            # Dataset __getitem__ 返回：x_clean, y_impaired, z_h, labels
            x_clean, y_impaired, z_h, labels = batch

            x = y_impaired.float().to(device)  # 受损信号作为模型输入
            z = z_h.float().to(device)         # 信道
            y = labels.long().to(device)       # 0/1/2/3 标签

            outputs = model(x, z)              # [B, 2]
            outputs_i = outputs[:, 0]
            outputs_q = outputs[:, 1]

            y0, y1 = create_targets(y, device)

            loss0 = criterion_i(outputs_i, y0)
            loss1 = criterion_q(outputs_q, y1)
            loss = loss0 + loss1
            total_loss += loss.item() * y.size(0)

            predicted0 = (outputs_i > 0).float()
            predicted1 = (outputs_q > 0).float()

            correct0 += (predicted0 == y0).sum().item()
            correct1 += (predicted1 == y1).sum().item()
            total_bits += y.size(0) * 2  # 每个符号 2 bit

    ber = (total_bits - (correct0 + correct1)) / total_bits
    avg_loss = total_loss / (total_bits / 2)  # 除以符号数
    return ber, avg_loss


# ==========================================
# 3. 画 BER 曲线（模型 vs baseline）
# ==========================================
def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(10, 7))

    # 模型 BER
    plt.semilogy(snr_range, model_bers, 'o-', label='ResNet Model')

    # baseline（从 csv 读出来）
    if len(ref_bers) == len(snr_range) and any(ref_bers):
        plt.semilogy(snr_range, ref_bers, 's--', label='Baseline')

    # 只显示偶数刻度
    min_snr = np.min(snr_range)
    max_snr = np.max(snr_range)
    xticks = np.arange(np.ceil(min_snr / 2) * 2,
                       np.floor(max_snr / 2) * 2 + 2, 2)
    plt.xticks(xticks)

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('Bit Error Rate vs. SNR (QPSK, ResNet Equalizer)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.show()


# ==========================================
# 4. 主程序：扫 SNR，读 baseline CSV，画曲线
# ==========================================
if __name__ == "__main__":
    # ----- 配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    sps = 16

    ckpt_path = r"F:\LJN\bishe\bishe\CNN\P2P_nakagmi\results\best_resnet_model.pth"
    baseline_csv_path = r"F:\LJN\bishe\bishe\CNN\P2P_nakagmi\ber_result\baseline_ber.csv"
    result_save_path = r"F:\LJN\bishe\bishe\CNN\P2P_nakagmi\ber_result\ber_curve_resnet.png"

    print(f"使用设备: {device}")

    # ----- 1. 加载模型 -----
    model = RESNET_model(num_classes=2, dropout=0.2, use_se=True).to(device)
    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}")

    # ----- 2. SNR 扫描 -----
    snr_range = np.arange(0, 19, 1) 
    model_bers = []

    print("开始测试 ResNet BER 曲线...")
    for snr_db in snr_range:
        print(f"\n=== SNR (Symbol): {snr_db} dB ===")

        # Dataset 区间 [400000, 500000)
        test_data = QPSKDataset(400000, 500000)

        # 符号级 SNR -> 每样本 / 每比特 SNR（考虑 sps=16）
        ebn0_db = snr_db - 10 * math.log(16, 10) + 10 * math.log10(2)

        # 对 y_impaired 加 AWGN 噪声
        test_data.y = add_awgn_noise_np(test_data.y, ebn0_db)

        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

        ber, avg_loss = eval_resnet_ber_for_loader(model, test_loader, device=device)
        model_bers.append(ber)

        print(f"SNR = {snr_db:.1f} dB, BER = {ber:.6e}, Avg Loss = {avg_loss:.6f}")

    # ----- 3. 读取 baseline CSV -----
    ref_bers = []
    if os.path.exists(baseline_csv_path):
        try:
            with open(baseline_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                # 假设 csv 里有列名 'snr_db' 和 'baseline_ber'
                baseline_data = {float(row['snr_db']): float(row['baseline_ber']) for row in reader}
                for snr in snr_range:
                    ref_bers.append(baseline_data.get(float(snr), 0.0))
            print(f"已从 {baseline_csv_path} 读取 baseline BER。")
        except Exception as e:
            print(f"读取 baseline CSV 时出错: {e}")
    else:
        print(f"⚠️ 未找到 baseline CSV: {baseline_csv_path}")
        ref_bers = [0.0 for _ in snr_range]

    # 也顺手把本次结果存一个 csv
    results_df = pd.DataFrame({
        'snr_db': snr_range,
        'resnet_ber': model_bers,
        'baseline_ber': ref_bers
    })
    save_csv_path = os.path.splitext(result_save_path)[0] + "_values.csv"
    os.makedirs(os.path.dirname(save_csv_path), exist_ok=True)
    results_df.to_csv(save_csv_path, index=False)
    print(f"数值结果已保存到: {save_csv_path}")

    # ----- 4. 绘图 -----
    plot_ber(model_bers, ref_bers, snr_range, result_save_path)
    print(f"BER 曲线已保存到: {result_save_path}")
