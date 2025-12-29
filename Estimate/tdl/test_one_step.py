import os
import math
import csv
import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import DataLoader

# ====== 项目模块 ======
from model.resnet_se import SETimeResNet1D
from dataset.dataset import QPSKDataset
from noise_utils import add_awgn_noise_torch   # ✅ 直接用你给的版本


# 中文字体（Windows 一般 OK；如果你没装这些字体就改成你有的）
rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
rcParams["axes.unicode_minus"] = False


# ==========================================
# 1) 判决 & BER
# ==========================================
def decision_bits_from_IQ(I, Q):
    """
    按你原来的 decision_making 映射：
      ++ -> (0,0)
      -+ -> (0,1)
      -- -> (1,1)
      +- -> (1,0)
    等价于：
      b0 = (Q < 0)
      b1 = (I < 0)
    I,Q: numpy 1D
    return bits: [N,2]
    """
    b0 = (Q < 0).astype(np.int32)
    b1 = (I < 0).astype(np.int32)
    return np.stack([b0, b1], axis=1)


def labels_to_bits(labels):
    """
    labels: numpy [N]，取值 0..3
    映射同上：
      0->(0,0), 1->(0,1), 2->(1,1), 3->(1,0)
    """
    b0 = np.isin(labels, [2, 3]).astype(np.int32)
    b1 = np.isin(labels, [1, 2]).astype(np.int32)
    return np.stack([b0, b1], axis=1)


def calculate_ber(bits_true, bits_pred):
    err = np.sum(bits_true != bits_pred)
    return err / (bits_true.shape[0] * bits_true.shape[1])


# ==========================================
# 2) NMSE(h)
# ==========================================
def nmse_h(h_hat, h_true, eps=1e-12):
    """
    h_hat/h_true: torch [B,10,S]
    NMSE = ||e||^2 / ||h||^2
    """
    num = torch.sum((h_hat - h_true) ** 2).item()
    den = torch.sum(h_true ** 2).item()
    return num / max(den, eps)


# ==========================================
# 3) 展开 est_h
# ==========================================
def expand_h_to_wave(est_h, sps):
    """
    est_h: [B,10,S] -> [B,10,L], L=S*sps
    """
    return torch.repeat_interleave(est_h, repeats=sps, dim=-1)


# ==========================================
# 4) 保存 & 绘图
# ==========================================
def save_results_to_csv(snr_list, ber_list, nmse_list, save_path):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["snr_db", "ber", "nmse"])
        for s, b, n in zip(snr_list, ber_list, nmse_list):
            w.writerow([float(s), f"{b:.6e}", f"{n:.6e}"])
    print(f"✅ 结果已保存 CSV: {save_path}")


def plot_semilogy(x, y, title, ylabel, save_path, label=None):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.semilogy(x, y, "o-", linewidth=2, label=label)
    plt.grid(True, which="both", linestyle="--", alpha=0.7)
    plt.xlabel("SNR per symbol (dB)")
    plt.ylabel(ylabel)
    plt.title(title)
    if label:
        plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"✅ 曲线已保存: {save_path}")


# ==========================================
# 5) 核心：One-Step 评估（BER + NMSE）
# ==========================================
@torch.no_grad()
def eval_onestep_ber_nmse(
    model,
    loader,
    snr_db_sym,
    n_steps,
    sps,
    device="cuda",
):
    model.eval()

    # 符号SNR -> 采样SNR（保持你训练脚本一致）
    snr_db_sample = snr_db_sym - 10 * math.log10(sps) + 10 * math.log10(2)

    # 统计
    all_bits_true = []
    all_bits_pred = []
    nmse_list = []

    for clean_x, impaired_y, est_h, true_h, label in tqdm(loader, leave=False, desc=f"SNR={snr_db_sym}dB"):
        impaired_y = impaired_y.to(device).float()  # [B,2,L]
        est_h = est_h.to(device).float()            # [B,10,S]
        true_h = true_h.to(device).float()          # [B,10,S]

        B, _, L = impaired_y.shape
        mid = L // 2

        # 1) 加噪（✅ 直接用你给的函数，EbN0_db 传 float）
        y_noisy = add_awgn_noise_torch(impaired_y, float(snr_db_sample))

        # 2) One-step：t_idx 固定最大
        t_idx = torch.full((B,), n_steps - 1, device=device, dtype=torch.long)

        # 3) 输入：y_noisy + est_h_expanded
        est_h_exp = expand_h_to_wave(est_h, sps=sps)      # [B,10,L]
        net_input = torch.cat([y_noisy, est_h_exp], dim=1)  # [B,12,L]

        out = model(net_input, t_idx)

        # 你现在的训练是：x0_hat, h_hat = model(...)
        # 这里就按 tuple 解包
        x0_hat, h_hat = out  # [B,2,L], [B,10,S]

        # 4) BER：取中点符号 I/Q
        x0_np = x0_hat.detach().cpu().numpy()
        I = x0_np[:, 0, mid]
        Q = x0_np[:, 1, mid]
        bits_pred = decision_bits_from_IQ(I, Q)

        labels_np = label.numpy().astype(np.int32)
        bits_true = labels_to_bits(labels_np)

        all_bits_pred.append(bits_pred)
        all_bits_true.append(bits_true)

        # 5) NMSE(h)
        nmse_list.append(nmse_h(h_hat, true_h))

    bits_pred = np.concatenate(all_bits_pred, axis=0)
    bits_true = np.concatenate(all_bits_true, axis=0)
    ber = calculate_ber(bits_true, bits_pred)

    nmse_avg = float(np.mean(nmse_list))  # 每个 batch 一个 NMSE，做均值
    return ber, nmse_avg


# ==========================================
# 6) 主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----- 配置 -----
    data_dir = r"F:\LJN\bishe\bishe\Estimate\data\tdl_data_5h"
    ckpt_path = r"F:\LJN\bishe\bishe\Estimate\tdl\results\best_joint_i2sb_steps20.pth"  # 你按实际改
    save_dir = r"F:\LJN\bishe\bishe\Estimate\tdl\ber_results"

    n_steps = 20
    sps = 16
    h_symbols = 11

    test_start, test_end = 400000, 500000
    batch_size = 4096

    os.makedirs(save_dir, exist_ok=True)

    snr_range = np.arange(0, 19, 1)  # 0~18 dB(symbol)

    # ----- 模型 -----
    model = SETimeResNet1D(
        in_channels=12,        # y_noisy(2) + est_h_expanded(10)
        out_wave_channels=2,
        out_h_channels=10,
        hidden_dim=128,
        num_blocks=12,
        time_emb_dim=128,
        sps=sps,
        h_symbols=h_symbols,
    ).to(device)

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"权重文件未找到: {ckpt_path}")
    print(f"加载模型权重: {ckpt_path}")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # ----- 数据 -----
    test_ds = QPSKDataset(
        start_samples=test_start,
        end_samples=test_end,
        data_dir=data_dir,
        expand_h_to_wave=False
    )
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ----- 跑 SNR sweep -----
    ber_list, nmse_list = [], []
    print("开始 One-Step 评估：BER + 信道 NMSE")

    for snr_db in snr_range:
        ber, nmse = eval_onestep_ber_nmse(
            model=model,
            loader=test_loader,
            snr_db_sym=float(snr_db),
            n_steps=n_steps,
            sps=sps,
            device=device,
        )
        ber_list.append(ber)
        nmse_list.append(nmse)
        print(f"SNR(sym)={snr_db:>2} dB | BER={ber:.6e} | NMSE={nmse:.6e}")

    # ----- 保存 CSV -----
    csv_path = os.path.join(save_dir, "onestep_ber_nmse.csv")
    save_results_to_csv(snr_range, ber_list, nmse_list, csv_path)

    # ----- 画图 -----
    plot_semilogy(
        snr_range, ber_list,
        title="One-Step BER",
        ylabel="BER",
        save_path=os.path.join(save_dir, "ber_curve.png"),
        label="One-Step"
    )
    plot_semilogy(
        snr_range, nmse_list,
        title="One-Step Channel NMSE",
        ylabel="NMSE",
        save_path=os.path.join(save_dir, "nmse_curve.png"),
        label="One-Step"
    )

    print("✅ One-Step BER + NMSE 测试完成。")
