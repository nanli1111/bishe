import os
import math
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib import rcParams
from torch.utils.data import DataLoader, Subset

from model.resnet_se import SETimeResNet1D        # 你的双head模型（forward 返回 x0_hat, h_hat）
from dataset.dataset import QPSKDataset           # 你的 dataset
from noise_utils import add_awgn_noise_torch      # 你的加噪


# 中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# -------------------------
# 工具：est_h (B,10,S) -> (B,10,L)
# -------------------------
def expand_h_to_wave(est_h, sps: int):
    return torch.repeat_interleave(est_h, repeats=sps, dim=-1)


# -------------------------
# 简洁版 Joint I2SB 采样器（不喂 y，只喂 x_t + est_h）
# -------------------------
class I2SBJointSampler(torch.nn.Module):
    def __init__(self, model, n_steps=20, sps=16, device="cuda"):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.sps = sps
        self.device = device

    @torch.no_grad()
    def sample(self, y_noisy, est_h_sym, use_heun=True):
        """
        y_noisy:   [B,2,L]
        est_h_sym: [B,10,S]
        return:
          x0_hat:  [B,2,L]
          h_hat:   [B,10,S]  (最后一次网络输出)
        """
        y_noisy = y_noisy.to(self.device).float()
        est_h_sym = est_h_sym.to(self.device).float()

        B, _, L = y_noisy.shape
        est_h_exp = expand_h_to_wave(est_h_sym, self.sps)  # [B,10,L]

        x = y_noisy.clone()
        ts = np.linspace(1.0, 0.0, self.n_steps + 1)

        last_h = None

        def pred(x_curr, t_scalar):
            t_idx = int(round(t_scalar * (self.n_steps - 1)))
            t_idx = max(0, min(self.n_steps - 1, t_idx))
            t_idx = torch.full((B,), t_idx, device=self.device, dtype=torch.long)

            net_in = torch.cat([x_curr, est_h_exp], dim=1)   # [B,12,L]
            x0_hat, h_hat = self.model(net_in, t_idx)
            return x0_hat, h_hat

        for i in range(self.n_steps):
            t_curr = float(ts[i])
            t_next = float(ts[i + 1])
            dt = t_curr - t_next

            x0_hat, h_hat = pred(x, t_curr)
            last_h = h_hat

            if use_heun:
                # v = (x - x0)/t
                v1 = (x - x0_hat) / max(t_curr, 1e-6)
                x_probe = x - dt * v1

                x0_hat2, _ = pred(x_probe, t_next)
                v2 = (x_probe - x0_hat2) / max(t_next, 1e-6)

                v = 0.5 * (v1 + v2)
                x = x - dt * v
            else:
                ratio = t_next / max(t_curr, 1e-6)
                x = ratio * x + (1.0 - ratio) * x0_hat

        # 最后一帧再算一次更稳（可省略）
        x0_hat, h_hat = pred(x, 0.0)
        last_h = h_hat
        return x0_hat, last_h


# -------------------------
# 1) 波形叠加可视化
# -------------------------
@torch.no_grad()
def visualize_waveform_overlay(model, sampler, dataset, snr_list, sps=16, device="cuda", save_dir="vis_wave"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    idx = np.random.randint(0, len(dataset))
    clean_x, impaired_y, est_h, true_h, _ = dataset[idx]

    clean_x = clean_x.unsqueeze(0).to(device).float()      # [1,2,L]
    impaired_y = impaired_y.unsqueeze(0).to(device).float()
    est_h = est_h.unsqueeze(0).to(device).float()          # [1,10,S]

    B, _, L = clean_x.shape

    for snr_db in snr_list:
        # 你训练里用的：符号SNR -> 采样SNR
        snr_db_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)
        y_noisy = add_awgn_noise_torch(impaired_y, snr_db_sample)

        # One-Step: t=1 => x_t = y_noisy
        t_max = torch.full((B,), sampler.n_steps - 1, device=device, dtype=torch.long)
        est_h_exp = expand_h_to_wave(est_h, sps)
        x_onestep, h_onestep = model(torch.cat([y_noisy, est_h_exp], dim=1), t_max)

        # I2SB 多步
        x_rf, h_rf = sampler.sample(y_noisy, est_h, use_heun=True)

        clean_np = clean_x.cpu().numpy()
        y_np = y_noisy.cpu().numpy()
        os_np = x_onestep.cpu().numpy()
        rf_np = x_rf.cpu().numpy()

        fig, axs = plt.subplots(2, 3, figsize=(18, 10))

        axs[0, 0].plot(clean_np[0, 0, :], label="Clean", alpha=0.7, linewidth=2)
        axs[0, 0].plot(y_np[0, 0, :], label="Noisy y", linestyle="--", alpha=0.8)
        axs[0, 0].set_title("I：Clean vs Noisy")
        axs[0, 0].legend()
        axs[0, 0].grid(True, linestyle="--", alpha=0.4)

        axs[0, 1].plot(os_np[0, 0, :], label="One-Step", alpha=0.9)
        axs[0, 1].set_title("I：One-Step")
        axs[0, 1].grid(True, linestyle="--", alpha=0.4)

        axs[0, 2].plot(clean_np[0, 0, :], label="Clean", alpha=0.35, linewidth=3)
        axs[0, 2].plot(os_np[0, 0, :], label="One-Step", linestyle="--", alpha=0.85)
        axs[0, 2].plot(rf_np[0, 0, :], label="I2SB", alpha=0.95)
        axs[0, 2].set_title("I：结果对比")
        axs[0, 2].legend()
        axs[0, 2].grid(True, linestyle="--", alpha=0.4)

        axs[1, 0].plot(clean_np[0, 1, :], label="Clean", alpha=0.7, linewidth=2)
        axs[1, 0].plot(y_np[0, 1, :], label="Noisy y", linestyle="--", alpha=0.8)
        axs[1, 0].set_title("Q：Clean vs Noisy")
        axs[1, 0].grid(True, linestyle="--", alpha=0.4)

        axs[1, 1].plot(os_np[0, 1, :], label="One-Step", alpha=0.9)
        axs[1, 1].set_title("Q：One-Step")
        axs[1, 1].grid(True, linestyle="--", alpha=0.4)

        axs[1, 2].plot(clean_np[0, 1, :], label="Clean", alpha=0.35, linewidth=3)
        axs[1, 2].plot(os_np[0, 1, :], label="One-Step", linestyle="--", alpha=0.85)
        axs[1, 2].plot(rf_np[0, 1, :], label="I2SB", alpha=0.95)
        axs[1, 2].set_title("Q：结果对比")
        axs[1, 2].legend()
        axs[1, 2].grid(True, linestyle="--", alpha=0.4)

        plt.suptitle(f"波形对比（输入=x_t+est_h）| SNR={snr_db} dB | idx={idx}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"wave_snr{snr_db}_idx{idx}.png"), dpi=220)
        plt.close()

    print(f"✅ 波形图已保存到: {save_dir}")


# -------------------------
# 2) 星座图可视化（取中点采样）
# -------------------------
@torch.no_grad()
def visualize_constellation(model, sampler, dataset, snr_list, sps=16, device="cuda",
                            save_dir="vis_constellation", num_points=2048, batch_size=256):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    n_use = min(num_points, len(dataset))
    loader = DataLoader(Subset(dataset, list(range(n_use))), batch_size=batch_size, shuffle=False)

    # Clean reference
    gt_I, gt_Q = [], []
    for clean_x, _, _, _, _ in loader:
        mid = clean_x.shape[-1] // 2
        gt_I.append(clean_x[:, 0, mid].numpy())
        gt_Q.append(clean_x[:, 1, mid].numpy())
    gt_I = np.concatenate(gt_I, axis=0)
    gt_Q = np.concatenate(gt_Q, axis=0)

    for snr_db in snr_list:
        rec_I, rec_Q, os_I, os_Q, rf_I, rf_Q = [], [], [], [], [], []

        for clean_x, impaired_y, est_h, true_h, _ in loader:
            impaired_y = impaired_y.to(device).float()
            est_h = est_h.to(device).float()

            B, _, L = impaired_y.shape
            mid = L // 2

            snr_db_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)
            y_noisy = add_awgn_noise_torch(impaired_y, snr_db_sample)

            # One-Step
            t_max = torch.full((B,), sampler.n_steps - 1, device=device, dtype=torch.long)
            est_h_exp = expand_h_to_wave(est_h, sps)
            x_onestep, _ = model(torch.cat([y_noisy, est_h_exp], dim=1), t_max)

            # I2SB
            x_rf, _ = sampler.sample(y_noisy, est_h, use_heun=True)

            y_np = y_noisy.detach().cpu().numpy()
            os_np = x_onestep.detach().cpu().numpy()
            rf_np = x_rf.detach().cpu().numpy()

            rec_I.append(y_np[:, 0, mid]);  rec_Q.append(y_np[:, 1, mid])
            os_I.append(os_np[:, 0, mid]);  os_Q.append(os_np[:, 1, mid])
            rf_I.append(rf_np[:, 0, mid]);  rf_Q.append(rf_np[:, 1, mid])

        rec_I = np.concatenate(rec_I); rec_Q = np.concatenate(rec_Q)
        os_I  = np.concatenate(os_I);  os_Q  = np.concatenate(os_Q)
        rf_I  = np.concatenate(rf_I);  rf_Q  = np.concatenate(rf_Q)

        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        alpha_val, s_val, lim = 0.35, 5, 2.5

        axs[0].scatter(rec_I, rec_Q, s=s_val, alpha=alpha_val)
        axs[0].set_title(f"Received (Noisy y)\nSNR={snr_db}dB")
        axs[0].set_xlim(-lim, lim); axs[0].set_ylim(-lim, lim); axs[0].grid(True, linestyle="--", alpha=0.5)

        axs[1].scatter(os_I, os_Q, s=s_val, alpha=alpha_val)
        axs[1].set_title("One-Step 输出")
        axs[1].set_xlim(-lim, lim); axs[1].set_ylim(-lim, lim); axs[1].grid(True, linestyle="--", alpha=0.5)

        axs[2].scatter(rf_I, rf_Q, s=s_val, alpha=alpha_val)
        axs[2].set_title("I2SB 多步输出")
        axs[2].set_xlim(-lim, lim); axs[2].set_ylim(-lim, lim); axs[2].grid(True, linestyle="--", alpha=0.5)

        axs[3].scatter(gt_I, gt_Q, s=s_val, alpha=alpha_val, c="black")
        axs[3].set_title("Clean Reference (GT)")
        axs[3].set_xlim(-lim, lim); axs[3].set_ylim(-lim, lim); axs[3].grid(True, linestyle="--", alpha=0.5)

        plt.suptitle(f"星座图对比（输入=x_t+est_h）| SNR={snr_db} dB", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"constellation_snr{snr_db}.png"), dpi=260)
        plt.close()

    print(f"✅ 星座图已保存到: {save_dir}")


# -------------------------
# 3) 信道补全可视化（5条路径幅度/相位）
# -------------------------
@torch.no_grad()
def visualize_channel(model, sampler, dataset, snr_list, sps=16, device="cuda", save_dir="vis_channel"):
    os.makedirs(save_dir, exist_ok=True)
    model.eval()

    idx = np.random.randint(0, len(dataset))
    clean_x, impaired_y, est_h, true_h, _ = dataset[idx]

    impaired_y = impaired_y.unsqueeze(0).to(device).float()  # [1,2,L]
    est_h = est_h.unsqueeze(0).to(device).float()            # [1,10,S]
    true_h = true_h.unsqueeze(0).to(device).float()          # [1,10,S]

    B, _, L = impaired_y.shape
    S = true_h.shape[-1]
    n_paths = 5
    sym = np.arange(S)

    def split_mag_phase(h10S):
        re = h10S[0::2, :]  # [5,S]
        im = h10S[1::2, :]  # [5,S]
        mag = torch.sqrt(re**2 + im**2 + 1e-12)
        ang = torch.atan2(im, re)
        return mag, ang

    # 导频位置（est_h 非零）
    pilot_mask = (torch.sum(torch.abs(est_h[0]), dim=0) > 1e-9).detach().cpu().numpy()
    pilot_pos = np.where(pilot_mask)[0]

    for snr_db in snr_list:
        snr_db_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)
        y_noisy = add_awgn_noise_torch(impaired_y, snr_db_sample)

        # One-Step
        t_max = torch.full((B,), sampler.n_steps - 1, device=device, dtype=torch.long)
        est_h_exp = expand_h_to_wave(est_h, sps)
        _, h_onestep = model(torch.cat([y_noisy, est_h_exp], dim=1), t_max)

        # I2SB
        _, h_rf = sampler.sample(y_noisy, est_h, use_heun=True)

        true_mag, true_ang = split_mag_phase(true_h[0])
        rf_mag, rf_ang     = split_mag_phase(h_rf[0])
        os_mag, os_ang     = split_mag_phase(h_onestep[0])
        est_mag, _         = split_mag_phase(est_h[0])

        fig, axs = plt.subplots(n_paths, 2, figsize=(16, 3.2 * n_paths))

        for p in range(n_paths):
            axs[p, 0].plot(sym, true_mag[p].cpu().numpy(), label="True |h|", linewidth=2)
            axs[p, 0].plot(sym, os_mag[p].cpu().numpy(), label="One-Step |h|", linestyle="--", alpha=0.8)
            axs[p, 0].plot(sym, rf_mag[p].cpu().numpy(), label="I2SB |h|", linestyle="-.", alpha=0.9)
            if len(pilot_pos) > 0:
                axs[p, 0].scatter(pilot_pos, est_mag[p].cpu().numpy()[pilot_pos], s=90, marker="*", label="est_h(导频)")
            axs[p, 0].set_title(f"路径 {p} 幅度")
            axs[p, 0].grid(True, linestyle="--", alpha=0.4)
            if p == 0: axs[p, 0].legend()

            axs[p, 1].plot(sym, true_ang[p].cpu().numpy(), label="True ∠h", linewidth=2)
            axs[p, 1].plot(sym, os_ang[p].cpu().numpy(), label="One-Step ∠h", linestyle="--", alpha=0.8)
            axs[p, 1].plot(sym, rf_ang[p].cpu().numpy(), label="I2SB ∠h", linestyle="-.", alpha=0.9)
            axs[p, 1].set_title(f"路径 {p} 相位")
            axs[p, 1].grid(True, linestyle="--", alpha=0.4)
            if p == 0: axs[p, 1].legend()

        plt.suptitle(f"信道补全对比（输入=x_t+est_h）| SNR={snr_db} dB | idx={idx}", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"h_snr{snr_db}_idx{idx}.png"), dpi=230)
        plt.close()

    print(f"✅ 信道图已保存到: {save_dir}")


# -------------------------
# main
# -------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = r"F:\LJN\bishe\bishe\Estimate\data\tdl_data_5h"
    ckpt_path = r"F:\LJN\bishe\bishe\Estimate\tdl\results\best_joint_i2sb_steps20.pth"  # 你按实际改
    save_dir = r"F:\LJN\bishe\bishe\Estimate\tdl\results"
    
    n_steps = 20
    sps = 16
    h_symbols = 11
    snr_list = [0, 5, 10, 15, 20]

    # ✅ 输入通道 = x_t(2) + est_h_expanded(10) = 12
    model = SETimeResNet1D(
        in_channels=12,
        out_wave_channels=2,
        out_h_channels=10,
        hidden_dim=128,
        num_blocks=12,
        time_emb_dim=128,
        sps=sps,
        h_symbols=h_symbols,
    ).to(device)

    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"❌ 找不到权重文件: {ckpt_path}")
        raise FileNotFoundError(ckpt_path)

    sampler = I2SBJointSampler(model, n_steps=n_steps, sps=sps, device=device)

    test_dataset = QPSKDataset(
        start_samples=400000,
        end_samples=402048,
        data_dir=data_dir,
        expand_h_to_wave=False
    )

    visualize_waveform_overlay(
        model, sampler, test_dataset, snr_list, sps=sps, device=device,
        save_dir=os.path.join(save_dir, "vis_wave_joint_new")
    )

    visualize_constellation(
        model, sampler, test_dataset, snr_list, sps=sps, device=device,
        save_dir=os.path.join(save_dir, "vis_constellation_joint_new"),
        num_points=2048, batch_size=256
    )

    visualize_channel(
        model, sampler, test_dataset, snr_list, sps=sps, device=device,
        save_dir=os.path.join(save_dir, "vis_channel_joint_new")
    )

    print("✅ 所有可视化完成。")
