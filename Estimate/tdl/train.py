import os
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import math
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]   # 或者 "SimHei"
plt.rcParams["axes.unicode_minus"] = False

from dataset.dataset import get_train_loader
from noise_utils import add_awgn_noise_torch
from model.resnet_se import SETimeResNet1D


def expand_h_to_wave(est_h, sps):
    # est_h: [B,10,S] -> [B,10,L], L=S*sps
    return torch.repeat_interleave(est_h, repeats=sps, dim=-1)


def weighted_h_mse(h_hat, true_h, main_path_weight=3.0):
    """
    对 h 的不同路径做加权 MSE
    h_hat/true_h: [B,10,S] = 5条路径 * (Re,Im)
    默认认为主径是 path0 -> 通道(0,1)
    """
    device = h_hat.device
    # 5条路径权重（你也可以自己改成更细的，比如 [3,1.5,1,1,1]）
    path_w = torch.tensor([main_path_weight, 1.2, 1.0, 1.0, 1.0], device=device)  # [5]
    ch_w = torch.repeat_interleave(path_w, repeats=2)  # [10] 对应 Re/Im

    diff2 = (h_hat - true_h) ** 2                      # [B,10,S]
    diff2 = diff2 * ch_w.view(1, -1, 1)                # 加权

    # 用权重做归一化，让 loss 尺度稳定
    denom = ch_w.mean()                                # 标量
    return diff2.mean() / denom


def train_joint_i2sb(
    model,
    train_loader,
    val_loader,
    n_steps=20,
    epochs=200,
    lr=2e-4,
    device="cuda",
    save_dir="./results",
    patience=10,
    sps=16,
    snr_min=-3.0,
    snr_max=22.0,
    lambda_h=3.0,
    main_path_weight=3.0,
    grad_clip=1.0,
):
    os.makedirs(save_dir, exist_ok=True)

    mse = nn.MSELoss()
    opt = optim.AdamW(model.parameters(), lr=lr)
    sch = optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.8, patience=3)

    best_val = 1e18
    bad_epochs = 0

    # 记录：总 / 波形 / 信道
    tr_total, tr_wave, tr_h = [], [], []
    va_total, va_wave, va_h = [], [], []

    print("开始训练：Joint I2SB（不喂 y，只喂 x_t + est_h）")
    print(f"设备={device} | steps={n_steps} | epochs={epochs} | bs={train_loader.batch_size}")
    print(f"Loss = L_wave + {lambda_h}*L_h(weighted), 主径权重={main_path_weight}")

    for ep in range(1, epochs + 1):
        # -------------------
        # train
        # -------------------
        model.train()
        sum_loss = 0.0
        sum_lw = 0.0
        sum_lh = 0.0

        pbar = tqdm(train_loader, desc=f"第 {ep}/{epochs} 轮", ncols=120, leave=False)
        for clean_x, impaired_y, est_h, true_h, _ in pbar:
            clean_x = clean_x.to(device).float()        # [B,2,L]
            impaired_y = impaired_y.to(device).float()  # [B,2,L]
            est_h = est_h.to(device).float()            # [B,10,S]
            true_h = true_h.to(device).float()          # [B,10,S]

            B, _, L = clean_x.shape

            # 1) 动态加噪（符号SNR -> 采样SNR）
            snr_sym  = torch.rand(B, 1, device=device) * (snr_max - snr_min) + snr_min
            snr_samp = snr_sym - 10 * math.log10(sps) + 10 * math.log10(2)
            y_noisy = add_awgn_noise_torch(impaired_y, snr_samp)  # [B,2,L]

            # 2) 采样 t ∈ [0,1]，并得到离散 t_idx
            t = torch.rand(B, device=device).view(-1, 1, 1)        # [B,1,1]
            t_idx = (t.view(-1) * (n_steps - 1)).long()            # [B]

            # 3) x-pre：x_t = (1-t)*x0 + t*y
            x_t = (1 - t) * clean_x + t * y_noisy                  # [B,2,L]

            # 4) 输入：x_t + est_h_expanded（不喂 y）
            est_h_exp = expand_h_to_wave(est_h, sps=sps)            # [B,10,L]
            net_input = torch.cat([x_t, est_h_exp], dim=1)          # [B,12,L]

            # 5) forward
            opt.zero_grad()
            x0_hat, h_hat = model(net_input, t_idx)                # [B,2,L], [B,10,S]

            # 6) losses：分别计算，再组合
            lw = mse(x0_hat, clean_x)
            lh = weighted_h_mse(h_hat, true_h, main_path_weight=main_path_weight)
            loss = lw + lambda_h * lh

            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            opt.step()

            sum_loss += loss.item()
            sum_lw += lw.item()
            sum_lh += lh.item()

            pbar.set_postfix({
                "总": f"{loss.item():.4f}",
                "波形": f"{lw.item():.4f}",
                "信道": f"{lh.item():.4f}",
            })

        avg_tr = sum_loss / len(train_loader)
        avg_tr_w = sum_lw / len(train_loader)
        avg_tr_h = sum_lh / len(train_loader)

        tr_total.append(avg_tr)
        tr_wave.append(avg_tr_w)
        tr_h.append(avg_tr_h)

        # -------------------
        # val
        # -------------------
        model.eval()
        v_sum = 0.0
        v_lw = 0.0
        v_lh = 0.0

        with torch.no_grad():
            for clean_x, impaired_y, est_h, true_h, _ in val_loader:
                clean_x = clean_x.to(device).float()
                impaired_y = impaired_y.to(device).float()
                est_h = est_h.to(device).float()
                true_h = true_h.to(device).float()

                B, _, L = clean_x.shape

                snr_sym  = torch.rand(B, 1, device=device) * (snr_max - snr_min) + snr_min
                snr_samp = snr_sym - 10 * math.log10(sps) + 10 * math.log10(2)
                y_noisy = add_awgn_noise_torch(impaired_y, snr_samp)

                t = torch.rand(B, device=device).view(-1, 1, 1)
                t_idx = (t.view(-1) * (n_steps - 1)).long()

                x_t = (1 - t) * clean_x + t * y_noisy
                est_h_exp = expand_h_to_wave(est_h, sps=sps)
                net_input = torch.cat([x_t, est_h_exp], dim=1)

                x0_hat, h_hat = model(net_input, t_idx)

                lw = mse(x0_hat, clean_x)
                lh = weighted_h_mse(h_hat, true_h, main_path_weight=main_path_weight)
                loss = lw + lambda_h * lh

                v_sum += loss.item()
                v_lw += lw.item()
                v_lh += lh.item()

        avg_val = v_sum / len(val_loader)
        avg_val_w = v_lw / len(val_loader)
        avg_val_h = v_lh / len(val_loader)

        va_total.append(avg_val)
        va_wave.append(avg_val_w)
        va_h.append(avg_val_h)

        sch.step(avg_val)
        lr_now = opt.param_groups[0]["lr"]

        print(
            f"[第 {ep} 轮] "
            f"训练: 总={avg_tr:.6f}, 波形={avg_tr_w:.6f}, 信道={avg_tr_h:.6f} | "
            f"验证: 总={avg_val:.6f}, 波形={avg_val_w:.6f}, 信道={avg_val_h:.6f} | "
            f"LR={lr_now:.2e}"
        )

        # checkpoint + early stop
        if avg_val < best_val:
            best_val = avg_val
            bad_epochs = 0
            ckpt = os.path.join(save_dir, f"best_joint_i2sb_steps{n_steps}.pth")
            torch.save(model.state_dict(), ckpt)
            print(f"✅ 保存最优模型：{ckpt}")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print(f"⏹️ 触发早停：patience={patience}，best_val={best_val:.6f}")
                break

    # -------------------
    # plots
    # -------------------
    plt.figure()
    plt.plot(tr_total, label="训练-总")
    plt.plot(va_total, label="验证-总")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.title("Joint I2SB 总损失")
    plt.savefig(os.path.join(save_dir, "loss_total.png"), dpi=200)

    plt.figure()
    plt.plot(tr_wave, label="训练-波形")
    plt.plot(va_wave, label="验证-波形")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.title("Joint I2SB 波形损失")
    plt.savefig(os.path.join(save_dir, "loss_wave.png"), dpi=200)

    plt.figure()
    plt.plot(tr_h, label="训练-信道(加权)")
    plt.plot(va_h, label="验证-信道(加权)")
    plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.title("Joint I2SB 信道损失（主径加权）")
    plt.savefig(os.path.join(save_dir, "loss_h.png"), dpi=200)

    print("训练结束，曲线已保存。")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = r"F:\LJN\bishe\bishe\Estimate\data\tdl_data_5h"
    save_dir = r"F:\LJN\bishe\bishe\Estimate\tdl\results"

    n_steps = 20
    sps = 16
    h_symbols = 11

    batch_size = 256
    epochs = 500

    model = SETimeResNet1D(
        in_channels=12,         # x_t(2) + est_h_expanded(10)
        out_wave_channels=2,
        out_h_channels=10,
        hidden_dim=128,
        num_blocks=12,
        time_emb_dim=128,
        sps=sps,
        h_symbols=h_symbols,
    ).to(device)

    train_loader, val_loader = get_train_loader(
        data_dir=data_dir,
        start=0,
        end=400000,
        batch_size=batch_size,
        val_split=0.1,
        expand_h_to_wave=False,
        shuffle=True,
    )

    train_joint_i2sb(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        n_steps=n_steps,
        epochs=epochs,
        lr=2e-4,
        device=device,
        save_dir=save_dir,
        patience=10,
        sps=sps,
        snr_min=-3.0,
        snr_max=22.0,
        lambda_h=0.5,
        main_path_weight=3.0,   # ✅ 主径更重要：这里调
        grad_clip=1.0,
    )
