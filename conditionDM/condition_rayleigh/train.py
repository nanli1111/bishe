import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F

# 自定义模块
from model.unet import UNet
from conditionDM.condition_rayleigh.ddpm import CDDM
from dataset.dataset import get_train_QPSKdataloader 


def train_cddm(model,
               cddm,
               train_loader,
               val_loader,
               epochs=50,
               lr=1e-4,
               device='cuda',
               save_dir='./results',
               patience=6,
               sigma=0.3):
    """
    使用 CDDM 进行训练，对应论文 Algorithm 1：
    - 随机采样 t
    - 用真实信道 h_c 计算 H_r, W_s, W_n
    - 采样 eps ~ N(0, I)
    - 最小化 || eps - eps_theta(x_t, h_r, t) ||^2
    """
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    val_loss_history = []

    model.train()

    best_val_loss = float('inf')
    epochs_since_improvement = 0

    sigma_t = torch.tensor(sigma, device=device)

    for epoch in range(1, epochs + 1):
        epoch_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")

        # ========= 训练一个 epoch =========
        for x, _, h_raw in pbar:
            # x: [B, 2, L]（QPSK I/Q 波形）
            x = x.to(device).float()
            h_raw = h_raw.to(device).float()      # [B, 2] 或 [B, 2, 1] / [B, 2, Lh]

            B = x.size(0)

            # 展平信号： [B, 2, L] -> [B, 2k]
            x_flat = x.view(B, -1)                # [B, 2k]
            two_k = x_flat.size(1)
            k = two_k // 2                        # 子载波数量 / 序列长度

            # 用数据集里的 h 构造 [B, k] 的复信道向量
            seq_len = x.size(2)
            h = h_raw[:, :, np.newaxis]         # (N,2,1)
            # 将最后一维从 1 repeat 到 48
            h = h.repeat(1, 1, seq_len) # (N, 2, 48）
            h_real = h[:, 0, :]      # [B,L]
            h_imag = h[:, 1, :]      # [B,L]
            h_c = torch.complex(h_real, h_imag)   # [B,L]
            # 1) 随机时间步 t ∈ {1,...,n_steps-1}
            t = torch.randint(1, cddm.n_steps, (B,), device=device).long()

            # 2) 计算 W_s, W_n, h_r（对应论文公式 (6)）
            W_s, W_n, h_r = cddm.compute_W_matrices(h_c, sigma_t)
            # W_s, W_n: [B, 2k, 2k]; h_r: [B, 2k]

            # 3) 计算 x_0 = W_s * x，并加噪得到 x_t
            x0_flat = torch.matmul(W_s, x_flat.unsqueeze(-1)).squeeze(-1)  # [B, 2k]

            # 采样 eps ~ N(0, I_{2k})
            eps = torch.randn_like(x0_flat)        # [B, 2k]

            # alpha_bar_t: [B, 1]
            alpha_bar_t = cddm.alpha_bars[t].unsqueeze(1)  # [B,1]

            # W_n * eps
            noise_flat = torch.matmul(W_n, eps.unsqueeze(-1)).squeeze(-1)  # [B, 2k]

            # x_t = sqrt(alpha_bar)*x0 + sqrt(1-alpha_bar)*W_n*eps
            x_t_flat = torch.sqrt(alpha_bar_t) * x0_flat \
                       + torch.sqrt(1 - alpha_bar_t) * noise_flat          # [B, 2k]

            # 还原成 [B, 2, L] 输入网络
            C = x.size(1)              # =2
            L = x.size(2)              # =k
            x_t = x_t_flat.view(B, C, L)           # [B, 2, L]

            # t 作为条件：形状 [B, 1]
            t_cond = t.view(B, 1)

            # h_r: [B, 2k] -> [B, 2, L]
            h_r_img = h_r.view(B, 2, k)           # [B, 2, L]

            # 拼成 4 通道输入：2 路 x_t + 2 路 h_r
            net_in = torch.cat([x_t, h_r_img], dim=1)   # [B, 4, L]

            # 4) 网络预测 eps_theta(x_t, h_r, t)
            eps_pred_img = model(net_in, t_cond)        # [B, 2, L]

            # 展平成 [B, 2L] = [B, 2k]
            eps_pred = eps_pred_img.view(B, -1)         # [B, 2k]

            # 5) 损失：|| eps - eps_theta ||^2
            loss = F.mse_loss(eps_pred, eps)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch}] Avg Train Loss: {avg_loss:.6f}")

        # ========= 验证 =========
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _, h_raw in val_loader:
                x = x.to(device).float()
                h_raw = h_raw.to(device).float()

                B = x.size(0)
                x_flat = x.view(B, -1)
                two_k = x_flat.size(1)
                k = two_k // 2

                seq_len = x.size(2)
                h = h_raw[:, :, np.newaxis]         # (N,2,1)
                # 将最后一维从 1 repeat 到 48
                h = h.repeat(1, 1, seq_len) # (N, 2, 48）
                h_real = h[:, 0, :]      # [B,L]
                h_imag = h[:, 1, :]      # [B,L]
                h_c = torch.complex(h_real, h_imag)   # [B,L]

                t = torch.randint(1, cddm.n_steps, (B,), device=device).long()

                W_s, W_n, h_r = cddm.compute_W_matrices(h_c, sigma_t)

                x0_flat = torch.matmul(W_s, x_flat.unsqueeze(-1)).squeeze(-1)
                eps = torch.randn_like(x0_flat)

                alpha_bar_t = cddm.alpha_bars[t].unsqueeze(1)
                noise_flat = torch.matmul(W_n, eps.unsqueeze(-1)).squeeze(-1)
                x_t_flat = torch.sqrt(alpha_bar_t) * x0_flat \
                           + torch.sqrt(1 - alpha_bar_t) * noise_flat

                C = x.size(1)
                L = x.size(2)
                x_t = x_t_flat.view(B, C, L)
                t_cond = t.view(B, 1)

                # h_r: [B, 2k] -> [B, 2, L]
                h_r_img = h_r.view(B, 2, k)               # [B, 2, L]

                # 拼成 4 通道输入
                net_in = torch.cat([x_t, h_r_img], dim=1) # [B, 4, L]

                eps_pred_img = model(net_in, t_cond)      # [B, 2, L]
                eps_pred = eps_pred_img.view(B, -1)       # [B, 2k]

                loss = F.mse_loss(eps_pred, eps)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"[Epoch {epoch}] Avg Validation Loss: {avg_val_loss:.6f}")

        # ========= 早停/过拟合检测 =========
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            print("Validation loss improved, saving model...")
            torch.save(
                model.state_dict(),
                os.path.join(save_dir, f"best_model_epoch_with_n_steps{cddm.n_steps}.pth")
            )
        else:
            epochs_since_improvement += 1
            print(f"Validation loss did not improve at epoch {epoch}. "
                  f"{epochs_since_improvement}/{patience} epochs without improvement.")

        if epochs_since_improvement >= patience:
            print(f"⚠️ Overfitting detected! Validation loss has not improved for {patience} epochs.")
            print(f"Stopping training early at epoch {epoch}.")
            break

        model.train()

    # ===== 绘制训练/验证曲线 =====
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CDDM QPSK Training Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CDDM QPSK Validation Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'validation_loss_curve.png'))
    plt.close()

    print("✅ CDDM Training finished and results saved!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 150        # 扩散步数
    batch_size = 64
    epochs = 1000
    lr = 1e-4

    # ===== 构建网络（直接 new，不用 cfg） =====
    model = UNet(
        n_steps=n_steps,
        channels=[10, 20, 40, 80],
        pe_dim=128,
        residual=False,   # 如果用残差版就 True
        in_channels=4,    # 2 路 x_t + 2 路 h_r
    ).to(device)

    # ===== 构建 CDDM 对象 =====
    cddm = CDDM(
        device=device,
        n_steps=n_steps,
        channel_type='rayleigh',   # 或 'awgn'
        min_beta=1e-4,
        max_beta=0.02
    )

    # ===== 数据加载 =====
    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.2
    )

    # ===== 训练 CDDM =====
    train_cddm(
        model, cddm,
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        device=device,
        save_dir='CDDM/cddm_rayleigh/results',
        patience=10,
        sigma=0.6
    )
