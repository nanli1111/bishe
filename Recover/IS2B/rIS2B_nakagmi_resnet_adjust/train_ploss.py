import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# === 引入项目模块 ===
# 建议使用能力更强的 Pro 版或 LargeKernel 版
try:
    from model.resnet_pro import DilatedTimeResNet1D as CurrentModel
except ImportError:
    from model.resnet_large_kernel import LargeKernelTimeResNet1D as CurrentModel

from IS2B_x_pre import IS2B
from dataset.dataset import get_train_QPSKdataloader
from test_fig_x_pre import add_awgn_noise_torch

def train_IS2B_phase_aware(model, is2b, train_loader, val_loader, 
                           epochs=50, lr=5e-4, device='cuda', 
                           save_dir='./results_phase_loss', patience=15, 
                           sps=16, 
                           l1_weight=1.0,    # L1 Loss 的基础权重
                           phase_weight=0.1, # 相位 Loss 的权重
                           warmup_epochs=5): 
    
    os.makedirs(save_dir, exist_ok=True)
    
    # === 定义损失函数 ===
    criterion_l1 = nn.L1Loss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # 学习率调度: Warmup + Cosine
    scheduler_warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs])
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    print(f"🚀 Start Training with Phase Loss & Weighted SNR...")
    print(f"Loss Config: L1_weight={l1_weight}, Phase_weight={phase_weight}")
    print(f"SNR Strategy: 70% (0-10dB), 30% (10-20dB)")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_phase_loss = 0 # 记录相位损失以便观察
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False, mininterval=2.0)
        
        for clean_x, faded_y, h_est in pbar:
            clean_x = clean_x.to(device).float()
            faded_y = faded_y.to(device).float()
            h_est = h_est.to(device).float()
            
            batch_size = clean_x.shape[0]
            seq_len = clean_x.shape[2]

            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # ==========================================
            # 1. 加权 SNR 采样 (Weighted SNR)
            # ==========================================
            # 生成 [0, 1] 随机数
            r = torch.rand(batch_size, 1, 1, device=device)
            
            # 定义分布：
            # r < 0.7 (70%概率): 映射到 [0, 10] dB
            # r >= 0.7 (30%概率): 映射到 [10, 20] dB
            # 这里的数学变换将 [0, 0.7] -> [0, 10], [0.7, 1.0] -> [10, 20]
            snr_db = torch.where(
                r < 0.7, 
                r * (10.0 / 0.7), 
                10.0 + (r - 0.7) * (10.0 / 0.3)
            )
            
            # 转换为采样点 SNR
            sample_snr = snr_db - 10 * math.log10(sps)
            
            # 加噪
            noisy_y = add_awgn_noise_torch(faded_y, sample_snr)

            # ==========================================
            # 2. 构造输入与前向传播
            # ==========================================
            t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
            t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()

            x_t = (1 - t_float) * clean_x + t_float * noisy_y
            net_input = torch.cat([x_t, h_expanded], dim=1)

            optimizer.zero_grad()
            predicted_x0 = model(net_input, t_idx)
            
            # ==========================================
            # 3. 计算混合损失 (L1 + Phase)
            # ==========================================
            # A. 基础 L1 Loss (关注幅度与数值)
            loss_l1 = criterion_l1(predicted_x0, clean_x)
            
            # B. Phase Loss (关注角度)
            # atan2(Imag, Real) -> atan2(Q, I)
            # Channel 1 is Q (Imag), Channel 0 is I (Real)
            pred_phase = torch.atan2(predicted_x0[:, 1, :], predicted_x0[:, 0, :])
            clean_phase = torch.atan2(clean_x[:, 1, :], clean_x[:, 0, :])
            
            # 计算差值并归一化到 [-pi, pi]
            diff = pred_phase - clean_phase
            # 这是一个处理角度周期的经典 trick
            angle_diff = torch.atan2(torch.sin(diff), torch.cos(diff))
            loss_phase = torch.mean(torch.abs(angle_diff))
            
            # C. 总 Loss
            loss = l1_weight * loss_l1 + phase_weight * loss_phase
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_phase_loss += loss_phase.item()
            
            pbar.set_postfix({
                'Total': f"{loss.item():.4f}", 
                'Phase': f"{loss_phase.item():.4f}"
            })

        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_train_loss = epoch_loss / len(train_loader)
        avg_phase_loss = epoch_phase_loss / len(train_loader)
        loss_history.append(avg_train_loss)

        # =======================
        #      验证阶段
        # =======================
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for clean_x, faded_y, h_est in val_loader:
                clean_x = clean_x.to(device).float()
                faded_y = faded_y.to(device).float()
                h_est = h_est.to(device).float()
                
                batch_size = clean_x.shape[0]
                seq_len = clean_x.shape[2]
                if h_est.dim() == 2: h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else: h_expanded = h_est

                # 验证 SNR 固定为 10dB (作为基准)
                val_sample_snr = 10.0 - 10 * math.log10(sps)
                noisy_y = add_awgn_noise_torch(faded_y, val_sample_snr)

                t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
                t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()
                
                x_t = (1 - t_float) * clean_x + t_float * noisy_y
                net_input = torch.cat([x_t, h_expanded], dim=1)
                
                predicted_x0 = model(net_input, t_idx)
                
                # 验证集同样计算混合 Loss
                l1_v = criterion_l1(predicted_x0, clean_x)
                
                p_ph = torch.atan2(predicted_x0[:, 1, :], predicted_x0[:, 0, :])
                c_ph = torch.atan2(clean_x[:, 1, :], clean_x[:, 0, :])
                diff_v = torch.atan2(torch.sin(p_ph - c_ph), torch.cos(p_ph - c_ph))
                ph_v = torch.mean(torch.abs(diff_v))
                
                loss_v = l1_weight * l1_v + phase_weight * ph_v
                val_loss += loss_v.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        print(f"[Epoch {epoch}] Train: {avg_train_loss:.5f} (Ph: {avg_phase_loss:.4f}) | Val: {avg_val_loss:.5f} | LR: {current_lr:.2e}")

        # 保存策略
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            # 保存时带上 tag 方便区分
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_phase_loss.pth"))
            print("--> Best Model Saved.")
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered.")
            break
            
    # 画图
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.yscale('log')
    plt.title(f'Training with Phase Loss (w={phase_weight})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve_phase.png'))
    print("Training Finished.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 20 
    batch_size = 64
    epochs = 200 # 建议多跑一些 epoch，因为相位损失的优化可能比较精细
    sps = 16 
    
    print(f"Building Model (DilatedTimeResNet1D) on {device}...")
    model = CurrentModel(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=256,   # 宽度 256
        num_blocks=12,    # 深度 12
        time_emb_dim=128
    ).to(device)
    
    is2b_wrapper = IS2B(model, n_steps=n_steps, device=device)

    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    train_IS2B_phase_aware(
        model, is2b_wrapper,
        train_loader, val_loader,
        epochs=epochs,
        device=device,
        save_dir='IS2B/rIS2B_nakagmi_phase_loss/results', 
        sps=sps,
        patience=20,
        l1_weight=1.0,    # 主损失权重
        phase_weight=0.2  # 建议设为 0.1 或 0.2，太大会导致幅度优化变慢
    )