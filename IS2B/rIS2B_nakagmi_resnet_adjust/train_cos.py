import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math


from model.resnet_pro import DilatedTimeResNet1D

from IS2B_x_pre import IS2B
from dataset.dataset import get_train_QPSKdataloader
from test_fig_x_pre import add_awgn_noise_torch

def train_IS2B_resnet_warmup(model, is2b, train_loader, val_loader, 
                             epochs=50, lr=5e-4, device='cuda',  # <--- 建议初始 LR 设大一点
                             save_dir='./results_resnet_warmup', patience=15, 
                             sps=16, 
                             l1_weight=0.7, # 保持之前的高 L1 权重建议
                             warmup_epochs=5): # <--- 新增：热身轮数
    
    os.makedirs(save_dir, exist_ok=True)
    
    # === 定义混合损失函数 ===
    criterion_l2 = nn.MSELoss() 
    criterion_l1 = nn.L1Loss()
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    
    # ==========================================
    # 核心修改：Warmup + Cosine 调度器组合
    # ==========================================
    # 1. 热身调度器：在前 warmup_epochs 轮，LR 从 lr*0.1 线性增加到 lr
    scheduler_warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    
    # 2. 主调度器：余弦退火，从 lr 降到 eta_min
    scheduler_cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs - warmup_epochs, eta_min=1e-6
    )
    
    # 3. 串联调度器：先跑 warmup，跑完后自动切换到 cosine
    # milestones=[warmup_epochs] 表示在第 5 个 epoch 切换
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[scheduler_warmup, scheduler_cosine], milestones=[warmup_epochs]
    )
    # ==========================================
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    print(f"🚀 Start Training with Warmup({warmup_epochs}) + Cosine...")
    print(f"Base LR: {lr:.2e}, Loss: {l1_weight}*L1 + {1-l1_weight}*L2")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
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

            # === 优化建议：加权 SNR 采样 (重低轻高) ===
            # 让 70% 的样本落在 0-10dB (难样本)，30% 落在 10-20dB
            r = torch.rand(batch_size, 1, 1, device=device)
            random_symbol_snr = torch.where(r < 0.7, r * (10/0.7), 10 + (r-0.7) * (10/0.3))
            random_sample_snr = random_symbol_snr - 10 * math.log10(sps)
            
            noisy_y = add_awgn_noise_torch(faded_y, random_sample_snr)

            # 生成随机时间步 t
            t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
            t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()

            # 构造 I2SB 中间态
            x_t = (1 - t_float) * clean_x + t_float * noisy_y

            # 构造网络输入
            net_input = torch.cat([x_t, h_expanded], dim=1)

            # 预测与反向传播
            optimizer.zero_grad()
            predicted_x0 = model(net_input, t_idx)
            
            # 混合 Loss
            loss = l1_weight * criterion_l1(predicted_x0, clean_x) + (1 - l1_weight) * criterion_l2(predicted_x0, clean_x)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.5f}"})

        # === 关键：更新学习率 (不再传入 val_loss) ===
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        avg_train_loss = epoch_loss / len(train_loader)
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

                # 验证 SNR 固定为 10dB
                val_sample_snr = 10.0 - 10 * math.log10(sps)
                noisy_y = add_awgn_noise_torch(faded_y, val_sample_snr)

                t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
                t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()
                
                x_t = (1 - t_float) * clean_x + t_float * noisy_y
                net_input = torch.cat([x_t, h_expanded], dim=1)
                
                predicted_x0 = model(net_input, t_idx)
                
                # 验证 Loss
                loss_v = l1_weight * criterion_l1(predicted_x0, clean_x) + (1 - l1_weight) * criterion_l2(predicted_x0, clean_x)
                val_loss += loss_v.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        print(f"[Epoch {epoch}] Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f} | LR: {current_lr:.2e}")

        # 保存策略 (只存最佳)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_IS2B_resnet_cos.pth"))
            print("--> Best Model Saved.")
        else:
            epochs_since_improvement += 1

        # 注意：Cosine 调度通常建议跑完全程，早停可以设得宽容一点
        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered.")
            break
            
    # 画图
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.yscale('log')
    plt.title(f'Training Loss (Warmup+Cosine, L1 weight={l1_weight})')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print("Training Finished.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 20 
    batch_size = 64
    epochs = 200 # Cosine 策略通常需要较长的 epochs 来充分下降
    sps = 16 
    
    # 确保导入的是升级版的 Pro 网络
    from model.resnet_pro import DilatedTimeResNet1D
    
    print(f"Building DilatedTimeResNet1D on {device}...")
    model = DilatedTimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=256,   # 按照建议加宽网络
        num_blocks=8,     # 稍微减少深度，换取宽度
        time_emb_dim=128
    ).to(device)
    
    is2b_wrapper = IS2B(model, n_steps=n_steps, device=device)

    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    train_IS2B_resnet_warmup(
        model, is2b_wrapper,
        train_loader, val_loader,
        epochs=epochs,
        device=device,
        save_dir='IS2B/rIS2B_nakagmi_resnet_adjust/results', 
        sps=sps,
        patience=20, # Cosine 可以在最后阶段才大幅下降，耐心要给足
        l1_weight=0.8 # 提高 L1 权重，增强锐度
    )