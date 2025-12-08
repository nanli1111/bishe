import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# 引入项目模块
from IS2B_x_pre import IS2B
from dataset.dataset import get_train_QPSKdataloader
from test_fig_x_pre import add_awgn_noise_torch
from model.resnet_large_kernel import LargeKernelTimeResNet1D

def train_IS2B_resnet(model, is2b, train_loader, val_loader, 
                      epochs=50, lr=2e-4, device='cuda', 
                      save_dir='./results_resnet_is2b', patience=10, 
                      sps=16, 
                      # === 课程学习参数 ===
                      curriculum_epochs=30,  # 前30个Epoch逐渐增加难度
                      start_snr_min=15.0,    # 初始最小SNR (较简单)
                      target_snr_min=0.0     # 最终最小SNR (较难)
                      ): 
    
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=3)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    print(f"Start Training IS2B with Curriculum Learning...")
    print(f"Curriculum: SNR min drops from {start_snr_min}dB to {target_snr_min}dB over {curriculum_epochs} epochs.")

    for epoch in range(1, epochs + 1):
        # ==========================================
        # 1. 计算当前 Epoch 的 SNR 下限 (课程学习)
        # ==========================================
        if epoch <= curriculum_epochs:
            # 线性下降
            progress = (epoch - 1) / (curriculum_epochs - 1) if curriculum_epochs > 1 else 1.0
            current_snr_min = start_snr_min - progress * (start_snr_min - target_snr_min)
        else:
            # 课程结束，保持最难
            current_snr_min = target_snr_min
            
        # 设置 SNR 上限 (通常保持较高值，比如 20dB 或 25dB)
        current_snr_max = 22.0 
        
        # 打印当前难度
        if epoch % 5 == 1:
            print(f"[Curriculum Info] Epoch {epoch}: Training SNR range [{current_snr_min:.1f}, {current_snr_max:.1f}] dB")

        # =======================
        #      训练阶段
        # =======================
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} (SNR>{current_snr_min:.1f}dB)", leave=False, mininterval=2.0)
        
        for clean_x, faded_y, h_est in pbar:
            clean_x = clean_x.to(device).float()
            faded_y = faded_y.to(device).float()
            h_est = h_est.to(device).float()
            
            batch_size = clean_x.shape[0]
            seq_len = clean_x.shape[2]

            # 扩展 h
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # === 动态加噪 (基于当前课程难度) ===
            # 在 [current_snr_min, current_snr_max] 之间均匀采样
            random_symbol_snr = (torch.rand(batch_size, 1, 1, device=device) * (current_snr_max - current_snr_min) + current_snr_min)
            random_sample_snr = random_symbol_snr - 10 * math.log10(sps) 
            
            noisy_y = add_awgn_noise_torch(faded_y, random_sample_snr)

            # ... (后续训练逻辑保持不变) ...
            # 3. 生成随机时间步 t
            t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
            t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()

            # 4. 构造 I2SB 中间态
            x_t = (1 - t_float) * clean_x + t_float * noisy_y

            # 5. 构造输入
            net_input = torch.cat([x_t, h_expanded], dim=1)

            # 6. 预测与反向传播
            optimizer.zero_grad()
            predicted_x0 = model(net_input, t_idx)
            loss = criterion(predicted_x0, clean_x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Loss': f"{loss.item():.5f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)

        # =======================
        #      验证阶段
        # =======================
        model.eval()
        val_loss = 0
        
        # 验证时我们通常固定一个具有挑战性的 SNR (例如 10dB)，或者全范围随机
        # 这里为了稳定监控指标，建议固定一个中等难度
        val_fixed_snr = 10.0
        
        with torch.no_grad():
            for i, (clean_x, faded_y, h_est) in enumerate(val_loader):
                clean_x = clean_x.to(device).float()
                faded_y = faded_y.to(device).float()
                h_est = h_est.to(device).float()
                
                seq_len = clean_x.shape[2]-0
                batch_size = clean_x.shape[0]

                if h_est.dim() == 2:
                    h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else:
                    h_expanded = h_est

                # 验证 SNR (固定)
                val_sample_snr = val_fixed_snr - 10 * math.log10(sps)
                noisy_y = add_awgn_noise_torch(faded_y, val_sample_snr)

                t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
                t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()
                x_t = (1 - t_float) * clean_x + t_float * noisy_y
                net_input = torch.cat([x_t, h_expanded], dim=1)
                
                predicted_x0 = model(net_input, t_idx)
                loss = criterion(predicted_x0, clean_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        # 保存策略
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_IS2B_resnet_pro_crum_andlarge_{n_steps}.pth"))
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
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 20 
    batch_size = 64
    epochs = 500
    sps = 16 
    
    print(f"Building DilatedTimeResNet1D on {device}...")
    model = LargeKernelTimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=256,   # 宽度：256 能承载更多信息
        num_blocks=8,     # 深度：8 层足够，因为单层感受野很大
        kernel_size=7,    # 核心：使用 7 或 9 的卷积核
        time_emb_dim=128
    ).to(device)
    is2b_wrapper = IS2B(model, n_steps=n_steps, device=device)

    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    train_IS2B_resnet(
        model, is2b_wrapper,
        train_loader, val_loader,
        epochs=epochs,
        device=device,
        save_dir='IS2B/rIS2B_nakagmi_resnet_adjust/results', 
        sps=sps,
        patience=10,
        # 课程学习配置
        curriculum_epochs=30, # 前50个epoch逐渐增加难度
        start_snr_min=15.0,   # 从 15dB 开始
        target_snr_min=0.0    # 最终降到 0dB
    )