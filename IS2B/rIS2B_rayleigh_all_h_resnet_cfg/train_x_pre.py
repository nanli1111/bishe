import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# === 引入项目模块 ===
# 1. 导入新的 ResNet 模型 
from model.resnet import TimeResNet1D

# 2. 导入 IS2B 包装器
from IS2B_x_pre import IS2B

# 3. 数据集与加噪函数
from dataset.dataset import get_train_QPSKdataloader
from test_fig_x_pre import add_awgn_noise_torch

def train_IS2B_resnet(model, is2b, train_loader, val_loader, 
                      epochs=50, lr=2e-4, device='cuda', 
                      save_dir='./results_resnet_is2b', patience=10, 
                      sps=16, cfg_prob=0.1): # <--- 新增 cfg_prob 参数
    
    # 1. 初始化设置
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss() 
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    
    # === 学习率调度器 ===
    # ReduceLROnPlateau: 当 val_loss 不下降时减小学习率
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.8, patience=2)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    print(f"Start Training IS2B with TimeResNet1D...")
    print(f"Device: {device}, Epochs: {epochs}, Batch Size: {train_loader.batch_size}")
    print(f"Scheduler: ReduceLROnPlateau (Patience=2, Factor=0.8)")
    print(f"CFG Training Enabled: Probability = {cfg_prob}") # 打印 CFG 设置

    for epoch in range(1, epochs + 1):
        # =======================
        #      训练阶段
        # =======================
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for clean_x, faded_y, h_est in pbar:
            clean_x = clean_x.to(device).float()
            faded_y = faded_y.to(device).float()
            h_est = h_est.to(device).float()
            
            batch_size = clean_x.shape[0]
            seq_len = clean_x.shape[2]

            # 1. 扩展 h 维度
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # 2. 动态加噪 (Data Augmentation)
            snr_min, snr_max = 0.0, 20.0
            random_symbol_snr = (torch.rand(batch_size, 1, 1, device=device) * (snr_max - snr_min) + snr_min)
            random_sample_snr = random_symbol_snr - 10 * math.log10(sps)
            
            noisy_y = add_awgn_noise_torch(faded_y, random_sample_snr)

            # 3. 生成随机时间步 t \in [0, 1]
            t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
            t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()

            # 4. 构造 I2SB 中间态 (Linear Interpolation)
            x_t = (1 - t_float) * clean_x + t_float * noisy_y

            # ====================================================
            # 5. CFG 核心逻辑: 随机丢弃条件 h (Unconditional Training)
            # ====================================================
            # 生成一个随机数，如果小于 cfg_prob，则将 h 置零
            if np.random.random() < cfg_prob:
                h_input = torch.zeros_like(h_expanded)
            else:
                h_input = h_expanded

            # 6. 构造网络输入 [x_t, h]
            # 注意：这里使用的是 h_input (可能是 h 也可能是 0)
            net_input = torch.cat([x_t, h_input], dim=1)

            # 7. 预测与反向传播
            optimizer.zero_grad()
            
            # 模型预测 x0 (传入 t_idx)
            predicted_x0 = model(net_input, t_idx)
            
            loss = criterion(predicted_x0, clean_x)
            loss.backward()
            
            # 梯度裁剪
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
        
        with torch.no_grad():
            for i, (clean_x, faded_y, h_est) in enumerate(val_loader):
                clean_x = clean_x.to(device).float()
                faded_y = faded_y.to(device).float()
                h_est = h_est.to(device).float()
                
                seq_len = clean_x.shape[2]
                batch_size = clean_x.shape[0]

                if h_est.dim() == 2:
                    h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else:
                    h_expanded = h_est

                # 验证 SNR
                random_symbol_snr = (torch.rand(batch_size, 1, 1, device=device) * (snr_max - snr_min) + snr_min)
                random_sample_snr = random_symbol_snr - 10 * math.log10(sps)
                noisy_y = add_awgn_noise_torch(faded_y, random_sample_snr)

                t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
                t_idx = (t_float.view(-1) * (is2b.n_steps - 1)).long()
                
                x_t = (1 - t_float) * clean_x + t_float * noisy_y
                
                # 验证时始终使用完整的条件 h (不进行 Masking)
                net_input = torch.cat([x_t, h_expanded], dim=1)
                
                predicted_x0 = model(net_input, t_idx)
                loss = criterion(predicted_x0, clean_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # === 更新学习率 ===
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        # =======================
        #      保存策略
        # =======================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_IS2B_resnet.pth"))
            print("--> Best Model Saved.")
        else:
            epochs_since_improvement += 1

        # 早停
        if epochs_since_improvement >= patience:
            print(f"Early stopping triggered after {patience} epochs.")
            break
            
    # 画图
    plt.figure()
    plt.plot(loss_history, label='Train Loss')
    plt.plot(val_loss_history, label='Val Loss')
    plt.yscale('log')
    plt.legend()
    plt.title('IS2B Training Loss (TimeResNet1D + CFG)')
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print("Training Finished.")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 20 
    batch_size = 64
    epochs = 500
    sps = 16 
    
    # 1. 实例化 TimeResNet1D
    print(f"Building TimeResNet1D on {device}...")
    model = TimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=128,  
        num_blocks=8,    
        time_emb_dim=64  
    ).to(device)
    
    # 2. IS2B 包装器
    is2b_wrapper = IS2B(model, n_steps=n_steps, device=device)

    # 3. 数据加载
    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    # 4. 开始训练
    # 这里的 save_dir 修改了一下以区分之前的实验
    train_IS2B_resnet(
        model, is2b_wrapper,
        train_loader, val_loader,
        epochs=epochs,
        device=device,
        save_dir='IS2B/rIS2B_rayleigh_all_h_resnet_cfg/results', 
        sps=sps,
        patience=10,
        cfg_prob=0.1 # 开启 CFG 训练，10% 概率丢弃条件
    )