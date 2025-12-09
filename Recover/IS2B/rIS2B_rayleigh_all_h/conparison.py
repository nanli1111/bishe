import os
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# === 引入项目模块 ===
from model.unet import build_network
from dataset.dataset import get_train_QPSKdataloader
from test_fig_x_pre import add_awgn_noise_torch

def train_supervised_pure(model, train_loader, val_loader, 
                          epochs=100, lr=1e-4, device='cuda', 
                          save_dir='./results_supervised_pure', 
                          sps=16, patience=5): 
    
    # 1. 初始化
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    
    # 早停计数器
    epochs_no_improve = 0
    
    # 计算固定的采样 SNR (Symbol SNR = 10dB)
    target_symbol_snr = 10.0
    fixed_sample_snr = target_symbol_snr - 10 * math.log10(sps)
    
    print(f"🚀 开始纯监督训练 (Fixed SNR={target_symbol_snr}dB)...")
    print(f"设备: {device}, Epochs: {epochs}, Patience: {patience}")
    print(f"Data Save Dir: {save_dir}")

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

            # 1. h 维度扩展
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # 2. 固定 SNR 加噪
            # 不再生成随机 SNR，直接使用 fixed_sample_snr
            noisy_y = add_awgn_noise_torch(faded_y, fixed_sample_snr)

            # 3. 构造输入 (t=0)
            t_dummy = torch.zeros(batch_size, device=device, dtype=torch.long)
            
            # 拼接: [Noisy_Y, H]
            # [修复] 修正了你原代码中的 'no sy_y' 拼写错误
            net_input = torch.cat([noisy_y, h_expanded], dim=1)

            # 4. 前向与反向
            optimizer.zero_grad()
            predicted_x = model(net_input, t_dummy)
            
            loss = criterion(predicted_x, clean_x)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Train MSE': f"{loss.item():.5f}"})

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
                
                seq_len = clean_x.shape[2]
                batch_size = clean_x.shape[0]

                if h_est.dim() == 2:
                    h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else:
                    h_expanded = h_est

                # 验证也使用相同的固定 SNR
                noisy_y = add_awgn_noise_torch(faded_y, fixed_sample_snr)
                
                t_dummy = torch.zeros(batch_size, device=device, dtype=torch.long)
                net_input = torch.cat([noisy_y, h_expanded], dim=1)
                
                predicted_x = model(net_input, t_dummy)
                
                loss = criterion(predicted_x, clean_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        # 更新学习率
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch}] Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        # =======================
        # 保存与早停策略 (Early Stopping)
        # =======================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0  # 重置计数器
            # 保存权重
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_supervised.pth"))
            print("--> Best Model Saved.")
        else:
            epochs_no_improve += 1
            print(f"--> No improvement for {epochs_no_improve} epochs.")

        # 检查是否触发早停
        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered! Validation loss hasn't improved for {patience} epochs.")
            break

    # =======================
    #      训练结束画图
    # =======================
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='Train MSE')
    plt.plot(val_loss_history, label='Val MSE')
    plt.title(f'MSE Loss Curve (Fixed SNR={target_symbol_snr}dB)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    print("训练结束，Loss 曲线已保存。")

if __name__ == "__main__":
    # === 参数配置 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100     
    batch_size = 64
    epochs = 100      
    sps = 16 
    
    # === 1. 数据准备 ===
    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    # === 2. 模型构建 ===
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 4,  
        'out_channels': 2  
    }
    
    print(f"Building Model on {device}...")
    model = build_network(net_cfg, n_steps).to(device)
    
    # === 3. 开始训练 ===
    train_supervised_pure(
        model, 
        train_loader, 
        val_loader,
        epochs=epochs,
        device=device,
        save_dir='IS2B/rIS2B_rayleigh_all_h/results',
        sps=sps,
        patience=5 # 设置耐心值为 5
    )