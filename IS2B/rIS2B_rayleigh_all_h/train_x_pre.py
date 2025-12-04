import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math

# 引入项目模块
from model.unet import build_network
from IS2B_x_pre import IS2B
from dataset.dataset import get_train_QPSKdataloader

# === 关键修改：从 test_fig_x_pre 导入加噪函数 ===
from test_fig_x_pre import add_awgn_noise_torch

def train_IS2B_x0_pred(model, is2b, train_loader, val_loader, 
                       epochs=50, lr=1e-4, device='cuda', 
                       save_dir='./results', patience=10, 
                       sps=16): 
    
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss() 
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    print(f"Start Training IS2B (Target: x0, Linear Bridge with Dynamic AWGN, No Masking)...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for clean_x, faded_y, h_est in pbar:
            clean_x = clean_x.to(device)
            faded_y = faded_y.to(device)
            h_est = h_est.to(device)
            
            batch_size = clean_x.shape[0]
            seq_len = clean_x.shape[2]

            # 1. 扩展 h 维度
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # 2. 动态加噪
            snr_min, snr_max = 2.0, 18.0
            random_symbol_snr = (torch.rand(batch_size, 1, 1, device=device) * (snr_max - snr_min) + snr_min)
            random_sample_snr = random_symbol_snr - 10 * math.log10(sps)
            
            noisy_y = add_awgn_noise_torch(faded_y, random_sample_snr)

            # 3. 生成随机时间步 t \in [0, 1]
            t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
            t_idx = (t_float.squeeze() * (is2b.n_steps - 1)).long()

            # 4. 构造 I2SB 中间态 (Linear Interpolation)
            x_t = (1 - t_float) * clean_x + t_float * noisy_y

            # 5. 构造网络输入 [x_t, h] (4通道)
            # 移除了 CFG mask，始终传入完整的 h
            net_input = torch.cat([x_t, h_expanded], dim=1)

            # 6. 预测与反向传播
            optimizer.zero_grad()
            
            predicted_x0 = model(net_input, t_idx)
            loss = criterion(predicted_x0, clean_x)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Recon Loss': loss.item()})

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for i, (clean_x, faded_y, h_est) in enumerate(val_loader):
                clean_x = clean_x.to(device)
                faded_y = faded_y.to(device)
                h_est = h_est.to(device)
                
                batch_size = clean_x.shape[0]
                seq_len = clean_x.shape[2]

                if h_est.dim() == 2:
                    h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else:
                    h_expanded = h_est

                random_symbol_snr = (torch.rand(batch_size, 1, 1, device=device) * 35.0 - 5.0)
                random_sample_snr = random_symbol_snr - 10 * math.log10(sps)
                noisy_y = add_awgn_noise_torch(faded_y, random_sample_snr)

                t_float = torch.rand(batch_size, device=device).view(-1, 1, 1)
                t_idx = (t_float.squeeze() * (is2b.n_steps - 1)).long()
                
                x_t = (1 - t_float) * clean_x + t_float * noisy_y
                
                net_input = torch.cat([x_t, h_expanded], dim=1)
                
                predicted_x0 = model(net_input, t_idx)
                loss = criterion(predicted_x0, clean_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.6f}")

        # --- 保存策略 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_IS2B_n{is2b.n_steps}.pth"))
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print("Early stopping.")
            break
            
    # 画图
    plt.figure()
    plt.plot(loss_history, label='Train')
    plt.plot(val_loss_history, label='Val')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'loss.png'))

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 50 
    batch_size = 64
    epochs = 500
    sps = 16 
    
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 4,  
        'out_channels': 2 
    }
    
    model = build_network(net_cfg, n_steps).to(device)
    is2b_wrapper = IS2B(model, n_steps=n_steps, device=device)

    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    train_IS2B_x0_pred(
        model, is2b_wrapper,
        train_loader, val_loader,
        epochs=epochs,
        device=device,
        save_dir='IS2B/rIS2B_rayleigh_all_h/results',
        sps=sps
    )