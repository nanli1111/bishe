import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model.unet import build_network
from ddrm_core import DDRM
from dataset.dataset import get_train_QPSKdataloader

# 辅助函数：提取系数
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def train_ddrm_hybrid_loss(model, ddrm, train_loader, val_loader, 
                           epochs=50, lr=1e-4, device='cuda', 
                           save_dir='./results', patience=10, 
                           p_uncond=0.15, lambda_recon=1.0): 
    
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    # 预先获取 DDPM 反推系数 (用于从 epsilon 反推 x0)
    # x0 = (x_t - sqrt(1-ab) * eps) / sqrt(ab)
    # sqrt_recip_alphas_cumprod: 1/sqrt(ab)
    # sqrt_recipm1_alphas_cumprod: sqrt(1-ab)/sqrt(ab)
    ddrm.register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / ddrm.alpha_bars))
    ddrm.register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / ddrm.alpha_bars - 1))

    print(f"Start Training (Epsilon-Pred + Recon Loss, p={p_uncond}, lambda={lambda_recon})...")

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        epoch_mse_loss = 0
        epoch_recon_loss = 0
        
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

            # 2. 扩散加噪 (Standard DDPM forward)
            t = torch.randint(0, ddrm.n_steps, (batch_size,), device=device).long()
            noise = torch.randn_like(clean_x)
            
            coef1 = extract(ddrm.sqrt_alphas_cumprod, t, clean_x.shape)
            coef2 = extract(ddrm.sqrt_one_minus_alphas_cumprod, t, clean_x.shape)
            x_t = coef1 * clean_x + coef2 * noise

            # 3. CFG Masking
            mask = torch.rand(batch_size, 1, 1, device=device) < p_uncond
            
            cond_y = faded_y.clone()
            cond_h = h_expanded.clone()
            
            if mask.any():
                cond_y.masked_fill_(mask, 0.0)
                cond_h.masked_fill_(mask, 0.0)

            # 4. 构造网络输入 [x_t, y, h]
            net_input = torch.cat([x_t, cond_y, cond_h], dim=1)

            # 5. 预测 (预测的是噪声 epsilon)
            optimizer.zero_grad()
            predicted_noise = model(net_input, t)
            
            # --- 损失计算 ---
            
            # A. 基础噪声损失 (保证扩散过程稳定)
            loss_mse = criterion(predicted_noise, noise)
            
            # B. 强监督重建损失 (让模型学会去噪和光滑化)
            # 反推 x0_hat
            c1 = extract(ddrm.sqrt_recip_alphas_cumprod, t, clean_x.shape)
            c2 = extract(ddrm.sqrt_recipm1_alphas_cumprod, t, clean_x.shape)
            x0_hat = c1 * x_t - c2 * predicted_noise
            
            # 直接对比 x0_hat 和 clean_x
            loss_recon = criterion(x0_hat, clean_x)
            
            # 总损失: 组合两者
            # 仅用 loss_recon 在 t 很大时梯度会不稳定，加上 loss_mse 会更稳健
            loss = loss_mse + lambda_recon * loss_recon
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            epoch_mse_loss += loss_mse.item()
            epoch_recon_loss += loss_recon.item()
            
            pbar.set_postfix({'MSE': loss_mse.item(), 'Recon': loss_recon.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0 
        val_mse = 0
        val_recon = 0
        
        with torch.no_grad():
            for clean_x, faded_y, h_est in val_loader:
                clean_x = clean_x.to(device)
                faded_y = faded_y.to(device)
                h_est = h_est.to(device)
                
                # 维度处理
                if h_est.dim() == 2:
                    h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else:
                    h_expanded = h_est

                t = torch.randint(0, ddrm.n_steps, (batch_size,), device=device).long()
                noise = torch.randn_like(clean_x)
                
                coef1 = extract(ddrm.sqrt_alphas_cumprod, t, clean_x.shape)
                coef2 = extract(ddrm.sqrt_one_minus_alphas_cumprod, t, clean_x.shape)
                x_t = coef1 * clean_x + coef2 * noise
                
                # 验证时不 Drop 条件
                net_input = torch.cat([x_t, faded_y, h_expanded], dim=1)
                
                predicted_noise = model(net_input, t)
                
                # 1. MSE Loss
                curr_mse = criterion(predicted_noise, noise)
                
                # 2. Recon Loss
                c1 = extract(ddrm.sqrt_recip_alphas_cumprod, t, clean_x.shape)
                c2 = extract(ddrm.sqrt_recipm1_alphas_cumprod, t, clean_x.shape)
                x0_hat = c1 * x_t - c2 * predicted_noise
                curr_recon = criterion(x0_hat, clean_x)
                
                # 总 Loss (保持与训练一致的权重)
                total_loss = curr_mse + lambda_recon * curr_recon
                
                val_loss += total_loss.item()
                val_mse += curr_mse.item()
                val_recon += curr_recon.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mse = val_mse / len(val_loader)
        avg_val_recon = val_recon / len(val_loader)
        
        val_loss_history.append(avg_val_loss)
        
        print(f"[Epoch {epoch}] Val Total: {avg_val_loss:.6f} | MSE: {avg_val_mse:.6f} | Recon: {avg_val_recon:.6f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            # 保存时加上 _hybrid 标记
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_with_n_step{ddrm.n_steps}.pth"))
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print("Early stopping.")
            break

    # 绘图代码略...

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100
    batch_size = 64
    epochs = 500
    
    # 网络配置 (保持 6 通道输入，输出 2 通道噪声)
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 6,
        'out_channels': 2
    }
    
    model = build_network(net_cfg, n_steps).to(device)
    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    train_ddrm_hybrid_loss(
        model, ddrm,
        train_loader, val_loader,
        epochs=epochs,
        device=device,
        save_dir='ddrm/ddrm_rayleigh/results',
        p_uncond=0.15,
        lambda_recon=2.0 # 重建损失权重
    )