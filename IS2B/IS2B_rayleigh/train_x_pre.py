import os
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from model.unet import build_network
from IS2B_x_pre import IS2B
from dataset.dataset import get_train_QPSKdataloader

# 辅助函数：提取系数
def extract(a, t, x_shape):
    batch_size = t.shape[0]
    out = a.gather(-1, t)
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

# 辅助函数：复数乘法 (用于验证阶段的物理一致性检查，可选)
def complex_mult(a, b):
    ar, ai = a[:, 0, :], a[:, 1, :]
    br, bi = b[:, 0, :], b[:, 1, :]
    real = ar * br - ai * bi
    imag = ar * bi + ai * br
    return torch.stack([real, imag], dim=1)

def train_IS2B_x0_pred(model, IS2B, train_loader, val_loader, 
                       epochs=50, lr=1e-4, device='cuda', 
                       save_dir='./results', patience=10, 
                       p_uncond=0.15): 
    
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss() # 使用 MSE 损失
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    print(f"Start Training (Target: x0-Prediction, p_uncond={p_uncond})...")

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

            # 1. 扩展 h 维度 [B, 2] -> [B, 2, L]
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # 2. 扩散加噪
            t = torch.randint(0, IS2B.n_steps, (batch_size,), device=device).long()
            noise = torch.randn_like(faded_y)
            
            # x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eps
            coef1 = extract(IS2B.sqrt_alphas_cumprod, t, faded_y.shape)
            coef2 = extract(IS2B.sqrt_one_minus_alphas_cumprod, t, faded_y.shape)
            x_t = coef1 * faded_y + coef2 * noise

            # 3. CFG Masking
            mask = torch.rand(batch_size, 1, 1, device=device) < p_uncond
            
            # 构造训练条件 (Clone 以免影响原数据)
            h_train = h_expanded.clone()
            
            if mask.any():
                h_train.masked_fill_(mask, 0.0)

            # 4. 构造网络输入 [x_t, y, h]
            net_input = torch.cat([x_t, h_train], dim=1)

            # 5. 预测与反向传播
            optimizer.zero_grad()
            
            # --- 关键修改：模型预测的是 x0 ---
            predicted_x0 = model(net_input, t)
            
            # --- 关键修改：Loss 计算 ---
            # 直接计算预测波形与真实波形的 MSE
            # 这是最直接的强监督，迫使模型学会去噪和光滑化
            loss = criterion(predicted_x0, clean_x)
            
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'Recon Loss': loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")

        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for clean_x, faded_y, h_est in val_loader:
                clean_x = clean_x.to(device)
                faded_y = faded_y.to(device)
                h_est = h_est.to(device)
                
                batch_size = clean_x.shape[0]
                seq_len = clean_x.shape[2]

                if h_est.dim() == 2:
                    h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else:
                    h_expanded = h_est

                t = torch.randint(0, IS2B.n_steps, (batch_size,), device=device).long()
                noise = torch.randn_like(faded_y)
                
                coef1 = extract(IS2B.sqrt_alphas_cumprod, t, faded_y.shape)
                coef2 = extract(IS2B.sqrt_one_minus_alphas_cumprod, t, faded_y.shape)
                x_t = coef1 * faded_y + coef2 * noise
                
                # 验证时不 drop 条件
                net_input = torch.cat([x_t, h_expanded], dim=1)
                
                # 预测 x0
                predicted_x0 = model(net_input, t)
                
                # 计算验证损失 (Reconstruction MSE)
                loss = criterion(predicted_x0, clean_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"[Epoch {epoch}] Val Loss: {avg_val_loss:.6f}")

        # --- 保存策略 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            # 文件名加 _x0 标记
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_with_n_step{IS2B.n_steps}_x0.pth"))
        else:
            epochs_since_improvement += 1

        if epochs_since_improvement >= patience:
            print("Early stopping.")
            break

    # 绘图代码略... (loss curve)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100
    batch_size = 64
    epochs = 500
    
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 4,  # 2(y) + 2(h)
        'out_channels': 2  # 输出 2 通道 (这次代表 x0，不是 noise)
    }
    
    model = build_network(net_cfg, n_steps).to(device)
    IS2B = IS2B(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    train_IS2B_x0_pred(
        model, IS2B,
        train_loader, val_loader,
        epochs=epochs,
        device=device,
        save_dir='IS2B/IS2B_rayleigh/results',
        p_uncond=0.15
    )