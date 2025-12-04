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
# 保持原有的数据加载和加噪函数引入
from dataset.dataset import get_train_QPSKdataloader
from test_fig_x_pre import add_awgn_noise_torch

# ==========================================
# 1. 定义 SimpleResNet1D 模型 (替代 U-Net)
# ==========================================
class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SimpleResNet1D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=64, num_blocks=6):
        super().__init__()
        
        # 头部：特征提取
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 中间：残差堆叠
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 尾部：映射回 IQ
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x, t=None):
        # t 参数是为了兼容之前的接口，这里忽略
        x = x.float()
        out = self.entry(x)
        for block in self.blocks:
            out = block(out)
        out = self.exit(out)
        return out

# ==========================================
# 2. 训练主函数
# ==========================================
def train_supervised_resnet(model, train_loader, val_loader, 
                            epochs=100, lr=1e-3, device='cuda', 
                            save_dir='./results_resnet_pure', 
                            sps=16, patience=5): 
    
    # 初始化
    os.makedirs(save_dir, exist_ok=True)
    criterion = nn.MSELoss() 
    # ResNet 通常可以使用稍大的 LR (1e-3)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # 学习率调度: 不下降则减半
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # 固定 SNR (10dB Symbol SNR)
    target_symbol_snr = 10.0
    fixed_sample_snr = target_symbol_snr - 10 * math.log10(sps)
    
    print(f"🚀 开始训练 SimpleResNet1D (Fixed SNR={target_symbol_snr}dB)...")
    print(f"设备: {device}, Epochs: {epochs}, Patience: {patience}")
    print(f"Save Dir: {save_dir}")

    for epoch in range(1, epochs + 1):
        # --- 训练 ---
        model.train()
        epoch_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)
        
        for clean_x, faded_y, h_est in pbar:
            clean_x = clean_x.to(device).float()
            faded_y = faded_y.to(device).float()
            h_est = h_est.to(device).float()
            
            seq_len = clean_x.shape[2]
            
            # h 扩展
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # 加噪
            noisy_y = add_awgn_noise_torch(faded_y, fixed_sample_snr)

            # 构造输入 (B, 4, L)
            t_dummy = None # ResNet 不需要 t
            net_input = torch.cat([noisy_y, h_expanded], dim=1)

            # 优化步
            optimizer.zero_grad()
            predicted_x = model(net_input, t_dummy)
            loss = criterion(predicted_x, clean_x)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'MSE': f"{loss.item():.5f}"})

        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)

        # --- 验证 ---
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for clean_x, faded_y, h_est in val_loader:
                clean_x = clean_x.to(device).float()
                faded_y = faded_y.to(device).float()
                h_est = h_est.to(device).float()
                
                seq_len = clean_x.shape[2]

                if h_est.dim() == 2:
                    h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
                else:
                    h_expanded = h_est

                # 验证 SNR 保持一致
                noisy_y = add_awgn_noise_torch(faded_y, fixed_sample_snr)
                net_input = torch.cat([noisy_y, h_expanded], dim=1)
                
                predicted_x = model(net_input, None)
                loss = criterion(predicted_x, clean_x)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        # 更新 LR
        scheduler.step(avg_val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f"[Epoch {epoch}] Train MSE: {avg_train_loss:.6f} | Val MSE: {avg_val_loss:.6f} | LR: {current_lr:.2e}")

        # --- 保存与早停 ---
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(save_dir, "best_model_resnet.pth"))
            print("--> Best Model Saved.")
        else:
            epochs_no_improve += 1
            print(f"--> No improvement: {epochs_no_improve}/{patience}")

        if epochs_no_improve >= patience:
            print(f"🛑 Early stopping triggered.")
            break

    # 画图
    plt.figure(figsize=(8, 6))
    plt.plot(loss_history, label='Train MSE')
    plt.plot(val_loss_history, label='Val MSE')
    plt.title(f'MSE Curve (SimpleResNet1D, SNR={target_symbol_snr}dB)')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve_resnet.png'))
    print("训练结束。")

# ==========================================
# 3. 入口
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 64
    epochs = 100
    sps = 16 
    
    # 1. 加载数据
    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.1
    )

    # 2. 构建 SimpleResNet1D
    # hidden_dim=64, num_blocks=6 是一个很轻量但足够强的配置
    print(f"Building SimpleResNet1D on {device}...")
    model = SimpleResNet1D(in_channels=4, out_channels=2, hidden_dim=64, num_blocks=6).to(device)
    
    # 3. 开始训练
    train_supervised_resnet(
        model, 
        train_loader, 
        val_loader,
        epochs=epochs,
        device=device,
        save_dir='CNN/RESNET/results', # 保存到新文件夹区分
        sps=sps,
        patience=5
    )