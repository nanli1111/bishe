import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt

# ==========================================
# 导入自定义模块 (假设文件名为 model.py 和 ddpm_core.py)
# ==========================================
# 请根据实际文件名修改 import 路径
from model.net import SETimeResNet1D  # 刚才定义的双通道 ResNet
from ddpm_core import DDPM     # 刚才定义的 X-Prediction DDPM
from dataset.dataset import get_train_QPSKdataloader 

def train_ddpm(model, ddpm, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda', save_dir='./results', patience=10):
    os.makedirs(save_dir, exist_ok=True)
    
    # 优化器
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    # 学习率调度器
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    loss_history = []
    val_loss_history = []
    best_val_loss = float('inf')
    epochs_since_improvement = 0

    print(f"Start training on {device}...")

    for epoch in range(1, epochs + 1):
        # ==================== 训练阶段 ====================
        model.train()
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch in pbar:
            # 适配不同的 DataLoader 返回格式
            if isinstance(batch, (list, tuple)):
                x = batch[0] # 取出信号数据 [B, 2, L]
            else:
                x = batch

            x = x.to(device).float()
            
            # 1. 随机采样时间步 t [0, n_steps-1]
            t = torch.randint(0, ddpm.n_steps, (x.size(0),), device=device).long()
            
            # 2. 计算 DDPM 损失 (p_losses 内部会自动加噪并计算 MSE)
            loss = ddpm.p_losses(x, t)
            
            # 3. 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # (可选) 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        scheduler.step()
        avg_train_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_train_loss)
        
        # ==================== 验证阶段 ====================
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                if isinstance(batch, (list, tuple)):
                    x = batch[0]
                else:
                    x = batch
                    
                x = x.to(device).float()
                t = torch.randint(0, ddpm.n_steps, (x.size(0),), device=device).long()
                
                # 计算验证损失
                loss = ddpm.p_losses(x, t)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)

        # 打印日志
        print(f"[Epoch {epoch}] Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # ==================== 早停 (Early Stopping) & 保存 ====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_since_improvement = 0
            save_path = os.path.join(save_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"⭐ Validation loss improved. Model saved to {save_path}")
        else:
            epochs_since_improvement += 1
            print(f"Info: No improvement for {epochs_since_improvement}/{patience} epochs.")

        if epochs_since_improvement >= patience:
            print(f"⚠️ Early stopping triggered at epoch {epoch}.")
            break

    # ==================== 绘图 ====================
    plt.figure(figsize=(10, 5))
    plt.plot(loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss (x0 prediction)')
    plt.title('DDPM Signal Recovery Training')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, 'loss_curve.png'))
    plt.close()
    
    print("✅ Training finished.")

if __name__ == "__main__":
    # 1. 配置参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 70        # 扩散步数 (对于简单信道 50-100 够了，复杂信道可能要 1000)
    batch_size = 64      # 批次大小
    epochs = 1000        # 最大轮数
    lr = 2e-4            # 学习率
    
    # 2. 准备数据
    # 请确保 get_train_QPSKdataloader 能够正常导入并工作
    # 假设它返回的数据形状是 [Batch, 2, Length] (I/Q 两路)
    print("Loading data...")
    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=100000, batch_size=batch_size, val_split=0.2
    )

    # 3. 构建模型 (SETimeResNet1D)
    # 输入输出通道设为 2 (对应实部和虚部)
    print("Building model...")
    model = SETimeResNet1D(
        in_channels=2, 
        out_channels=2, 
        hidden_dim=128, 
        num_blocks=8,      # 根据显存调整，显存小改 4
        time_emb_dim=128
    ).to(device)

    # 4. 构建扩散过程 (DDPM X-Prediction)
    ddpm = DDPM(
        model, 
        n_steps=n_steps, 
        min_beta=1e-4, 
        max_beta=0.02, 
        device=device
    )

    # 5. 开始训练
    train_ddpm(
        model=model, 
        ddpm=ddpm, 
        train_loader=train_loader, 
        val_loader=val_loader, 
        epochs=epochs, 
        lr=lr, 
        device=device, 
        save_dir='./results'
    )