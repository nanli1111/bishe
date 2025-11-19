import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 自定义模块
from model.unet import UNet, build_network
from ddrm_core import DDRM
from dataset.dataset import get_train_QPSKdataloader 

def train_ddrm(model, ddrm, train_loader, val_loader, epochs=50, lr=1e-4, device='cuda', save_dir='./results', patience=10):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []
    val_loss_history = []

    model.train()

    best_val_loss = float('inf')  # 初始验证损失为无穷大
    epochs_since_improvement = 0  # 记录验证损失没有改善的轮次

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")
        
        # 注意：train_loader 返回的是 (x, ..., ...)，只用第一个 x
        for x, _, _ in pbar:
            x = x.to(device)
            # 随机选择时间步
            t = torch.randint(0, ddrm.n_steps, (x.size(0),), device=device).long()
            # 计算 DDRM 损失
            loss = ddrm.p_losses(x, t)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(train_loader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch}] Avg Train Loss: {avg_loss:.6f}")

        # 进行验证
        model.eval()  # 设置模型为评估模式
        val_loss = 0
        with torch.no_grad():  # 禁用梯度计算
            # 这里同样需要解包 (x, ..., ...)
            for x, _, _ in val_loader:
                x = x.to(device)
                t = torch.randint(0, ddrm.n_steps, (x.size(0),), device=device).long()
                # 计算验证集损失
                loss = ddrm.p_losses(x, t)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_loss_history.append(avg_val_loss)
        print(f"[Epoch {epoch}] Avg Validation Loss: {avg_val_loss:.6f}")

        # ==================== 过拟合报警 ====================
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss  # 更新最优验证损失
            epochs_since_improvement = 0  # 重置计数器
            print(f"Validation loss improved, saving model...")
            # 保存模型
            torch.save(model.state_dict(), os.path.join(save_dir, f"best_model_epoch_with_n_steps{ddrm.n_steps}.pth"))
        else:
            epochs_since_improvement += 1
            print(f"Validation loss did not improve at epoch {epoch}. {epochs_since_improvement}/{patience} epochs without improvement.")

        if epochs_since_improvement >= patience:
            print(f"⚠️ Overfitting detected! Validation loss has not improved for {patience} epochs.")
            print(f"Stopping training early at epoch {epoch}.")
            break

        model.train()  # 恢复训练模式
        

    # ===== 绘制训练曲线 =====
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDRM QPSK Training Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'))
    plt.close()

    plt.figure()
    plt.plot(val_loss_history, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDRM QPSK Validation Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'validation_loss_curve.png'))
    plt.close()

    print("✅ Training finished and results saved!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 60  # 扩散步数，可调整
    batch_size = 64  # 批次大小
    epochs = 1000 # 训练轮数（给一个足够大的值即可）   
    lr = 1e-4

    # ===== 构建模型 =====
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)
    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # ===== 数据（保持你原来的数据加载器不变） =====
    train_loader, val_loader = get_train_QPSKdataloader(
        start=0, end=400000, batch_size=batch_size, val_split=0.2
    )

    # ===== 训练 =====
    train_ddrm(
        model, ddrm,
        train_loader, val_loader,
        epochs=epochs, lr=lr,
        device=device,
        save_dir='ddrm/ddrm_nak/results'
    )
