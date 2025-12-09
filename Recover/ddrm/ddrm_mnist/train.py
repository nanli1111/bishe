# main_train_ddrm.py
import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 自定义模块
from model.unet import UNet, build_network
from ddrm_core import DDRM
from dataset.dataset import get_dataloader

def train_ddrm(model, ddrm, dataloader, epochs=50, lr=1e-4, device='cuda', save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []

    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for x, _ in pbar:
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
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

       
        if epoch % 5 ==0:
            # 保存模型
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch}.pth"))
            # ===== 可视化去噪效果 =====
            model.eval()
            with torch.no_grad():
                x_sample, _ = next(iter(dataloader))
                x_sample = x_sample[:8].to(device)  # 前8张
                t_sample = torch.randint(0, ddrm.n_steps, (x_sample.size(0),), device=device)
                x_noisy = ddrm.q_sample(x_sample, t_sample)
                x_denoised = ddrm.denoise(x_noisy)

                # 保存图片
                save_image((x_noisy + 1) / 2, os.path.join(save_dir, f"noisy_epoch{epoch}.png"))
                save_image((x_denoised + 1) / 2, os.path.join(save_dir, f"denoised_epoch{epoch}.png"))

        model.train()

    # ===== 绘制训练曲线 =====
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDRM MNIST Training Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'))
    plt.close()
    print("✅ Training finished and results saved!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 80  # 扩散步数，可调整
    batch_size = 64
    epochs = 50
    lr = 1e-4

    # ===== 构建模型 =====
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)
    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # ===== 数据 =====
    dataloader = get_dataloader(batch_size=batch_size)

    # ===== 训练 =====
    train_ddrm(model, ddrm, dataloader, epochs=epochs, lr=lr, device=device, save_dir='ddrm_mnist/results')
