import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 自定义模块
from model.unet import UNet, build_network
from ddrm_core import DDRM
from dataset import get_QPSKdataloader  

def train_ddrm(model, ddrm, dataloader, epochs=50, lr=1e-4, device='cuda', save_dir='./results'):
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []

    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        for  in pbar:

            x = x.float().to(device)

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

        if epoch % 20 == 0:
            # 保存模型
            torch.save(model.state_dict(), os.path.join(save_dir, f"model_epoch{epoch}_100.pth"))

            # ===== 可视化去噪效果 =====
            model.eval()
            with torch.no_grad():
                input_sample = next(iter(dataloader))
                input_sample = input_sample[:8].to(device)  # 前8个信号
                t_sample = torch.randint(0, ddrm.n_steps, (input_sample.size(0),), device=device)

                # 获取带噪信号
                input_noisy = ddrm.q_sample(input_sample, t_sample)
                # 进行恢复
                input_denoised = ddrm.denoise(input_noisy)

                # 创建一个 2x3 的子图布局
                fig, axs = plt.subplots(2, 3, figsize=(18, 10))

                # 绘制原始信号的幅度图 (I 路)
                axs[0, 0].plot(input_sample[0][0].cpu().numpy(), label='Original I Channel', color='blue')
                axs[0, 0].set_title(f"Original Signal - I Channel - Epoch {epoch}")
                axs[0, 0].set_xlabel('Sample Index')
                axs[0, 0].set_ylabel('Magnitude')
                axs[0, 0].legend()

                # 绘制原始信号的幅度图 (Q 路)
                axs[0, 1].plot(input_sample[0][1].cpu().numpy(), label='Original Q Channel', color='purple')
                axs[0, 1].set_title(f"Original Signal - Q Channel - Epoch {epoch}")
                axs[0, 1].set_xlabel('Sample Index')
                axs[0, 1].set_ylabel('Magnitude')
                axs[0, 1].legend()

                # 绘制带噪信号的幅度图 (I 路)
                axs[0, 2].plot(input_noisy[0][0].cpu().numpy(), label='Noisy I Channel', color='red')
                axs[0, 2].set_title(f"Noisy Signal - I Channel - Epoch {epoch}")
                axs[0, 2].set_xlabel('Sample Index')
                axs[0, 2].set_ylabel('Magnitude')
                axs[0, 2].legend()

                # 绘制带噪信号的幅度图 (Q 路)
                axs[1, 0].plot(input_noisy[0][1].cpu().numpy(), label='Noisy Q Channel', color='blue')
                axs[1, 0].set_title(f"Noisy Signal - Q Channel - Epoch {epoch}")
                axs[1, 0].set_xlabel('Sample Index')
                axs[1, 0].set_ylabel('Magnitude')
                axs[1, 0].legend()

                # 绘制去噪信号的幅度图 (I 路)
                axs[1, 1].plot(input_denoised[0][0].cpu().numpy(), label='Denoised I Channel', color='green')
                axs[1, 1].set_title(f"Denoised Signal - I Channel - Epoch {epoch}")
                axs[1, 1].set_xlabel('Sample Index')
                axs[1, 1].set_ylabel('Magnitude')
                axs[1, 1].legend()

                # 绘制去噪信号的幅度图 (Q 路)
                axs[1, 2].plot(input_denoised[0][1].cpu().numpy(), label='Denoised Q Channel', color='orange')
                axs[1, 2].set_title(f"Denoised Signal - Q Channel - Epoch {epoch}")
                axs[1, 2].set_xlabel('Sample Index')
                axs[1, 2].set_ylabel('Magnitude')
                axs[1, 2].legend()

                # 调整布局，防止图像重叠
                plt.tight_layout()

                # 保存图像
                plt.savefig(os.path.join(save_dir, f"signal_comparison_epoch{epoch}.png"))
                plt.close()  # 关闭当前图像

                print(f"Epoch {epoch}: 图片已保存 - signal_comparison_epoch{epoch}.png")

        model.train()

    # ===== 绘制训练曲线 =====
    plt.figure()
    plt.plot(loss_history, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('DDRM QPSK Training Curve')
    plt.legend()
    plt.savefig(os.path.join(save_dir, 'training_loss_curve.png'))
    plt.close()
    print("✅ Training finished and results saved!")

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100  # 扩散步数，可调整
    batch_size = 64
    epochs = 50
    lr = 1e-4

    # ===== 构建模型 =====
    net_cfg = {'type': 'UNet', 'channels': [10, 20, 40, 80], 'pe_dim': 128}
    model = build_network(net_cfg, n_steps).to(device)
    ddrm = DDRM(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # ===== 数据 =====
    dataloader = get_QPSKdataloader(start = 0, end = 100000, batch_size=batch_size)  # 请确保返回的数据为 (batch_size, 24 48)

    # ===== 训练 =====
    train_ddrm(model, ddrm, dataloader, epochs=epochs, lr=lr, device=device, save_dir='ddrm_nak/results')
