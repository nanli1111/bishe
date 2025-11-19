import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 自定义模块
from model.unet import CDDMUNet, build_cddm_network  # CDDM网络
from cddm_core import CDDM  # CDDM核心类
from dataset.dataset import get_QPSKdataloader  # 需要支持信道数据的dataloader

def train_cddm(model, cddm, dataloader, epochs=50, lr=1e-4, device='cuda', save_dir='./cddm_results'):
    """
    CDDM训练函数
    参数:
        model: CDDM噪声预测网络 (CDDMUNet或CDDMConvNet)
        cddm: CDDM核心类实例
        dataloader: 数据加载器，应返回 (x, h_c, sigma) 或 (x, h_c)
        epochs: 训练轮数
        lr: 学习率
        device: 训练设备
        save_dir: 结果保存目录
    """
    os.makedirs(save_dir, exist_ok=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    loss_history = []

    model.train()

    for epoch in range(1, epochs + 1):
        epoch_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{epochs}")
        
        for batch_data in pbar:
            
            x, h_c, _ = batch_data
            x = x.to(device)
            h_c = h_c.to(device)
            sigma = torch.tensor(0.1).to(device)  # 默认sigma
            
            # ==================== CDDM训练步骤 ====================
            # 随机选择时间步
            t = torch.randint(0, cddm.n_steps, (x.size(0),), device=device).long()

            # 生成噪声 eps (从标准正态分布中采样)
            eps = torch.randn_like(x)  # 生成标准正态分布的噪声 [batch_size, 2, 48]
            
            # 1. 获取带噪信号 (前向扩散)
            x_t = cddm.sample_forward(x, t, h_c, sigma, eps)
            
            # 2. 使用网络预测噪声
            eps_pred = model(x_t, t, h_c)  # 模型输出预测的噪声
            
            # 3. 计算损失: 预测噪声与真实噪声的 MSE
            loss = torch.mean((eps - eps_pred) ** 2)
            
            # 4. 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        scheduler.step()
        avg_loss = epoch_loss / len(dataloader)
        loss_history.append(avg_loss)
        print(f"[Epoch {epoch}] Avg Loss: {avg_loss:.6f}")

        # ==================== 保存和可视化 ====================
        if epoch % 5 == 0:
            # 保存模型
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(save_dir, f"cddm_model_epoch{epoch}.pth"))

            # ===== 可视化去噪效果 =====
            model.eval()
            with torch.no_grad():
                # 获取测试批次
                test_batch = next(iter(dataloader))
                
                # 处理测试数据
                if len(test_batch) == 3:
                    x_test, h_c_test, sigma_test = test_batch
                else:
                    x_test, h_c_test = test_batch
                    sigma_test = torch.tensor(0.1)
                
                x_test = x_test[:8].to(device)  # 前8个样本
                h_c_test = h_c_test[:8].to(device)
                sigma_test = sigma_test.to(device) if isinstance(sigma_test, torch.Tensor) else torch.tensor(sigma_test).to(device)

                # 随机时间步
                t_test = torch.randint(0, cddm.n_steps, (x_test.size(0),), device=device)

                # 获取带噪信号 (前向扩散)
                x_noisy, _ = cddm.sample_forward(x_test, t_test, h_c_test, sigma_test)
                
                # 进行CDDM去噪 (逆向采样)
                x_denoised = cddm.sample_backward(x_noisy, model, h_c_test, sigma_test)

                # ===== 可视化信号对比 =====
                # 创建一个 2x3 的子图布局
                fig, axs = plt.subplots(2, 3, figsize=(18, 10))

                # 绘制原始信号的I路
                axs[0, 0].plot(x_test[0][0].cpu().numpy(), label='Original I', color='blue', linewidth=2)
                axs[0, 0].set_title(f"Original Signal - I Channel - Epoch {epoch}")
                axs[0, 0].set_xlabel('Sample Index')
                axs[0, 0].set_ylabel('Amplitude')
                axs[0, 0].legend()
                axs[0, 0].grid(True, alpha=0.3)

                # 绘制原始信号的Q路
                axs[0, 1].plot(x_test[0][1].cpu().numpy(), label='Original Q', color='purple', linewidth=2)
                axs[0, 1].set_title(f"Original Signal - Q Channel - Epoch {epoch}")
                axs[0, 1].set_xlabel('Sample Index')
                axs[0, 1].set_ylabel('Amplitude')
                axs[0, 1].legend()
                axs[0, 1].grid(True, alpha=0.3)

                # 绘制带噪信号的I路
                axs[0, 2].plot(x_noisy[0][0].cpu().numpy(), label='Noisy I', color='red', linewidth=2)
                axs[0, 2].set_title(f"Noisy Signal - I Channel - Epoch {epoch}")
                axs[0, 2].set_xlabel('Sample Index')
                axs[0, 2].set_ylabel('Amplitude')
                axs[0, 2].legend()
                axs[0, 2].grid(True, alpha=0.3)

                # 绘制带噪信号的Q路
                axs[1, 0].plot(x_noisy[0][1].cpu().numpy(), label='Noisy Q', color='orange', linewidth=2)
                axs[1, 0].set_title(f"Noisy Signal - Q Channel - Epoch {epoch}")
                axs[1, 0].set_xlabel('Sample Index')
                axs[1, 0].set_ylabel('Amplitude')
                axs[1, 0].legend()
                axs[1, 0].grid(True, alpha=0.3)

                # 绘制去噪信号的I路
                axs[1, 1].plot(x_denoised[0][0].cpu().numpy(), label='Denoised I', color='green', linewidth=2)
                axs[1, 1].set_title(f"CDDM Denoised - I Channel - Epoch {epoch}")
                axs[1, 1].set_xlabel('Sample Index')
                axs[1, 1].set_ylabel('Amplitude')
                axs[1, 1].legend()
                axs[1, 1].grid(True, alpha=0.3)

                # 绘制去噪信号的Q路
                axs[1, 2].plot(x_denoised[0][1].cpu().numpy(), label='Denoised Q', color='brown', linewidth=2)
                axs[1, 2].set_title(f"CDDM Denoised - Q Channel - Epoch {epoch}")
                axs[1, 2].set_xlabel('Sample Index')
                axs[1, 2].set_ylabel('Amplitude')
                axs[1, 2].legend()
                axs[1, 2].grid(True, alpha=0.3)

                # 调整布局
                plt.tight_layout()

                # 保存图像
                plt.savefig(os.path.join(save_dir, f"cddm_signal_comparison_epoch{epoch}.png"), dpi=300, bbox_inches='tight')
                plt.close()

                print(f"Epoch {epoch}: CDDM可视化已保存 - cddm_signal_comparison_epoch{epoch}.png")

            model.train()

    # ===== 绘制训练曲线 =====
    plt.figure(figsize=(10, 6))
    plt.plot(loss_history, label='CDDM Training Loss', color='blue', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('CDDM Channel Denoising Training Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, 'cddm_training_loss_curve.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 保存训练记录
    training_log = {
        'epochs': epochs,
        'final_loss': loss_history[-1],
        'loss_history': loss_history,
        'learning_rate': lr
    }
    torch.save(training_log, os.path.join(save_dir, 'cddm_training_log.pth'))
    
    print("✅ CDDM Training finished and all results saved!")


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100  # CDDM通常需要更多步数
    batch_size = 64
    epochs = 100
    lr = 1e-4
    # ===== 获取数据加载器 =====    
    dataloader = get_QPSKdataloader(start=0, end=400000, batch_size=batch_size, shuffle=True)   #返回x,h_c


    # ===== 构建CDDM模型 =====
    cddm_net_cfg = {
        'type': 'CDDMUNet', 
        'channels': [10, 20, 40, 80], 
        'pe_dim': 128,
        'channel_embed_dim': 32
    }
    model = build_cddm_network(cddm_net_cfg, n_steps).to(device)
    
    # ===== 初始化CDDM =====
    cddm = CDDM(
        device=device,
        n_steps=n_steps,
        channel_type='rayleigh',  # 或 'awgn'
        min_beta=1e-4,
        max_beta=0.02
    )

    # ===== 获取数据 =====

    # ===== 训练CDDM =====
    train_cddm(
        model=model,
        cddm=cddm,
        dataloader=dataloader,
        epochs=epochs,
        lr=lr,
        device=device,
        save_dir='cddm_results'
    )