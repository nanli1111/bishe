import os
import torch
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
from torchvision.utils import save_image

# 自定义模块
from model.unet import CDDMUNet, build_cddm_network  # 假设您有CDDM网络
from cddm_core import CDDM  # 假设您有CDDM核心类
from dataset import get_QPSKdataloader_with_channel  # 需要支持信道数据的dataloader

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
            # ==================== 数据接口 ====================
            # 接口1: 如果dataloader返回 (x, h_c, sigma)
            if len(batch_data) == 3:
                x, h_c, sigma = batch_data
                x = x.to(device)
                h_c = h_c.to(device)
                sigma = sigma.to(device) if isinstance(sigma, torch.Tensor) else torch.tensor(sigma).to(device)
            
            # 接口2: 如果dataloader返回 (x, h_c)，使用固定sigma
            elif len(batch_data) == 2:
                x, h_c = batch_data
                x = x.to(device)
                h_c = h_c.to(device)
                sigma = torch.tensor(0.1).to(device)  # 默认sigma
            
            # 接口3: 如果dataloader返回复杂结构，使用自定义处理函数
            else:
                # 这里可以调用自定义的数据处理函数
                x, h_c, sigma = process_custom_batch_data(batch_data, device)
            
            # ==================== CDDM训练步骤 ====================
            # 随机选择时间步
            t = torch.randint(0, cddm.n_steps, (x.size(0),), device=device).long()
            
            # 计算CDDM损失
            loss = cddm.train_step(x, model, h_c, sigma, optimizer)
            
            epoch_loss += loss
            pbar.set_postfix({'loss': loss})

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

def process_custom_batch_data(batch_data, device):
    """
    自定义批次数据处理函数
    根据您的实际数据结构进行调整
    """
    # 示例：如果您的数据是复杂结构
    # 假设 batch_data 包含 x, h_c, 和其他信息
    if isinstance(batch_data, dict):
        x = batch_data['signal'].to(device)
        h_c = batch_data['channel'].to(device)
        sigma = batch_data.get('sigma', torch.tensor(0.1)).to(device)
    elif isinstance(batch_data, (list, tuple)):
        # 根据您的数据结构调整索引
        x = batch_data[0].to(device)
        h_c = batch_data[1].to(device)
        sigma = torch.tensor(0.1).to(device)  # 默认值
    else:
        raise ValueError("Unsupported batch data format")
    
    return x, h_c, sigma

def get_cddm_dataloader(start=0, end=100000, batch_size=64, include_channel=True):
    """
    获取CDDM数据加载器
    需要返回 (信号, 信道估计, 噪声水平) 或 (信号, 信道估计)
    """
    # 这里需要您根据实际数据源实现
    # 示例实现：
    if include_channel:
        # 返回 (x, h_c, sigma)
        dataloader = get_QPSKdataloader_with_channel(start, end, batch_size)
    else:
        # 返回 (x, h_c)，使用固定sigma
        dataloader = get_QPSKdataloader(start, end, batch_size)
    
    return dataloader

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 1000  # CDDM通常需要更多步数
    batch_size = 32
    epochs = 100
    lr = 1e-4

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
    # 使用支持信道数据的数据加载器
    dataloader = get_cddm_dataloader(
        start=0, 
        end=100000, 
        batch_size=batch_size, 
        include_channel=True
    )

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