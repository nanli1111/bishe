import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 引入项目模块
from model.unet import build_network
from ddrm_x_pre import DDRM
from dataset.dataset import QPSKDataset

# 中文字体设置
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# 加噪函数
def add_awgn_noise_np(clean_data, EbN0_db):
    noisy_data = np.zeros(clean_data.shape)
    # 计算信号功率 (batch wise)
    signal_power = np.mean((np.abs(clean_data[:,0,:]) ** 2) + (np.abs(clean_data[:,1,:]) ** 2))
    EbN0_linear = 10**(EbN0_db/10)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0/2)
    
    noise_I = noise_std * np.random.randn(*clean_data[:,0,:].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:,1,:].shape)
    
    noisy_data[:,0,:] = clean_data[:,0,:] + noise_I
    noisy_data[:,1,:] = clean_data[:,1,:] + noise_Q
    return noisy_data


def visualize_ddrm_restoration(model, ddrm, tx_clean, rx_faded, h_np, snr_list, sps=16, 
                               guidance_scale=1.0, # CFG 强度控制
                               device='cuda', save_dir='ddrm/ddrm_rayleigh/vis_results'):
    """
    可视化 DDRM 在 rayleigh 信道下的恢复效果（适配 x0-prediction + CFG）
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # 取 batch 中的第一个样本进行画图
    idx_to_plot = 0
    x_sample_clean = tx_clean[idx_to_plot:idx_to_plot+1] # [1, 2, L] 原始
    y_sample_faded = rx_faded[idx_to_plot:idx_to_plot+1] # [1, 2, L] 衰落后
    h_sample = h_np[idx_to_plot:idx_to_plot+1]           # [1, 2]

    # 转为 Tensor
    h_tensor = torch.from_numpy(h_sample).float().to(device)
    
    # 构造标题用的信道字符串
    h_val_str = f"{h_sample[0,0]:.2f} + {h_sample[0,1]:.2f}j"
    h_abs = np.sqrt(h_sample[0,0]**2 + h_sample[0,1]**2)

    for snr_db in snr_list:
        # 1. 准备噪声
        snr_db_sample = snr_db - 10 * math.log10(sps)
        
        # 加噪 (模拟接收到的 y)
        y_sample_noisy = add_awgn_noise_np(y_sample_faded, snr_db_sample)
        y_tensor_noisy = torch.from_numpy(y_sample_noisy).float().to(device)

        # 注意：在使用 sample_x0_prediction 时，不需要手动计算 sigma_y 传入
        # 因为网络直接预测去噪后的波形，不需要物理方程中的 sigma 参数

        # 2. DDRM 恢复 (使用 x0-prediction 采样器)
        with torch.no_grad():
            x_rec = ddrm.sample_x0_prediction(
                y=y_tensor_noisy,
                h=h_tensor,
                eta=0.0,               # 确定性采样
                guidance_scale=guidance_scale
            )
        
        x_rec_np = x_rec.cpu().numpy()
        
        # 3. 绘图
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # --- I 路 (第一行) ---
        # Col 1: 参考
        axs[0, 0].plot(x_sample_clean[0, 0, :], label='原始信号(干净)', color='gray', linewidth=2, alpha=0.5)
        axs[0, 0].plot(y_sample_faded[0, 0, :], label='衰落信号(无噪)', color='purple', linestyle='--', linewidth=1.5)
        axs[0, 0].set_title(f"参考：信道影响对比 (|h|={h_abs:.2f})", fontsize=12)
        axs[0, 0].set_ylabel('幅度 (I路)', fontsize=10)
        axs[0, 0].legend(loc='upper right', fontsize=9)
        axs[0, 0].grid(True, linestyle=':', alpha=0.6)
        
        # Col 2: 输入
        axs[0, 1].plot(y_sample_noisy[0, 0, :], label='含噪输入', color='red', alpha=0.7)
        axs[0, 1].set_title(f"输入：衰落 + 加噪", fontsize=12)
        axs[0, 1].legend(loc='upper right', fontsize=9)
        axs[0, 1].grid(True, linestyle=':', alpha=0.6)

        # Col 3: 输出
        axs[0, 2].plot(x_sample_clean[0, 0, :], label='原始真值', color='gray', alpha=0.4, linestyle='-')
        axs[0, 2].plot(x_rec_np[0, 0, :], label='DDRM 恢复', color='green', linewidth=1.5)
        axs[0, 2].set_title(f"输出：DDRM 恢复结果 (CFG w={guidance_scale})", fontsize=12)
        axs[0, 2].legend(loc='upper right', fontsize=9)
        axs[0, 2].grid(True, linestyle=':', alpha=0.6)

        # --- Q 路 (第二行) ---
        # Col 1: 参考
        axs[1, 0].plot(x_sample_clean[0, 1, :], label='原始信号(干净)', color='gray', linewidth=2, alpha=0.5)
        axs[1, 0].plot(y_sample_faded[0, 1, :], label='衰落信号(无噪)', color='purple', linestyle='--', linewidth=1.5)
        axs[1, 0].set_title(f"参考：信道影响对比 (Q路)", fontsize=12)
        axs[1, 0].set_ylabel('幅度 (Q路)', fontsize=10)
        axs[1, 0].set_xlabel('采样点索引', fontsize=10)
        axs[1, 0].legend(loc='upper right', fontsize=9)
        axs[1, 0].grid(True, linestyle=':', alpha=0.6)

        # Col 2: 输入
        axs[1, 1].plot(y_sample_noisy[0, 1, :], label='含噪输入', color='blue', alpha=0.7)
        axs[1, 1].set_title(f"输入：衰落 + 加噪", fontsize=12)
        axs[1, 1].set_xlabel('采样点索引', fontsize=10)
        axs[1, 1].legend(loc='upper right', fontsize=9)
        axs[1, 1].grid(True, linestyle=':', alpha=0.6)

        # Col 3: 输出
        axs[1, 2].plot(x_sample_clean[0, 1, :], label='原始真值', color='gray', alpha=0.4, linestyle='-')
        axs[1, 2].plot(x_rec_np[0, 1, :], label='DDRM 恢复', color='orange', linewidth=1.5)
        axs[1, 2].set_title(f"输出：DDRM 恢复结果 (CFG w={guidance_scale})", fontsize=12)
        axs[1, 2].set_xlabel('采样点索引', fontsize=10)
        axs[1, 2].legend(loc='upper right', fontsize=9)
        axs[1, 2].grid(True, linestyle=':', alpha=0.6)

        plt.suptitle(f"DDRM 信号恢复可视 (x0-Pred + CFG) @ 符号SNR {snr_db} dB\n信道 h = {h_val_str}, CFG Scale = {guidance_scale}", fontsize=16)
        plt.tight_layout()
        
        # 文件名包含 guidance scale
        save_name = os.path.join(save_dir, f"vis_snr{snr_db}_cfg{guidance_scale}.png")
        plt.savefig(save_name)
        plt.close()
        print(f"可视化结果已保存至: {save_name}")


if __name__ == "__main__":
    # ----- 全局配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100         # 需与训练时的 n_steps 一致
    sps = 16              
    
    # 路径配置
    # 注意：这里加载的是 _x0.pth 后缀的模型
    ckpt_path = fr'ddrm/ddrm_rayleigh/results/best_model_epoch_with_n_step{n_steps}_x0.pth'
    vis_save_dir = 'ddrm/ddrm_rayleigh/vis_results'

    # ----- 1. 加载模型 (与训练一致) -----
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 6,               # 2(x) + 2(y) + 2(h)
        'out_channels': 2
    }
    model = build_network(net_cfg, n_steps).to(device)

    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}，请检查文件名或路径")

    ddrm = DDRM(model, n_steps=n_steps,
                min_beta=1e-4, max_beta=0.02, device=device)

    # ----- 2. 加载数据 -----
    print("正在加载数据...")
    test_data = QPSKDataset(400000, 400100) # 取 100 个样本
    
    tx_clean = test_data.x   # [N, 2, L] 原始干净信号
    rx_faded = test_data.y   # [N, 2, L] 衰落信号
    h_np = test_data.z       # [N, 2]    信道系数

    # ----- 3. 运行可视化 -----
    snr_list = [5, 10, 20] 
    
    # x0-prediction 模式下，w 可以尝试不同范围
    # 0 < w < 1: 平滑模式
    # w = 1: 标准条件模式
    # w > 1: 强化条件模式
    cfg_scales = [0, 0.3, 0.5, 1.0, 2.0]

    for scale in cfg_scales:
        print(f"\n--- Running visualization with CFG Scale = {scale} ---")
        visualize_ddrm_restoration(
            model=model,
            ddrm=ddrm,
            tx_clean=tx_clean,
            rx_faded=rx_faded,
            h_np=h_np,
            snr_list=snr_list,
            sps=sps,
            guidance_scale=scale,
            device=device,
            save_dir=vis_save_dir
        )
    
    print("所有绘图任务已完成。")