import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === 引入项目模块 ===
# 1. 导入新的 ResNet 模型
from model.resnet_pro import DilatedTimeResNet1D
# 2. 导入 IS2B 包装器
from IS2B_x_pre import IS2B
# 3. 数据集与工具
from dataset.dataset import QPSKDataset

# 中文字体设置
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 辅助工具函数
# ==========================================

def add_awgn_noise_np(clean_data, EbN0_db):
    noisy_data = np.zeros(clean_data.shape)
    # 计算信号功率
    signal_power = np.mean((np.abs(clean_data[:,0,:]) ** 2) + (np.abs(clean_data[:,1,:]) ** 2))
    EbN0_linear = 10**(EbN0_db/10)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0/2)
    # 生成I/Q两路独立高斯噪声
    noise_I = noise_std * np.random.randn(*clean_data[:,0,:].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:,1,:].shape)
    
    # 加噪（模拟实际信道）
    noisy_data[:,0,:] = clean_data[:,0,:] + noise_I;  # I路接收信号
    noisy_data[:,1,:] = clean_data[:,1,:] + noise_Q;  # Q路接收信号

    return noisy_data

def add_awgn_noise_torch(clean_data, EbN0_db):
    """
    给信号添加高斯白噪声 (PyTorch版)
    """
    noisy_data = clean_data.clone()
    signal_power = torch.mean((torch.abs(clean_data[:, 0, :]) ** 2) + (torch.abs(clean_data[:, 1, :]) ** 2))
    EbN0_linear = 10 ** (EbN0_db / 10.0)
    N0 = signal_power / EbN0_linear
    noise_std = torch.sqrt(N0 / 2.0)

    if isinstance(noise_std, torch.Tensor) and noise_std.ndim == 3:
        noise_std = noise_std.view(clean_data.shape[0], 1)
    
    noise_I = noise_std * torch.randn_like(clean_data[:, 0, :])
    noise_Q = noise_std * torch.randn_like(clean_data[:, 1, :])
    
    noisy_data[:, 0, :] = clean_data[:, 0, :] + noise_I
    noisy_data[:, 1, :] = clean_data[:, 1, :] + noise_Q

    return noisy_data
# ==========================================
# 2. 波形可视化 (适配 Hybrid Anchor 策略)
# ==========================================
def visualize_comparison_overlay(model, is2b, tx_clean, rx_faded, h_np, snr_list, sps=16, 
                                 device='cuda', save_dir='IS2B/resnet_is2b/vis_compare_overlay'):
    """
    可视化对比：波形叠加 (Waveform Overlay)
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # === 随机挑选一个样本 ===
    num_samples = tx_clean.shape[0]
    idx_to_plot = np.random.randint(0, num_samples)
    print(f"[Waveform] Plotting sample index: {idx_to_plot}")
    
    # 取出数据 [1, 2, L]
    x_sample_clean = tx_clean[idx_to_plot:idx_to_plot+1] 
    y_sample_faded = rx_faded[idx_to_plot:idx_to_plot+1] 
    h_sample = h_np[idx_to_plot:idx_to_plot+1]           

    x_tensor_clean = torch.from_numpy(x_sample_clean).float().to(device)
    y_tensor_faded = torch.from_numpy(y_sample_faded).float().to(device)
    h_tensor = torch.from_numpy(h_sample).float().to(device)
    
    mid_point = h_sample.shape[-1] // 2 if h_sample.ndim > 2 else 0
    if h_sample.ndim > 2: h_val = h_sample[0, :, mid_point]
    else: h_val = h_sample[0, :]
    h_abs = np.sqrt(h_val[0]**2 + h_val[1]**2)

    for snr_db in snr_list:
        snr_db_sample = snr_db - 10 * math.log10(sps)
        y_tensor_noisy = add_awgn_noise_torch(y_tensor_faded, snr_db_sample)
        y_sample_noisy = y_tensor_noisy.cpu().numpy()

        with torch.no_grad():
            # A. One-Step Prediction (生成 Anchor)
            net_input = torch.cat([y_tensor_noisy, h_tensor], dim=1)
            t_max = torch.full((1,), is2b.n_steps - 1, device=device, dtype=torch.long)
            x_onestep = model(net_input, t_max)
            x_onestep_np = x_onestep.cpu().numpy()

            # B. Rectified Flow (Hybrid Sampling)
            x_rf = is2b.sample(
                y=y_tensor_noisy, 
                h=h_tensor, 
                guidance_scale=1.0, 
                stop_t=0.0,      
                anchor=x_onestep 
            )
            x_rf_np = x_rf.cpu().numpy()
        
        # 绘图
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # Row 1: I
        axs[0, 0].plot(x_sample_clean[0, 0, :], label='Clean', color='gray', alpha=0.5, linewidth=2)
        axs[0, 0].plot(y_sample_faded[0, 0, :], label='Faded', color='purple', linestyle='--')
        axs[0, 0].set_title(f"Ref I (|h|={h_abs:.2f})")
        axs[0, 0].legend()
        
        axs[0, 1].plot(y_sample_noisy[0, 0, :], label='Noisy Input', color='red', alpha=0.7)
        axs[0, 1].set_title(f"Input I (SNR {snr_db}dB)")
        
        axs[0, 2].plot(x_sample_clean[0, 0, :], label='Clean', color='gray', alpha=0.3, linewidth=3)
        axs[0, 2].plot(x_onestep_np[0, 0, :], label='One-Step', color='blue', linestyle='--', alpha=0.8)
        axs[0, 2].plot(x_rf_np[0, 0, :], label='Rectified Flow (Hybrid)', color='green', alpha=0.9)
        axs[0, 2].set_title("Result I")
        axs[0, 2].legend()

        # Row 2: Q
        axs[1, 0].plot(x_sample_clean[0, 1, :], label='Clean', color='gray', alpha=0.5, linewidth=2)
        axs[1, 0].plot(y_sample_faded[0, 1, :], label='Faded', color='purple', linestyle='--')
        
        axs[1, 1].plot(y_sample_noisy[0, 1, :], label='Noisy Input', color='orange', alpha=0.7)
        
        axs[1, 2].plot(x_sample_clean[0, 1, :], label='Clean', color='gray', alpha=0.3, linewidth=3)
        axs[1, 2].plot(x_onestep_np[0, 1, :], label='One-Step', color='blue', linestyle='--', alpha=0.8)
        axs[1, 2].plot(x_rf_np[0, 1, :], label='Rectified Flow (Hybrid)', color='green', alpha=0.9)
        axs[1, 2].set_title("Result Q")

        plt.suptitle(f"Waveform Comparison (Hybrid Anchor) - SNR: {snr_db}dB", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"waveform_snr{snr_db}_idx{idx_to_plot}.png"))
        plt.close()

# ==========================================
# 3. 星座图可视化 (核心修改部分)
# ==========================================
def visualize_constellation_comparison(model, is2b, tx_clean, rx_faded, h_np, snr_list, sps=16, 
                                       device='cuda', save_dir='IS2B/resnet_is2b/vis_constellation',
                                       num_points=2048):
    """
    可视化对比：星座图 (Constellation Diagram)
    修改：移除 Overlay，改为绘制 Clean Reference
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # 截取一批数据用于画星座图
    limit = min(num_points, tx_clean.shape[0])
    print(f"[Constellation] Using {limit} samples for constellation plots...")
    
    x_batch = tx_clean[:limit]
    y_batch = rx_faded[:limit]
    h_batch = h_np[:limit]
    
    # 转 Tensor
    y_tensor_faded = torch.from_numpy(y_batch).float().to(device)
    h_tensor = torch.from_numpy(h_batch).float().to(device)
    
    # 中点索引
    mid = x_batch.shape[2] // 2

    # 提取 Ground Truth (Clean) 数据 - 它不随 SNR 变化，提前提取
    gt_I = x_batch[:, 0, mid]
    gt_Q = x_batch[:, 1, mid]

    for snr_db in snr_list:
        print(f"  -> Processing Constellation for SNR={snr_db}dB...")
        snr_db_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)
        
        # 1. 加噪
        y_tensor_noisy = add_awgn_noise_torch(y_tensor_faded, snr_db_sample)
        
        with torch.no_grad():
            # 2. One-Step (Anchor)
            net_input = torch.cat([y_tensor_noisy, h_tensor], dim=1)
            t_max = torch.full((limit,), is2b.n_steps - 1, device=device, dtype=torch.long)
            x_onestep = model(net_input, t_max)
            
            # 3. Rectified Flow (Hybrid)
            x_rf = is2b.sample(
                y=y_tensor_noisy, 
                h=h_tensor, 
                guidance_scale=1.0, 
                stop_t=0.0,      
                anchor=x_onestep 
            )

        # 4. 提取中点数据 (转 Numpy)
        y_np = y_tensor_noisy.cpu().numpy()
        y_I = y_np[:, 0, mid]; y_Q = y_np[:, 1, mid]
        
        os_np = x_onestep.cpu().numpy()
        os_I = os_np[:, 0, mid]; os_Q = os_np[:, 1, mid]
        
        rf_np = x_rf.cpu().numpy()
        rf_I = rf_np[:, 0, mid]; rf_Q = rf_np[:, 1, mid]

        # 5. 绘图 (1行4列)
        fig, axs = plt.subplots(1, 4, figsize=(24, 6))
        
        # Config
        alpha_val = 0.4
        s_val = 5
        lim = 2.5
        
        # Subplot 1: Noisy Input
        axs[0].scatter(y_I, y_Q, s=s_val, c='red', alpha=alpha_val, label='Received')
        axs[0].set_title(f"Received (Faded+Noisy)\nSNR={snr_db}dB")
        axs[0].set_xlim(-lim, lim); axs[0].set_ylim(-lim, lim)
        axs[0].grid(True, linestyle='--', alpha=0.5)
        
        # Subplot 2: One-Step Output
        axs[1].scatter(os_I, os_Q, s=s_val, c='blue', alpha=alpha_val, label='One-Step')
        axs[1].set_title("One-Step Prediction")
        axs[1].set_xlim(-lim, lim); axs[1].set_ylim(-lim, lim)
        axs[1].grid(True, linestyle='--', alpha=0.5)
        
        # Subplot 3: Rectified Flow Output (Hybrid)
        axs[2].scatter(rf_I, rf_Q, s=s_val, c='green', alpha=alpha_val, label='Rectified Flow (Hybrid)')
        axs[2].set_title("RF Output (Hybrid Anchor)")
        axs[2].set_xlim(-lim, lim); axs[2].set_ylim(-lim, lim)
        axs[2].grid(True, linestyle='--', alpha=0.5)

        # Subplot 4: Clean Reference (Ground Truth) -- 修改部分
        axs[3].scatter(gt_I, gt_Q, s=s_val, c='black', alpha=alpha_val, label='Clean Ref')
        axs[3].set_title("Clean Reference (Ground Truth)")
        axs[3].set_xlim(-lim, lim); axs[3].set_ylim(-lim, lim)
        axs[3].grid(True, linestyle='--', alpha=0.5)

        plt.suptitle(f"Constellation Diagram Comparison (Hybrid) - SNR: {snr_db}dB", fontsize=16)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"constellation_snr{snr_db}.png"))
        plt.close()

# ==========================================
# 4. 主程序入口
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 20
    sps = 16 
    
    # === 路径配置 ===
    ckpt_path = fr'IS2B/rIS2B_nakagmi_resnet_adjust/results/best_model_IS2B_resnet_pro_scope_{n_steps}.pth'
    vis_wav_dir = 'IS2B/rIS2B_nakagmi_resnet_adjust/vis_waveforms'
    vis_con_dir = 'IS2B/rIS2B_nakagmi_resnet_adjust/vis_constellations'

    # === 1. 构建 TimeResNet1D 模型 ===
    print(f"Building TimeResNet1D on {device}...")
    model = DilatedTimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=128,   # 宽度
        num_blocks=12,    # 深度可以加深，例如 12 层
        time_emb_dim=128
    ).to(device)

    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}，请检查路径！")

    # 初始化 IS2B
    is2b_instance = IS2B(model, n_steps=n_steps, device=device)

    # === 2. 加载数据 ===
    print("正在加载数据...")
    # 我们多取一点数据，比如取 2048 个用于画星座图
    test_data = QPSKDataset(400000, 402048) 
    
    tx_clean = test_data.x   
    rx_faded = test_data.y   
    h_np = test_data.z       

    snr_list = [0, 5, 10, 15, 20] 
    
    # === 3. 运行波形可视化 (单样本) ===
    visualize_comparison_overlay(
        model=model,
        is2b=is2b_instance,
        tx_clean=tx_clean,
        rx_faded=rx_faded,
        h_np=h_np,
        snr_list=snr_list,
        sps=sps,
        device=device,
        save_dir=vis_wav_dir
    )
    
    # === 4. 运行星座图可视化 (批量样本) ===
    visualize_constellation_comparison(
        model=model,
        is2b=is2b_instance,
        tx_clean=tx_clean,
        rx_faded=rx_faded,
        h_np=h_np,
        snr_list=snr_list,
        sps=sps,
        device=device,
        save_dir=vis_con_dir,
        num_points=2048 
    )
    
    print("所有可视化任务已完成。")