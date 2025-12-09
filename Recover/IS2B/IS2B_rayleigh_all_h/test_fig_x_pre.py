import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 引入项目模块
from model.unet import build_network
from IS2B_x_pre import IS2B
from dataset.dataset import QPSKDataset

# 中文字体设置
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# ==========================================
# 1. 辅助工具函数
# ==========================================

def add_awgn_noise_np(clean_data, EbN0_db):
    """
    给信号添加高斯白噪声
    """
    signal_power = np.mean((np.abs(clean_data[:,0,:]) ** 2) + (np.abs(clean_data[:,1,:]) ** 2))
    EbN0_linear = 10**(EbN0_db/10)
    N0 = signal_power / EbN0_linear
    noise_std = np.sqrt(N0/2)
    
    noise_I = noise_std * np.random.randn(*clean_data[:,0,:].shape)
    noise_Q = noise_std * np.random.randn(*clean_data[:,1,:].shape)
    
    noisy_data = np.zeros_like(clean_data)
    noisy_data[:,0,:] = clean_data[:,0,:] + noise_I
    noisy_data[:,1,:] = clean_data[:,1,:] + noise_Q
    return noisy_data

def find_matching_timestep(schedule_class, snr_db, sps=16):
    """
    核心函数：根据物理信噪比，找到扩散过程中对应的"虚拟时间步 t"
    """
    snr_db_sample = snr_db - 10 * math.log10(sps)
    target_snr = 10 ** (snr_db_sample / 10.0)
    
    if hasattr(schedule_class, 'alpha_bars'):
        alpha_bars = schedule_class.alpha_bars.cpu().numpy()
    else:
        raise AttributeError("调度类中找不到 alpha_bars，请确保使用了 DDPM 调度策略")
    
    diff_snrs = alpha_bars / (1 - alpha_bars + 1e-8)
    
    diff = np.abs(diff_snrs - target_snr)
    best_t = np.argmin(diff)
    
    return int(best_t), alpha_bars[best_t]

# ==========================================
# 2. 可视化核心逻辑
# ==========================================

def visualize_oneshot_restoration(model, schedule_instance, tx_clean, rx_faded, h_np, snr_list, sps=16, 
                                  device='cuda', save_dir='IS2B/IS2B_rayleigh_all_h/vis_results'):
    """
    可视化：把含噪衰落信号视为 x_t，进行单步恢复
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # === 修改点：随机挑选一个样本 ===
    num_samples = tx_clean.shape[0]
    idx_to_plot = np.random.randint(0, num_samples)
    print(f"本次随机选取的样本索引: {idx_to_plot}")
    
    # 取出数据 [1, 2, L]
    x_sample_clean = tx_clean[idx_to_plot:idx_to_plot+1] 
    y_sample_faded = rx_faded[idx_to_plot:idx_to_plot+1] 
    h_sample = h_np[idx_to_plot:idx_to_plot+1]           

    # 转为 Tensor
    h_tensor = torch.from_numpy(h_sample).float().to(device)
    
    # 构造标题用的信道字符串 (取中间点)
    mid_point = h_sample.shape[-1] // 2 if h_sample.ndim > 2 else 0
    if h_sample.ndim > 2:
        h_val = h_sample[0, :, mid_point]
    else:
        h_val = h_sample[0, :]
        
    h_val_str = f"{h_val[0]:.2f} + {h_val[1]:.2f}j"
    h_abs = np.sqrt(h_val[0]**2 + h_val[1]**2)

    for snr_db in snr_list:
        # 1. 准备含噪接收信号 y
        snr_db_sample = snr_db - 10 * math.log10(sps)
        y_sample_noisy = add_awgn_noise_np(y_sample_faded, snr_db_sample)
        y_tensor_raw = torch.from_numpy(y_sample_noisy).float().to(device)

        # 2. === 核心：寻找对应的时间步 t ===
        t_idx, current_alpha_bar = find_matching_timestep(schedule_instance, snr_db, sps)
        
        # 3. === 核心：幅度缩放 ===
        scale_factor = math.sqrt(current_alpha_bar)
        x_t_input = y_tensor_raw * scale_factor

        print(f"SNR={snr_db}dB -> Matched Step t={t_idx}/{schedule_instance.n_steps}, Scale={scale_factor:.3f}")

        # 4. 单步预测 (One-shot Prediction)
        with torch.no_grad():
            t_tensor = torch.full((x_t_input.shape[0],), t_idx, device=device, dtype=torch.long)
            
            # 扩展 h
            if h_tensor.dim() == 2:
                h_expanded = h_tensor.unsqueeze(-1).repeat(1, 1, x_t_input.shape[2])
            else:
                h_expanded = h_tensor
                
            net_input = torch.cat([x_t_input, h_expanded], dim=1)
            
            # 预测 x0
            x_rec = model(net_input, t_tensor)
            
            # 截断
            x_rec = torch.clamp(x_rec, -2.0, 2.0)
        
        x_rec_np = x_rec.cpu().numpy()
        
        # 5. 绘图
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # --- I 路 (第一行) ---
        # Col 1: 参考
        axs[0, 0].plot(x_sample_clean[0, 0, :], label='原始信号(干净)', color='gray', linewidth=2, alpha=0.5)
        axs[0, 0].plot(y_sample_faded[0, 0, :], label='衰落信号(无噪)', color='purple', linestyle='--', linewidth=1.5)
        axs[0, 0].set_title(f"参考：信道影响 (|h|={h_abs:.2f})", fontsize=12)
        axs[0, 0].legend(loc='upper right', fontsize=9)
        axs[0, 0].grid(True, linestyle=':', alpha=0.6)
        
        # Col 2: 输入
        axs[0, 1].plot(y_sample_noisy[0, 0, :], label='含噪输入 (Raw y)', color='red', alpha=0.7)
        axs[0, 1].set_title(f"输入：衰落+加噪 (SNR {snr_db}dB)", fontsize=12)
        axs[0, 1].legend(loc='upper right', fontsize=9)
        axs[0, 1].grid(True, linestyle=':', alpha=0.6)

        # Col 3: 输出
        axs[0, 2].plot(x_sample_clean[0, 0, :], label='原始真值', color='gray', alpha=0.4, linestyle='-')
        axs[0, 2].plot(x_rec_np[0, 0, :], label='预测 x0', color='green', linewidth=1.5)
        axs[0, 2].set_title(f"输出：单步预测结果 (Assume t={t_idx})", fontsize=12)
        axs[0, 2].legend(loc='upper right', fontsize=9)
        axs[0, 2].grid(True, linestyle=':', alpha=0.6)

        # --- Q 路 (第二行) ---
        # Col 1
        axs[1, 0].plot(x_sample_clean[0, 1, :], label='原始信号', color='gray', linewidth=2, alpha=0.5)
        axs[1, 0].plot(y_sample_faded[0, 1, :], label='衰落信号', color='purple', linestyle='--', linewidth=1.5)
        axs[1, 0].set_title(f"参考：Q路", fontsize=12)
        axs[1, 0].grid(True, linestyle=':', alpha=0.6)

        # Col 2
        axs[1, 1].plot(y_sample_noisy[0, 1, :], label='含噪输入', color='blue', alpha=0.7)
        axs[1, 1].set_title(f"输入：Q路", fontsize=12)
        axs[1, 1].grid(True, linestyle=':', alpha=0.6)

        # Col 3
        axs[1, 2].plot(x_sample_clean[0, 1, :], label='原始真值', color='gray', alpha=0.4, linestyle='-')
        axs[1, 2].plot(x_rec_np[0, 1, :], label='预测 x0', color='orange', linewidth=1.5)
        axs[1, 2].set_title(f"输出：Q路", fontsize=12)
        axs[1, 2].grid(True, linestyle=':', alpha=0.6)

        plt.suptitle(f"模型去噪能力测试 (把y视为中间状态t={t_idx}) \n信道 h = {h_val_str}, Sample Index: {idx_to_plot}", fontsize=16)
        plt.tight_layout()
        
        # 文件名包含 idx，防止覆盖
        save_name = os.path.join(save_dir, f"vis_oneshot_snr{snr_db}_idx{idx_to_plot}.png")
        plt.savefig(save_name)
        plt.close()
        print(f"可视化结果已保存至: {save_name}")


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 250
    sps = 16
    
    ckpt_path = fr'IS2B/IS2B_rayleigh_all_h/results/best_model_epoch_with_n_step{n_steps}_x0.pth'
    vis_save_dir = 'IS2B/IS2B_rayleigh_all_h/vis_results'

    net_cfg = {'type': 'UNet', 'channels': [32, 64, 128, 256], 'pe_dim': 128, 'in_channels': 4, 'out_channels': 2}
    model = build_network(net_cfg, n_steps).to(device)

    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}，请检查路径！")

    schedule_instance = IS2B(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    print("正在加载数据...")
    test_data = QPSKDataset(400000, 400100) # 取 100 个样本
    
    tx_clean = test_data.x   
    rx_faded = test_data.y   
    h_np = test_data.z       

    snr_list = [0, 5, 10, 20] 
    
    visualize_oneshot_restoration(
        model=model,
        schedule_instance=schedule_instance,
        tx_clean=tx_clean,
        rx_faded=rx_faded,
        h_np=h_np,
        snr_list=snr_list,
        sps=sps,
        device=device,
        save_dir=vis_save_dir
    )
    
    print("所有绘图任务已完成。")