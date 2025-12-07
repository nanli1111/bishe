import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === 引入项目模块 ===
from model.resnet import TimeResNet1D
from IS2B_x_pre import IS2B
from dataset.dataset import QPSKDataset

# 中文字体设置
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# ==========================================
# 可视化核心逻辑 (Overlay Comparison)
# ==========================================

# ==========================================
# 1. 辅助工具函数
# ==========================================
# 加噪
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
    完全复刻 add_awgn_noise_np 的逻辑：
    1. 计算全局平均功率
    2. I/Q 两路独立生成噪声
    """
    # 复制数据 (对应 np.zeros + 赋值)
    noisy_data = clean_data.clone()
    
    # 1. 计算信号功率 (全局平均)
    # 对应: np.mean((np.abs(clean_data[:,0,:]) ** 2) + (np.abs(clean_data[:,1,:]) ** 2))
    signal_power = torch.mean((torch.abs(clean_data[:, 0, :]) ** 2) + (torch.abs(clean_data[:, 1, :]) ** 2))

    # 2. 计算噪声标准差
    EbN0_linear = 10 ** (EbN0_db / 10.0)
    N0 = signal_power / EbN0_linear
    noise_std = torch.sqrt(N0 / 2.0)

    # 3. 维度适配 (关键步骤)
    # 训练时 EbN0_db 可能是 [B, 1, 1]，而 clean_data[:, 0, :] 是 [B, L]
    # 如果直接相乘会报错 [B, 1, 1] * [B, L] -> [B, B, L]
    # 我们需要把 noise_std 调整为 [B, 1] 以便正确广播
    if isinstance(noise_std, torch.Tensor) and noise_std.ndim == 3:
        noise_std = noise_std.view(clean_data.shape[0], 1)
    elif isinstance(noise_std, torch.Tensor) and noise_std.ndim == 0:
        # 如果是标量，不需要操作
        pass

    # 4. 生成 I/Q 两路独立高斯噪声
    # 对应: noise_std * np.random.randn(*clean_data[:,0,:].shape)
    noise_I = noise_std * torch.randn_like(clean_data[:, 0, :])
    noise_Q = noise_std * torch.randn_like(clean_data[:, 1, :])
    
    # 5. 加噪
    # 对应: noisy_data[:,0,:] = clean_data[:,0,:] + noise_I ...
    noisy_data[:, 0, :] = clean_data[:, 0, :] + noise_I
    noisy_data[:, 1, :] = clean_data[:, 1, :] + noise_Q

    return noisy_data

def visualize_comparison_overlay(model, is2b, tx_clean, rx_faded, h_np, snr_list, sps=16, 
                                 device='cuda', save_dir='IS2B/resnet_is2b/vis_compare_overlay'):
    """
    可视化对比：将 One-Step 和 Rectified Flow 的结果画在同一张图上
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # === 随机挑选一个样本 ===
    num_samples = tx_clean.shape[0]
    idx_to_plot = np.random.randint(0, num_samples)
    print(f"本次随机选取的样本索引: {idx_to_plot}")
    
    # 取出数据 [1, 2, L]
    x_sample_clean = tx_clean[idx_to_plot:idx_to_plot+1] 
    y_sample_faded = rx_faded[idx_to_plot:idx_to_plot+1] 
    h_sample = h_np[idx_to_plot:idx_to_plot+1]           

    # 转为 Tensor
    x_tensor_clean = torch.from_numpy(x_sample_clean).float().to(device)
    y_tensor_faded = torch.from_numpy(y_sample_faded).float().to(device)
    h_tensor = torch.from_numpy(h_sample).float().to(device)
    
    # 构造标题用的信道信息
    mid_point = h_sample.shape[-1] // 2 if h_sample.ndim > 2 else 0
    if h_sample.ndim > 2:
        h_val = h_sample[0, :, mid_point]
    else:
        h_val = h_sample[0, :]
    h_val_str = f"{h_val[0]:.2f} + {h_val[1]:.2f}j"
    h_abs = np.sqrt(h_val[0]**2 + h_val[1]**2)

    for snr_db in snr_list:
        # 1. 准备含噪接收信号 y (起点)
        snr_db_sample = snr_db - 10 * math.log10(sps)
        y_tensor_noisy = add_awgn_noise_torch(y_tensor_faded, snr_db_sample)
        y_sample_noisy = y_tensor_noisy.cpu().numpy()

        print(f"SNR={snr_db}dB -> Inference...")

        with torch.no_grad():
            # A. One-Step Prediction
            net_input = torch.cat([y_tensor_noisy, h_tensor], dim=1)
            t_max = torch.full((1,), is2b.n_steps - 1, device=device, dtype=torch.long)
            x_onestep = model(net_input, t_max)
            x_onestep = torch.clamp(x_onestep, -2.5, 2.5)
            x_onestep_np = x_onestep.cpu().numpy()

            # B. Rectified Flow Sampling
            x_rf = is2b.sample(
                y=y_tensor_noisy,
                h=h_tensor,
                guidance_scale=1.0
            )
            x_rf = torch.clamp(x_rf, -2.5, 2.5)
            x_rf_np = x_rf.cpu().numpy()
        
        # ==========================================
        # 3. 绘图 (2行3列) - 关键修改
        # ==========================================
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # --- Row 1: I 路 ---
        # Col 1: 参考
        axs[0, 0].plot(x_sample_clean[0, 0, :], label='Clean', color='gray', alpha=0.5, linewidth=2)
        axs[0, 0].plot(y_sample_faded[0, 0, :], label='Faded', color='purple', linestyle='--')
        axs[0, 0].set_title(f"Ref I (|h|={h_abs:.2f})")
        axs[0, 0].legend(loc='upper right')
        
        # Col 2: 输入
        axs[0, 1].plot(y_sample_noisy[0, 0, :], label='Input (Noisy)', color='red', alpha=0.7)
        axs[0, 1].set_title(f"Input I (SNR {snr_db}dB)")
        
        # Col 3: 对比 (Clean vs OneStep vs RF)
        axs[0, 2].plot(x_sample_clean[0, 0, :], label='Clean', color='gray', alpha=0.3, linewidth=3)
        axs[0, 2].plot(x_onestep_np[0, 0, :], label='One-Step', color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
        axs[0, 2].plot(x_rf_np[0, 0, :], label='Rectified Flow', color='green', linestyle='-', linewidth=1.5, alpha=0.9)
        axs[0, 2].set_title("Result Comparison I")
        axs[0, 2].legend(loc='upper right')

        # --- Row 2: Q 路 ---
        # Col 1
        axs[1, 0].plot(x_sample_clean[0, 1, :], label='Clean', color='gray', alpha=0.5, linewidth=2)
        axs[1, 0].plot(y_sample_faded[0, 1, :], label='Faded', color='purple', linestyle='--')
        axs[1, 0].set_title("Ref Q")
        
        # Col 2
        axs[1, 1].plot(y_sample_noisy[0, 1, :], label='Input (Noisy)', color='orange', alpha=0.7)
        axs[1, 1].set_title("Input Q")
        
        # Col 3: 对比
        axs[1, 2].plot(x_sample_clean[0, 1, :], label='Clean', color='gray', alpha=0.3, linewidth=3)
        axs[1, 2].plot(x_onestep_np[0, 1, :], label='One-Step', color='blue', linestyle='--', linewidth=1.5, alpha=0.8)
        axs[1, 2].plot(x_rf_np[0, 1, :], label='Rectified Flow', color='green', linestyle='-', linewidth=1.5, alpha=0.9)
        axs[1, 2].set_title("Result Comparison Q")
        axs[1, 2].legend(loc='upper right')

        # 全局标题
        plt.suptitle(f"IS2B Restoration Comparison (TimeResNet1D)\nSample Index: {idx_to_plot}, SNR: {snr_db}dB", fontsize=16)
        plt.tight_layout()
        
        # 保存
        save_name = os.path.join(save_dir, f"overlay_snr{snr_db}_idx{idx_to_plot}.png")
        plt.savefig(save_name)
        plt.close()
        print(f"Saved: {save_name}")

# ==========================================
# 主程序入口
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 50
    sps = 16 
    
    # === 路径配置 ===
    ckpt_path = fr'IS2B/rIS2B_rayleigh_all_h_resnet_cfg/results/best_model_IS2B_resnet.pth'
    vis_save_dir = 'IS2B/rIS2B_rayleigh_all_h_resnet_cfg/vis_results'

    # === 1. 构建 TimeResNet1D 模型 ===
    print(f"Building TimeResNet1D on {device}...")
    model = TimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=128,  
        num_blocks=8,    
        time_emb_dim=64  
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
    test_data = QPSKDataset(400000, 400100) 
    
    tx_clean = test_data.x   
    rx_faded = test_data.y   
    h_np = test_data.z       

    snr_list = [0, 5, 10, 15, 20] 
    
    # === 3. 开始可视化 ===
    visualize_comparison_overlay(
        model=model,
        is2b=is2b_instance,
        tx_clean=tx_clean,
        rx_faded=rx_faded,
        h_np=h_np,
        snr_list=snr_list,
        sps=sps,
        device=device,
        save_dir=vis_save_dir
    )
    
    print("对比绘图完成。")