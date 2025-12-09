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

# ==========================================
# 2. 可视化核心逻辑 (Rectified Flow)
# ==========================================

def visualize_rectified_flow_restoration(model, is2b, tx_clean, rx_faded, h_np, snr_list, sps=16, 
                                         device='cuda', save_dir='IS2B/rIS2B_rayleigh_all_h/vis_results'):
    """
    可视化：真正的 IS2B / Rectified Flow 采样
    逻辑：起点 x_1 = y (含噪衰落)，求解 ODE 走到 x_0
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
    
    # 构造标题用的信道字符串
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
        
        # 加噪 (这就是我们推理的起点)
        y_tensor_noisy = add_awgn_noise_torch(y_tensor_faded, snr_db_sample)
        y_sample_noisy = y_tensor_noisy.cpu().numpy() # 用于画图

        print(f"SNR={snr_db}dB -> Starting Rectified Flow Sampling...")

        # 2. 执行采样 (ODE Solver)
        with torch.no_grad():
            # 调用 IS2B 的 sample_rectified_flow
            # y_tensor_noisy 是起点 (t=1)
            # h_tensor 是条件
            x_rec = is2b.sample_rectified_flow(
                y=y_tensor_noisy,
                h=h_tensor
            )
            
            # 截断 (保持物理合理性)
            x_rec = torch.clamp(x_rec, -2.0, 2.0)
        
        x_rec_np = x_rec.cpu().numpy()
        
        # 3. 绘图
        fig, axs = plt.subplots(2, 3, figsize=(18, 10))
        
        # --- I 路 ---
        # Col 1: 参考
        axs[0, 0].plot(x_sample_clean[0, 0, :], label='Clean', color='gray', alpha=0.5)
        axs[0, 0].plot(y_sample_faded[0, 0, :], label='Faded', color='purple', linestyle='--')
        axs[0, 0].set_title(f"Ref I (|h|={h_abs:.2f})")
        axs[0, 0].legend()
        
        # Col 2: 输入 (起点 t=1)
        axs[0, 1].plot(y_sample_noisy[0, 0, :], label='Noisy Input (Start t=1)', color='red', alpha=0.7)
        axs[0, 1].set_title(f"Input I (SNR {snr_db}dB)")
        axs[0, 1].legend()
        
        # Col 3: 输出 (终点 t=0)
        axs[0, 2].plot(x_sample_clean[0, 0, :], label='Clean', color='gray', alpha=0.4)
        axs[0, 2].plot(x_rec_np[0, 0, :], label='Rectified Flow Output', color='green')
        axs[0, 2].set_title(f"Output I (Restored)")
        axs[0, 2].legend()

        # --- Q 路 ---
        axs[1, 0].plot(x_sample_clean[0, 1, :], label='Clean', color='gray', alpha=0.5)
        axs[1, 0].plot(y_sample_faded[0, 1, :], label='Faded', color='purple', linestyle='--')
        axs[1, 0].set_title("Ref Q")
        
        axs[1, 1].plot(y_sample_noisy[0, 1, :], label='Noisy Input (Start t=1)', color='blue', alpha=0.7)
        axs[1, 1].set_title("Input Q")
        
        axs[1, 2].plot(x_sample_clean[0, 1, :], label='Clean', color='gray', alpha=0.4)
        axs[1, 2].plot(x_rec_np[0, 1, :], label='Rectified Flow Output', color='orange')
        axs[1, 2].set_title("Output Q")

        plt.suptitle(f"IS2B 信号恢复 (Rectified Flow ODE) \n信道 h = {h_val_str}, Sample Index: {idx_to_plot}", fontsize=16)
        plt.tight_layout()
        
        save_name = os.path.join(save_dir, f"vis_rf_snr{snr_db}_idx{idx_to_plot}.png")
        plt.savefig(save_name)
        plt.close()
        print(f"Saved: {save_name}")


# ==========================================
# 3. 主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100         # 采样步数 (欧拉法步数)
    sps = 16              
    
    # 路径配置
    ckpt_path = fr'IS2B/rIS2B_rayleigh_all_h/results/best_model_IS2B_n{n_steps}.pth'
    vis_save_dir = 'IS2B/rIS2B_rayleigh_all_h/vis_results'

    # 加载模型 (in_channels=4: [x_t, h])
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 4,               
        'out_channels': 2
    }
    model = build_network(net_cfg, n_steps).to(device)

    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}，请检查路径！")

    # 初始化 IS2B (这里只需要 n_steps)
    is2b_instance = IS2B(model, n_steps=n_steps, device=device)

    # 加载数据
    print("正在加载数据...")
    test_data = QPSKDataset(400000, 400100) # 取 100 个样本
    
    tx_clean = test_data.x   
    rx_faded = test_data.y   
    h_np = test_data.z       

    snr_list = [0, 5, 10, 20] 
    
    visualize_rectified_flow_restoration(
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
    
    print("所有绘图任务已完成。")