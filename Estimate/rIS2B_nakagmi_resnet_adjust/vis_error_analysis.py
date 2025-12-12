import os
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

# === 引入项目模块 ===
# 1. 导入 ResNet 模型
from model.resnet import TimeResNet1D
# 2. 导入 IS2B 包装器
from IS2B_x_pre import IS2B
# 3. 数据集与工具
from dataset.dataset import QPSKDataset
from test_fig_x_pre import add_awgn_noise_torch

# 中文字体设置
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 判决工具函数
# ==========================================
def get_hard_decision(symbols_complex):
    """
    输入: Numpy Complex Array (B,)
    输出: Numpy Int Array (B, 2)
    """
    real = np.real(symbols_complex)
    imag = np.imag(symbols_complex)
    bits = np.zeros((len(symbols_complex), 2), dtype=int)
    
    # 映射规则需与数据集一致: 00:(+,+), 01:(-,+), 11:(-,-), 10:(+,-)
    bits[(real > 0) & (imag > 0)] = (0, 0)
    bits[(real < 0) & (imag > 0)] = (0, 1)
    bits[(real < 0) & (imag < 0)] = (1, 1)
    bits[(real > 0) & (imag < 0)] = (1, 0)
    return bits

def check_correctness(pred_bits, true_bits):
    """
    检查比特是否完全正确 (Symbol Level Correctness)
    返回 Boolean Mask: (B,)
    """
    # 两个比特都对才算对
    return (pred_bits[:, 0] == true_bits[:, 0]) & (pred_bits[:, 1] == true_bits[:, 1])

# ==========================================
# 2. 核心可视化函数：抓取“反面教材”
# ==========================================
def visualize_specific_errors(model, is2b, tx_clean, rx_faded, h_np, snr_list, sps=16, 
                              device='cuda', save_dir='IS2B/rIS2B_rayleigh_all_h_resnet/vis_error_analysis'):
    """
    只画出：One-Step 对了，但 Rectified Flow 错了 的样本
    """
    os.makedirs(save_dir, exist_ok=True)
    model.eval()
    
    # 我们需要批量处理来寻找错误样本
    num_samples = tx_clean.shape[0]
    print(f"扫描 {num_samples} 个样本寻找特定的错误案例...")
    
    # 转为 Tensor
    x_tensor_clean = torch.from_numpy(tx_clean).float().to(device)
    y_tensor_faded = torch.from_numpy(rx_faded).float().to(device)
    h_tensor = torch.from_numpy(h_np).float().to(device)
    
    # 采样中点索引
    mid = tx_clean.shape[2] // 2
    
    # 获取 Ground Truth Bits
    clean_mid_np = x_tensor_clean.cpu().numpy()
    gt_symbols = clean_mid_np[:, 0, mid] + 1j * clean_mid_np[:, 1, mid]
    gt_bits = get_hard_decision(gt_symbols)

    for snr_db in snr_list:
        print(f"\n[SNR={snr_db}dB] Running Inference to find bad cases...")
        
        # 1. 准备含噪数据
        snr_db_sample = snr_db - 10 * math.log10(sps)
        y_tensor_noisy = add_awgn_noise_torch(y_tensor_faded, snr_db_sample)
        
        with torch.no_grad():
            # === A. One-Step Prediction ===
            net_input = torch.cat([y_tensor_noisy, h_tensor], dim=1)
            # t=1.0 对应索引 n_steps - 1
            t_max = torch.full((num_samples,), is2b.n_steps - 1, device=device, dtype=torch.long)
            
            # 使用新模型前向传播
            x_onestep = model(net_input, t_max)
            
            # 中点采样 & 判决
            os_np = x_onestep.cpu().numpy()
            os_symbols = os_np[:, 0, mid] + 1j * os_np[:, 1, mid]
            os_bits = get_hard_decision(os_symbols)
            
            # === B. Rectified Flow Sampling ===
            # 这里建议开启 guidance > 1.0 来观察过冲现象
            # 或者 stop_t=0.0 (完全采样) 来观察尾部抖动
            scale = 1.0 
            # 注意：sample 函数内部会自动处理模型调用
            x_rf = is2b.sample(y=y_tensor_noisy, h=h_tensor, guidance_scale=scale, stop_t=0.9)
            
            # 中点采样 & 判决
            rf_np = x_rf.cpu().numpy()
            rf_symbols = rf_np[:, 0, mid] + 1j * rf_np[:, 1, mid]
            rf_bits = get_hard_decision(rf_symbols)

        # === C. 筛选逻辑 ===
        # 条件：OneStep 正确 (True) AND RectifiedFlow 错误 (False)
        is_os_correct = check_correctness(os_bits, gt_bits)
        is_rf_correct = check_correctness(rf_bits, gt_bits)
        
        # 目标 Mask
        target_mask = is_os_correct & (~is_rf_correct)
        candidate_indices = np.where(target_mask)[0]
        
        print(f"  -> Found {len(candidate_indices)} samples where OneStep>RF.")
        
        if len(candidate_indices) == 0:
            print("  -> 本次 SNR 下没有发现此类样本，跳过绘图。")
            continue
            
        # === D. 绘图 (只画前 3 个找到的例子) ===
        plot_count = 0
        for idx in candidate_indices:
            if plot_count >= 3: break # 每个SNR只画3张
            plot_count += 1
            
            # 准备绘图数据
            h_val = h_np[idx]
            if h_val.ndim > 1: h_val_center = h_val[:, mid]
            else: h_val_center = h_val
            h_abs = np.sqrt(h_val_center[0]**2 + h_val_center[1]**2)
            
            # 数据切片
            clean_seq = x_tensor_clean[idx].cpu().numpy()
            noisy_seq = y_tensor_noisy[idx].cpu().numpy()
            os_seq = os_np[idx]
            rf_seq = rf_np[idx]
            faded_seq = rx_faded[idx]

            # 判决点数值 (用于 Title 展示)
            os_pt = os_symbols[idx]
            rf_pt = rf_symbols[idx]
            gt_pt = gt_symbols[idx]

            # 开始画图
            fig, axs = plt.subplots(2, 3, figsize=(18, 10))
            
            # Row 1: I Channel
            axs[0, 0].plot(clean_seq[0, :], label='Clean', color='gray', alpha=0.5, linewidth=2)
            axs[0, 0].plot(faded_seq[0, :], label='Faded', color='purple', linestyle='--')
            axs[0, 0].set_title(f"Ref I (|h|={h_abs:.2f})")
            axs[0, 0].legend()

            axs[0, 1].plot(noisy_seq[0, :], label='Input', color='red', alpha=0.7)
            axs[0, 1].set_title(f"Input I (SNR {snr_db}dB)")

            axs[0, 2].plot(clean_seq[0, :], label='Clean', color='gray', alpha=0.3, linewidth=3)
            axs[0, 2].plot(os_seq[0, :], label='One-Step (Right)', color='blue', linestyle='--')
            axs[0, 2].plot(rf_seq[0, :], label='RF (Wrong)', color='green')
            axs[0, 2].set_title(f"I: OS={os_pt.real:.2f}, RF={rf_pt.real:.2f}, GT={gt_pt.real:.0f}")
            axs[0, 2].legend()

            # Row 2: Q Channel
            axs[1, 0].plot(clean_seq[1, :], label='Clean', color='gray', alpha=0.5, linewidth=2)
            axs[1, 0].plot(faded_seq[1, :], label='Faded', color='purple', linestyle='--')
            
            axs[1, 1].plot(noisy_seq[1, :], label='Input', color='orange', alpha=0.7)
            
            axs[1, 2].plot(clean_seq[1, :], label='Clean', color='gray', alpha=0.3, linewidth=3)
            axs[1, 2].plot(os_seq[1, :], label='One-Step (Right)', color='blue', linestyle='--')
            axs[1, 2].plot(rf_seq[1, :], label='RF (Wrong)', color='green')
            axs[1, 2].set_title(f"Q: OS={os_pt.imag:.2f}, RF={rf_pt.imag:.2f}, GT={gt_pt.imag:.0f}")
            axs[1, 2].legend()

            plt.suptitle(f"Case Analysis: One-Step Correct vs RF Wrong\nIndex: {idx}, SNR: {snr_db}dB\n"
                         f"GT Bits: {gt_bits[idx]}, OS Bits: {os_bits[idx]}, RF Bits: {rf_bits[idx]}", fontsize=14)
            plt.tight_layout()
            save_name = os.path.join(save_dir, f"error_case_snr{snr_db}_idx{idx}.png")
            plt.savefig(save_name)
            plt.close()
            print(f"    Saved plot: {save_name}")

# ==========================================
# 主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 50 
    sps = 16 
    
    # 路径配置
    ckpt_path = fr'Estimate/rIS2B_rayleigh_all_h_resnet_stop/results/best_model_IS2B_resnet.pth'
    vis_save_dir = 'Estimate/rIS2B_rayleigh_all_h_resnet_stop/vis_error_analysis'

    # 1. 模型 (必须与训练一致)
    print(f"Building TimeResNet1D on {device}...")
    model = TimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=128,  
        num_blocks=8,    
        time_emb_dim=64  
    ).to(device)
    
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Model Loaded.")
    else:
        print(f"Error: Checkpoint not found at {ckpt_path}")
        exit()

    is2b_instance = IS2B(model, n_steps=n_steps, device=device)

    # 2. 数据 (取足够多的数据以保证能找到错误样本)
    print("Loading Data...")
    # 取 5000 个样本以增加找到 corner case 的概率
    test_data = QPSKDataset(400000, 405000) 
    
    tx_clean = test_data.x   
    rx_faded = test_data.y   
    h_np = test_data.z       

    # 重点关注低信噪比，那里更容易出现分歧
    snr_list = [0, 5, 10, 15] 
    
    # 3. 运行
    visualize_specific_errors(
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
    
    print("分析完成。")