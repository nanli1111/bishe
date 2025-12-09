import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import csv
from torch.utils.data import DataLoader

# === 引入项目模块 ===
from model.unet import build_network
from dataset.dataset import QPSKDataset
from test_fig_x_pre import add_awgn_noise_torch

# === 1. 判决函数 ===
def decision_making(symbols_complex):
    """
    QPSK 硬判决
    输入: Numpy Complex Array shape (B,)
    输出: Numpy Int Array shape (B, 2)
    """
    real_part = np.real(symbols_complex)
    imag_part = np.imag(symbols_complex)
    
    bits = np.zeros((len(symbols_complex), 2), dtype=int)
    # 00: (+, +), 01: (-, +), 11: (-, -), 10: (+, -)
    bits[(real_part > 0) & (imag_part > 0)] = (0, 0)
    bits[(real_part < 0) & (imag_part > 0)] = (0, 1)
    bits[(real_part < 0) & (imag_part < 0)] = (1, 1)
    bits[(real_part > 0) & (imag_part < 0)] = (1, 0)
    return bits

# === 2. 测试主函数 ===
def test_supervised_model(model, test_loader, all_labels_iq, 
                          snr_range, device, sps, save_dir):
    
    model.eval()
    ber_results = []
    
    # 获取序列长度 L (假设所有样本长度一致)
    # 从 loader 取一个样本看形状
    dummy_x, _, _ = next(iter(test_loader))
    L = dummy_x.shape[2]
    mid_point = L // 2
    
    print(f"🚀 开始测试 (Mode: Supervised Direct Pred | Sampling: Index {mid_point})")
    
    for snr_db in snr_range:
        # 换算 SNR
        snr_sample = snr_db - 10 * math.log10(sps)
        
        total_err = 0
        total_bits = 0
        
        # 遍历测试集
        # 使用 enumerate 配合 batch_size 来定位对应的标签
        for batch_idx, (clean_x, faded_y, h_est) in enumerate(tqdm(test_loader, desc=f"SNR {snr_db}dB", leave=False)):
            
            clean_x = clean_x.to(device).float()
            faded_y = faded_y.to(device).float()
            h_est = h_est.to(device).float()
            
            batch_size = clean_x.shape[0]
            seq_len = clean_x.shape[2]

            # 1. 扩展 h
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # 2. 加噪
            noisy_y = add_awgn_noise_torch(faded_y, snr_sample)

            # 3. 构造输入 (纯监督模型 t 恒为 0)
            t_dummy = torch.zeros(batch_size, device=device, dtype=torch.long)
            net_input = torch.cat([noisy_y, h_expanded], dim=1)

            # 4. 预测
            with torch.no_grad():
                pred_x = model(net_input, t_dummy) # Output: [B, 2, L]
            
            # 5. 中心采样 & 转复数
            pred_np = pred_x.cpu().numpy()
            pred_i = pred_np[:, 0, mid_point]
            pred_q = pred_np[:, 1, mid_point]
            pred_symbols = pred_i + 1j * pred_q # shape (B,)
            
            # 6. 判决
            pred_bits = decision_making(pred_symbols) # shape (B, 2)
            
            # 7. 获取对应标签
            # 计算当前 batch 在总标签中的索引范围
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + batch_size
            
            # 防止最后不足一个 batch 导致索引越界
            current_labels = all_labels_iq[start_idx : end_idx]
            
            # 8. 计算误码
            # current_labels 是 (B, 2)
            err_i = np.sum(current_labels[:, 0] != pred_bits[:, 0])
            err_q = np.sum(current_labels[:, 1] != pred_bits[:, 1])
            
            total_err += (err_i + err_q)
            total_bits += (batch_size * 2)
            
        # 计算当前 SNR 下的总 BER
        avg_ber = total_err / total_bits
        ber_results.append(avg_ber)
        print(f"SNR: {snr_db}dB | BER: {avg_ber:.6e}")
        
    return ber_results

# === 3. 保存与绘图 ===
def save_and_plot(snr_range, ber_list, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'ber_results_supervised.csv')
    png_path = os.path.join(save_dir, 'ber_curve_supervised.png')
    
    # 保存 CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr_db', 'ber'])
        for s, b in zip(snr_range, ber_list):
            writer.writerow([s, f"{b:.6e}"])
    print(f"数据已保存至: {csv_path}")
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.semilogy(snr_range, ber_list, 'o-', color='red', label='Supervised Model (UNet)')
    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER Performance: Supervised Learning')
    plt.legend()
    plt.savefig(png_path)
    print(f"曲线图已保存至: {png_path}")

if __name__ == "__main__":
    # === 配置 ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100
    batch_size = 4096 # 测试时可以开大一点
    sps = 16 
    
    # 权重路径 (请修改为你训练好的模型路径)
    model_path = r'IS2B/rIS2B_rayleigh_all_h/results/best_model_supervised.pth'
    
    # 结果保存目录
    save_dir = r'ber_results/supervised_test'
    
    # 数据范围 (测试集: 400000 ~ 500000)
    test_start = 400000
    test_end = 500000
    
    # 标签文件路径
    label_file_path = r'F:\LJN\bishe\bishe\data\rayleigh_data_all_h\labels.npy'

    # === 1. 准备数据 ===
    print("Loading Test Data...")
    # 关键：shuffle=False 保证顺序，num_workers=0 防止多进程乱序(虽然False通常没事，但0最稳)
    test_dataset = QPSKDataset(test_start, test_end)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("Loading Labels...")
    # 读取标签并截取对应部分
    all_labels_raw = np.load(label_file_path)
    test_labels_raw = all_labels_raw[test_start:test_end]
    
    # 映射为比特
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    test_labels_iq = np.array([map_label[int(v)] for v in test_labels_raw], dtype=int)
    
    print(f"Test Data Size: {len(test_dataset)}")
    print(f"Test Labels Size: {test_labels_iq.shape}")
    assert len(test_dataset) == test_labels_iq.shape[0], "数据量与标签量不匹配！"

    # === 2. 加载模型 ===
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 4,  
        'out_channels': 2 
    }
    print(f"Loading Model from {model_path}...")
    model = build_network(net_cfg, n_steps).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model Loaded Successfully.")
    else:
        raise FileNotFoundError(f"权重文件不存在: {model_path}")

    # === 3. 运行测试 ===
    snr_range = np.arange(2, 19, 1) # 2 ~ 18 dB
    
    ber_list = test_supervised_model(
        model=model,
        test_loader=test_loader,
        all_labels_iq=test_labels_iq,
        snr_range=snr_range,
        device=device,
        sps=sps,
        save_dir=save_dir
    )

    # === 4. 保存结果 ===
    save_and_plot(snr_range, ber_list, save_dir)