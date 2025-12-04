import os
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import csv
from torch.utils.data import DataLoader

# === 引入项目模块 ===
from dataset.dataset import QPSKDataset
from test_fig_x_pre import add_awgn_noise_torch

# ==========================================
# 1. 模型定义 (必须与训练代码完全一致)
# ==========================================
class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class SimpleResNet1D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=64, num_blocks=6):
        super().__init__()
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_dim) for _ in range(num_blocks)
        ])
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x, t=None):
        x = x.float()
        out = self.entry(x)
        for block in self.blocks:
            out = block(out)
        out = self.exit(out)
        return out

# ==========================================
# 2. 判决函数
# ==========================================
def decision_making(symbols_complex):
    """QPSK 硬判决"""
    real_part = np.real(symbols_complex)
    imag_part = np.imag(symbols_complex)
    
    bits = np.zeros((len(symbols_complex), 2), dtype=int)
    bits[(real_part > 0) & (imag_part > 0)] = (0, 0)
    bits[(real_part < 0) & (imag_part > 0)] = (0, 1)
    bits[(real_part < 0) & (imag_part < 0)] = (1, 1)
    bits[(real_part > 0) & (imag_part < 0)] = (1, 0)
    return bits

# ==========================================
# 3. 测试主逻辑
# ==========================================
def test_resnet_performance(model, test_loader, all_labels_iq, snr_range, device, sps):
    model.eval()
    ber_results = []
    
    # 确定中点位置
    dummy_x, _, _ = next(iter(test_loader))
    L = dummy_x.shape[2]
    mid_point = L // 2
    
    print(f"🚀 开始测试 SimpleResNet1D (Sampling Index: {mid_point})...")
    
    for snr_db in snr_range:
        # 换算采样 SNR
        snr_sample = snr_db - 10 * math.log10(sps)
        
        total_err = 0
        total_bits = 0
        
        # 遍历测试集
        for batch_idx, (clean_x, faded_y, h_est) in enumerate(tqdm(test_loader, desc=f"SNR {snr_db}dB", leave=False)):
            
            clean_x = clean_x.to(device).float()
            faded_y = faded_y.to(device).float()
            h_est = h_est.to(device).float()
            
            batch_size = clean_x.shape[0]
            seq_len = clean_x.shape[2]

            # h 扩展
            if h_est.dim() == 2:
                h_expanded = h_est.unsqueeze(-1).repeat(1, 1, seq_len)
            else:
                h_expanded = h_est

            # 加噪
            noisy_y = add_awgn_noise_torch(faded_y, snr_sample)

            # 构造输入
            net_input = torch.cat([noisy_y, h_expanded], dim=1)

            # 预测
            with torch.no_grad():
                pred_x = model(net_input, None) 
            
            # 中心采样 & 转复数
            pred_np = pred_x.cpu().numpy()
            pred_i = pred_np[:, 0, mid_point]
            pred_q = pred_np[:, 1, mid_point]
            pred_symbols = pred_i + 1j * pred_q 
            
            # 判决
            pred_bits = decision_making(pred_symbols) 
            
            # 获取对应标签
            start_idx = batch_idx * test_loader.batch_size
            end_idx = start_idx + batch_size
            current_labels = all_labels_iq[start_idx : end_idx]
            
            # 计算误码
            err_i = np.sum(current_labels[:, 0] != pred_bits[:, 0])
            err_q = np.sum(current_labels[:, 1] != pred_bits[:, 1])
            
            total_err += (err_i + err_q)
            total_bits += (batch_size * 2)
            
        avg_ber = total_err / total_bits
        ber_results.append(avg_ber)
        print(f"SNR: {snr_db}dB | BER: {avg_ber:.6e}")
        
    return ber_results

# ==========================================
# 4. 绘图与保存函数 (已更新：支持 Baseline)
# ==========================================
def save_and_plot(snr_range, ber_list, ref_bers, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    csv_path = os.path.join(save_dir, 'ber_results_resnet.csv')
    png_path = os.path.join(save_dir, 'ber_curve_compare.png')
    
    # 1. 保存模型预测结果到 CSV
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr_db', 'ber'])
        for s, b in zip(snr_range, ber_list):
            writer.writerow([s, f"{b:.6e}"])
    print(f"模型 BER 数据已保存至: {csv_path}")
    
    # 2. 绘图
    plt.figure(figsize=(10, 6))
    
    # 绘制本模型曲线
    plt.semilogy(snr_range, ber_list, 'o-', color='red', label='SimpleResNet1D (Supervised)')
    
    # 绘制 Baseline (如果存在)
    if len(ref_bers) > 0:
        # 截断以匹配长度 (防止维度不一致报错)
        limit = min(len(snr_range), len(ref_bers))
        plt.semilogy(snr_range[:limit], ref_bers[:limit], 's--', color='blue', alpha=0.7, label='Baseline (MMSE)')
        print("已添加 Baseline 曲线。")
    else:
        print("未检测到 Baseline 数据，仅绘制模型曲线。")

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR (dB)')
    plt.ylabel('BER')
    plt.title('BER Performance Comparison')
    plt.legend()
    plt.ylim(1e-6, 1.0) # 限制Y轴防止显示异常
    plt.savefig(png_path)
    print(f"对比曲线图已保存至: {png_path}")

# ==========================================
# 5. 入口
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4096
    sps = 16 
    
    # === 路径配置 ===
    # 模型权重路径
    model_path = r'CNN/RESNET/results/best_model_resnet.pth' 
    # 结果保存目录
    save_dir = r'CNN/RESNET/results'
    # 标签文件
    label_file_path = r'F:\LJN\bishe\bishe\data\rayleigh_data_all_h\labels.npy'
    # 基准文件路径 (Baseline)
    baseline_csv_path = r'CNN/RESNET/ber_results/baseline_ber.csv'

    # === 1. 数据准备 ===
    test_start = 400000
    test_end = 500000

    print("Loading Test Data...")
    test_dataset = QPSKDataset(test_start, test_end)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    print("Loading Labels...")
    all_labels_raw = np.load(label_file_path)
    test_labels_raw = all_labels_raw[test_start:test_end]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    test_labels_iq = np.array([map_label[int(v)] for v in test_labels_raw], dtype=int)
    
    # === 2. 加载模型 ===
    print(f"Loading Model from {model_path}...")
    model = SimpleResNet1D(in_channels=4, out_channels=2, hidden_dim=64, num_blocks=6).to(device)
    
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("✅ Model weights loaded.")
    else:
        raise FileNotFoundError(f"❌ 权重文件未找到: {model_path}")

    # === 3. 运行模型测试 ===
    snr_range = np.arange(2, 19, 1) # 2 ~ 18 dB
    
    model_bers = test_resnet_performance(
        model=model,
        test_loader=test_loader,
        all_labels_iq=test_labels_iq,
        snr_range=snr_range,
        device=device,
        sps=sps
    )

    # === 4. 读取基准 BER (你提供的代码片段) ===
    print("Reading Baseline Data...")
    ref_bers = []
    if os.path.exists(baseline_csv_path):
        try:
            with open(baseline_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                baseline_data = {float(row['snr_db']): float(row['baseline_ber']) for row in reader}
                for snr in snr_range:
                    ref_bers.append(baseline_data.get(snr, 0.0))
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"Warning: Baseline file not found at {baseline_csv_path}")

    # === 5. 绘图与保存 ===
    save_and_plot(snr_range, model_bers, ref_bers, save_dir)