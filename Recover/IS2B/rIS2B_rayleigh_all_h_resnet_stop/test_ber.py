import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv

# === 引入项目模块 ===
# 1. 导入新的 ResNet 模型
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
# 1. 辅助函数 (判决、BER计算、保存)
# ==========================================

def decision_making(symbols):
    """QPSK 硬判决"""
    real_part = np.real(symbols)
    imag_part = np.imag(symbols)
    bits = np.zeros((len(symbols), 2), dtype=int)
    # 映射规则: 00:(+,+), 01:(-,+), 11:(-,-), 10:(+,-)
    bits[(real_part > 0) & (imag_part > 0)] = (0, 0)
    bits[(real_part < 0) & (imag_part > 0)] = (0, 1)
    bits[(real_part < 0) & (imag_part < 0)] = (1, 1)
    bits[(real_part > 0) & (imag_part < 0)] = (1, 0)
    return bits

def calculate_ber(labels_true, labels_pred):
    labels_pred = labels_pred.astype(int)
    err_i = np.sum(labels_true[:, 0] != labels_pred[:, 0])
    err_q = np.sum(labels_true[:, 1] != labels_pred[:, 1])
    ber = (err_i + err_q) / (len(labels_true) * 2)
    print(f"BER: {ber:.6e}")
    return ber

def save_results_to_csv(snr_list, ber_list, save_path):
    """保存结果到 CSV"""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    try:
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['snr_db', 'ber'])
            for snr, ber in zip(snr_list, ber_list):
                writer.writerow([snr, f"{ber:.6e}"])
        print(f"✅ 结果已成功保存至 CSV: {save_path}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")

# ==========================================
# 2. IS2B 恢复逻辑 (使用 ResNet)
# ==========================================

def IS2B_restore_symbol_rectified_flow(snr_db_sample, is2b_instance, rx_clean, h_np, batch_size=256, guidance_scale=1.0):
    """
    执行 IS2B / Rectified Flow 恢复
    """
    device = is2b_instance.device
    n, c, L = rx_clean.shape

    # 1. 数据转 Tensor
    rx_clean_tensor = torch.from_numpy(rx_clean).float().to(device)
    
    # 2. 加噪
    y_all = add_awgn_noise_torch(rx_clean_tensor, snr_db_sample)
    
    # 处理 h (扩展维度)
    if h_np.ndim == 2:
        h_expanded = h_np[:, :, np.newaxis]
        h_expanded = np.repeat(h_expanded, L, axis=-1)
    else:
        h_expanded = h_np
    h_all = torch.from_numpy(h_expanded).float().to(device)

    recovered = []
    is2b_instance.model.eval()

    # 3. 批量恢复
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"SNR(sample)={snr_db_sample:.2f} dB"):
            end = min(start + batch_size, n)
            
            y_batch = y_all[start:end]
            h_batch = h_all[start:end]
            
            # === 核心调用：Rectified Flow 采样 ===
            # TimeResNet1D 支持传入 (x, t)
            x_rec = is2b_instance.sample_rectified_flow(
                y=y_batch,
                h=h_batch,
                guidance_scale=guidance_scale,
                stop_t=0.5  # <--- 新增参数，尝试 0.05, 0.1, 0.2
            )
            recovered.append(x_rec.cpu().numpy())

    recovered = np.concatenate(recovered, axis=0)
    
    # 4. 中点采样
    mid = L // 2
    sym_i = recovered[:, 0, mid]
    sym_q = recovered[:, 1, mid]
    symbols = sym_i + 1j * sym_q
    
    return symbols

def run_IS2B_chain(labels_iq, snr_db_sample, is2b_instance, rx_clean, h_np, batch_size=256, guidance_scale=1.0):
    symbols = IS2B_restore_symbol_rectified_flow(
        snr_db_sample=snr_db_sample,
        is2b_instance=is2b_instance,
        rx_clean=rx_clean,
        h_np=h_np,
        batch_size=batch_size,
        guidance_scale=guidance_scale
    )
    labels_pred = decision_making(symbols)
    return calculate_ber(labels_iq, labels_pred)

# ==========================================
# 3. 绘图函数
# ==========================================

def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(10, 6))
    snr_array = np.array(snr_range)
    
    plt.semilogy(snr_array, model_bers, 'o-', color='red', label='IS2B (TimeResNet1D)')
    
    if len(ref_bers) > 0:
        limit = min(len(snr_array), len(ref_bers))
        plt.semilogy(snr_array[:limit], ref_bers[:limit], 's--', color='blue', alpha=0.6, label='Baseline (MMSE)')

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('QPSK Performance: IS2B with TimeResNet1D')
    plt.legend()
    plt.ylim(5e-4, 1.0)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()

# ==========================================
# 4. 主程序
# ==========================================

if __name__ == "__main__":
    # ----- 配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 10   # 必须与训练时的设置一致
    batch_size = 4096 # 测试时可以大一点
    sps = 16 
    
    guidance_scale = 1 # 如果没用 CFG 训练，保持 1.0

    # === 路径配置 ===
    # 模型权重路径 (指向 train_resnet.py 生成的 best_model)
    ckpt_path = fr'IS2B/rIS2B_rayleigh_all_h_resnet_stop/results/best_model_IS2B_resnet.pth'
    
    # 结果保存路径
    result_save_path = f'IS2B/rIS2B_rayleigh_all_h_resnet_stop/ber_results/ber_curve_resnet.png'
    result_csv_path = f'IS2B/rIS2B_rayleigh_all_h_resnet_stop/ber_results/ber_data_resnet.csv'
    
    # 基准 BER 文件
    baseline_csv_path = 'IS2B/rIS2B_rayleigh_all_h_resnet_stop/ber_results/baseline_ber.csv'

    # ----- 1. 加载 TimeResNet1D 模型 -----
    print(f"Building TimeResNet1D on {device}...")
    # 参数必须与训练时完全一致！
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
        raise FileNotFoundError(f"❌ 未找到权重文件 {ckpt_path}")

    # 初始化 IS2B 包装器
    is2b_instance = IS2B(model, n_steps=n_steps, device=device)

    # ----- 2. 加载测试数据 -----
    print("正在加载数据...")
    # 测试集范围：400000 ~ 500000
    test_start, test_end = 400000, 420000
    test_data = QPSKDataset(test_start, test_end)
    
    rx_clean = test_data.y   # [N,2,L]
    h_np = test_data.z       # [N,2,L]
    n_win = rx_clean.shape[0]

    # ----- 3. 加载标签 -----
    label_path = r'F:\LJN\bishe\bishe\data\rayleigh_data_all_h\labels.npy'
    print(f"Loading Labels from {label_path}...")
    label_all = np.load(label_path)
    
    # 截取对应测试集的标签
    label_seg = label_all[test_start:test_end]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    labels_iq = np.array([map_label[int(v)] for v in label_seg], dtype=int)

    # ----- 4. 运行仿真 -----
    snr_range = np.arange(2, 19, 1) # 2 ~ 18 dB
    model_bers = []

    print(f"开始测试 (Rectified Flow + TimeResNet)...")
    for snr_db in snr_range:
        print(f"\n=== SNR (Symbol): {snr_db} dB ===")
        snr_db_sample = snr_db - 10 * math.log10(sps)

        ber = run_IS2B_chain(
            labels_iq=labels_iq,
            snr_db_sample=snr_db_sample,
            is2b_instance=is2b_instance,
            rx_clean=rx_clean,
            h_np=h_np,
            batch_size=batch_size,
            guidance_scale=guidance_scale
        )
        model_bers.append(ber)

    # ----- 5. 保存 CSV -----
    save_results_to_csv(snr_range, model_bers, result_csv_path)

    # ----- 6. 读取基准对比 -----
    ref_bers = []
    if os.path.exists(baseline_csv_path):
        try:
            with open(baseline_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                baseline_data = {float(row['snr_db']): float(row['baseline_ber']) for row in reader}
                for snr in snr_range:
                    ref_bers.append(baseline_data.get(snr, 0.0))
            print("基准数据加载成功。")
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print("未找到基准 CSV 文件。")

    # ----- 7. 绘图 -----
    plot_ber(model_bers, ref_bers, snr_range, result_save_path)
    print(f"Done. Saved Image to {result_save_path}")