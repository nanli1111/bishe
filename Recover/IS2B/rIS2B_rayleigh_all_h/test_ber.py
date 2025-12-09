import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv

from model.unet import build_network
from IS2B_x_pre import IS2B
from dataset.dataset import QPSKDataset
from test_fig_x_pre import add_awgn_noise_torch

# 中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


def decision_making(symbols):
    real_part = np.real(symbols)
    imag_part = np.imag(symbols)
    bits = np.zeros((len(symbols), 2), dtype=int)
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

# ===== 新增：保存 CSV 函数 =====
def save_results_to_csv(snr_list, ber_list, save_path):
    """
    将测试结果保存为 CSV 文件
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        with open(save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(['snr_db', 'ber'])
            # 写入数据
            for snr, ber in zip(snr_list, ber_list):
                writer.writerow([snr, f"{ber:.6e}"])
        print(f"✅ 结果已成功保存至 CSV: {save_path}")
    except Exception as e:
        print(f"❌ 保存 CSV 失败: {e}")


# ===== 真正的 IS2B 恢复逻辑 =====
def IS2B_restore_symbol_rectified_flow(snr_db_sample, is2b_instance, rx_clean, h_np, batch_size=256, guidance_scale=1.0):
    """
    执行真正的 IS2B / Rectified Flow 恢复
    逻辑：起点 x_T = y (含噪衰落信号)，求解 ODE 走到 x_0
    """
    device = is2b_instance.device
    n, c, L = rx_clean.shape

    # 1. 数据转 Tensor
    rx_clean_tensor = torch.from_numpy(rx_clean).float().to(device)
    
    # 2. 加噪 (使用 Torch 版本)
    y_all = add_awgn_noise_torch(rx_clean_tensor, snr_db_sample)
    
    # 处理 h
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
        # 不需要计算 t，直接对每个 batch 进行全流程采样
        for start in tqdm(range(0, n, batch_size),
                          desc=f"SNR(sample)={snr_db_sample:.2f} dB"):
            end = min(start + batch_size, n)
            
            y_batch = y_all[start:end]
            h_batch = h_all[start:end]
            
            # === 核心调用：Rectified Flow 采样 ===
            # y_batch 就是起点 (t=1)
            x_rec = is2b_instance.sample_rectified_flow(
                y=y_batch,
                h=h_batch,
                guidance_scale=guidance_scale
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


def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(10, 6))
    snr_array = np.array(snr_range)
    
    plt.semilogy(snr_array, model_bers, 'o-', label='IS2B (Rectified Flow)')
    
    if len(ref_bers) > 0:
        if len(ref_bers) == len(model_bers):
            plt.semilogy(snr_array, ref_bers, 's--', label='Baseline (MMSE)')
        else:
            limit = min(len(snr_array), len(ref_bers))
            plt.semilogy(snr_array[:limit], ref_bers[:limit], 's--', label='Baseline (MMSE)')

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('QPSK 在 Rayleigh+AWGN 下：IS2B 性能')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # ----- 配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100
    batch_size = 4096
    sps = 16 
    
    guidance_scale = 1.0

    # 路径
    ckpt_path = fr'IS2B/rIS2B_rayleigh_all_h/results/best_model_IS2B_n{n_steps}.pth'
    baseline_csv_path = 'IS2B/rIS2B_rayleigh_all_h/ber_result/baseline_ber.csv'
    
    result_save_path = f'IS2B/rIS2B_rayleigh_all_h/ber_result/ber_curve_nsteps{n_steps}_real_rf.png'
    # 新增 CSV 路径
    result_csv_path = f'IS2B/rIS2B_rayleigh_all_h/ber_result/ber_data_nsteps{n_steps}_real_rf.csv'

    # ----- 1. 加载模型 (in_channels=4) -----
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
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}")

    # 初始化 IS2B
    is2b_instance = IS2B(model, n_steps=n_steps, device=device)

    # ----- 2. 数据 -----
    print("正在加载数据...")
    # 使用包含全量 h 的数据集
    test_data = QPSKDataset(400000, 500000)
    rx_clean = test_data.y   # [N,2,L]
    h_np = test_data.z       # [N,2,L] 或 [N,2]
    n_win = rx_clean.shape[0]

    # ----- 3. 标签 -----
    label_path = r'F:\LJN\bishe\bishe\data\rayleigh_data_all_h\labels.npy'
    label_all = np.load(label_path)
    label_seg = label_all[400000:400000 + n_win]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    labels_iq = np.array([map_label[int(v)] for v in label_seg], dtype=int)

    # ----- 4. 运行仿真 -----
    snr_range = np.arange(2, 18, 1)
    model_bers = []

    print(f"开始测试 (Rectified Flow Mode)...")
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

    # ----- 5. 保存数据到 CSV (新增步骤) -----
    save_results_to_csv(snr_range, model_bers, result_csv_path)

    # ----- 6. 基准对比 -----
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

    # ----- 7. 绘图 -----
    plot_ber(model_bers, ref_bers, snr_range, result_save_path)
    print(f"Done. Saved Image to {result_save_path}")