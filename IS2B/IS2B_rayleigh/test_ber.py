import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv

from model.unet import build_network
from IS2B_core import IS2B
from dataset.dataset import QPSKDataset
from test_fig import add_awgn_noise_np

# 中文字体
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False


# ===== 基础工具 =====
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


# ===== 处理逻辑 =====
def IS2B_restore_symbol(snr_db_sample, IS2B, rx_clean, h_np, batch_size=256, guidance_scale=1.0):
    device = IS2B.device
    n, c, L = rx_clean.shape

    # 1. 加噪
    rx_noisy = add_awgn_noise_np(rx_clean, snr_db_sample)

    # 2. 转 Tensor
    y_all = torch.from_numpy(rx_noisy).float().to(device)
    
    # 处理 h
    if h_np.ndim == 2:
        h_expanded = h_np[:, :, np.newaxis]
        h_expanded = np.repeat(h_expanded, L, axis=-1)
    else:
        h_expanded = h_np
    h_all = torch.from_numpy(h_expanded).float().to(device)

    recovered = []
    
    # 确保模型处于 eval 模式
    IS2B.model.eval()

    # 3. 批量恢复
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size),
                          desc=f"SNR(sample)={snr_db_sample:.2f} dB"):
            end = min(start + batch_size, n)
            
            y_batch = y_all[start:end]
            h_batch = h_all[start:end]
            
            # === 调用 IS2B 类中的方法 ===
            x_rec = IS2B.sample_pure_conditional(
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


def run_IS2B_chain(labels_iq, snr_db_sample, IS2B, rx_clean, h_np, batch_size=256, guidance_scale=1.0):
    symbols = IS2B_restore_symbol(
        snr_db_sample=snr_db_sample,
        IS2B=IS2B,
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
    
    plt.semilogy(snr_array, model_bers, 'o-', label='IS2B 恢复')
    
    if len(ref_bers) > 0:
        if len(ref_bers) == len(model_bers):
            plt.semilogy(snr_array, ref_bers, 's--', label='Baseline (MMSE)')
        else:
            limit = min(len(snr_array), len(ref_bers))
            plt.semilogy(snr_array[:limit], ref_bers[:limit], 's--', label='Baseline (MMSE)')

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('QPSK 在 nakagmi+AWGN 下：IS2B 性能')
    plt.legend()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


if __name__ == "__main__":
    # ----- 配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 100         # 必须与训练一致
    batch_size = 4096
    sps = 16
    
    # 关键参数：Guidance Scale (w)
    # w=0.5 推荐用于获得光滑波形
    guidance_scale = 0.5 

    # 路径
    ckpt_path = fr'IS2B/IS2B_nakagmi/results/best_model_epoch_with_n_steps{n_steps}.pth'
    baseline_csv_path = 'CDDM/cddm_nakagmi/ber_result/baseline_ber.csv'
    result_save_path = f'IS2B/IS2B_nakagmi/ber_result/ber_curve_nsteps{n_steps}_cfg{guidance_scale}.png'

    # ----- 1. 加载模型 (适配训练配置) -----
    net_cfg = {
        'type': 'UNet',
        'channels': [32, 64, 128, 256], 
        'pe_dim': 128,
        'in_channels': 6,               # 6通道输入
        'out_channels': 2
    }
    model = build_network(net_cfg, n_steps).to(device)

    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}")

    IS2B = IS2B(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # ----- 2. 数据 -----
    data = QPSKDataset(400000, 500000)
    rx_clean = data.y   # [N,2,L] 仅瑞利衰落
    h_np = data.z       # [N,2]
    n_win = rx_clean.shape[0]

    # ----- 3. 标签 -----
    label_path = r'F:\LJN\bishe\bishe\data\nakagmi_data\labels.npy'
    label_all = np.load(label_path)
    label_seg = label_all[400000:400000 + n_win]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    labels_iq = np.array([map_label[int(v)] for v in label_seg], dtype=int)

    # ----- 4. 运行仿真 -----
    snr_range = np.arange(2, 18, 1)
    model_bers = []

    print(f"开始测试 (CFG Scale = {guidance_scale})...")
    for snr_db in snr_range:
        print(f"\n=== SNR (Symbol): {snr_db} dB ===")
        snr_db_sample = snr_db - 10 * math.log10(sps)

        ber = run_IS2B_chain(
            labels_iq=labels_iq,
            snr_db_sample=snr_db_sample,
            IS2B=IS2B,
            rx_clean=rx_clean,
            h_np=h_np,
            batch_size=batch_size,
            guidance_scale=guidance_scale
        )
        model_bers.append(ber)

    # ----- 5. 基准对比 -----
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

    # ----- 6. 绘图 -----
    plot_ber(model_bers, ref_bers, snr_range, result_save_path)
    print(f"Done. Saved to {result_save_path}")