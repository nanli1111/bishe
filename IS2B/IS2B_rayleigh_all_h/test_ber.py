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
from test_fig_x_pre import add_awgn_noise_np

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

def find_matching_timestep(schedule_class, snr_db, sps=16):
    """
    根据物理 SNR 寻找对应的扩散时间步 t (复用 test_fig 中的逻辑)
    """
    snr_db_sample = snr_db - 10 * math.log10(sps)
    target_snr = 10 ** (snr_db_sample / 10.0)
    
    if hasattr(schedule_class, 'alpha_bars'):
        alpha_bars = schedule_class.alpha_bars.cpu().numpy()
    else:
        raise AttributeError("Schedule class missing alpha_bars")
    
    # SNR = alpha / (1 - alpha)
    diff_snrs = alpha_bars / (1 - alpha_bars + 1e-8)
    diff = np.abs(diff_snrs - target_snr)
    best_t = np.argmin(diff)
    
    return int(best_t), alpha_bars[best_t]


# ===== 核心恢复逻辑 =====
def IS2B_restore_symbol_refinement(snr_db_symbol, is2b_instance, rx_clean, h_np, batch_size=256, sps=16):
    """
    执行 IS2B 单步精修恢复 (One-shot Refinement)
    逻辑与 test_fig.py 完全一致
    """
    device = is2b_instance.device
    model = is2b_instance.model
    n, c, L = rx_clean.shape

    # 1. 准备含噪接收信号 y (物理世界中的观测值)
    snr_db_sample = snr_db_symbol - 10 * math.log10(sps)
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

    # 3. === 关键：寻找对应的时间步 t ===
    # 我们假设这个含噪信号就是训练过程中某个时刻 t 的状态
    t_idx, current_alpha_bar = find_matching_timestep(is2b_instance, snr_db_symbol, sps)
    
    # 4. === 关键：幅度缩放 ===
    # 扩散公式: x_t = sqrt(alpha_bar) * x0 + ...
    # 物理信号: y = x0 + noise
    # 为了让网络认为这是 x_t，我们需要把 y 乘以 sqrt(alpha_bar)
    scale_factor = math.sqrt(current_alpha_bar)
    
    # 准备 x_t 输入 (全量数据缩放)
    x_t_all = y_all * scale_factor

    print(f"SNR={snr_db_symbol}dB -> Matched Step t={t_idx}/{is2b_instance.n_steps}, Scale={scale_factor:.3f}")

    recovered = []
    model.eval()

    # 5. 批量恢复
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inferring"):
            end = min(start + batch_size, n)
            
            # 取出当前批次
            x_t_batch = x_t_all[start:end]
            h_batch = h_all[start:end]
            
            # 构造时间 tensor
            t_tensor = torch.full((x_t_batch.shape[0],), t_idx, device=device, dtype=torch.long)
            
            # 构造网络输入 [x_t, h]
            net_input = torch.cat([x_t_batch, h_batch], dim=1)
            
            # 单步预测 x0
            x_rec = model(net_input, t_tensor)
            
            # 截断 (Clamping)
            x_rec = torch.clamp(x_rec, -2.0, 2.0)
            
            recovered.append(x_rec.cpu().numpy())

    recovered = np.concatenate(recovered, axis=0)
    
    # 6. 中点采样
    mid = L // 2
    sym_i = recovered[:, 0, mid]
    sym_q = recovered[:, 1, mid]
    symbols = sym_i + 1j * sym_q
    
    return symbols


def run_IS2B_chain(labels_iq, snr_db_symbol, is2b_instance, rx_clean, h_np, batch_size=256, sps=16):
    symbols = IS2B_restore_symbol_refinement(
        snr_db_symbol=snr_db_symbol,
        is2b_instance=is2b_instance,
        rx_clean=rx_clean,
        h_np=h_np,
        batch_size=batch_size,
        sps=sps
    )
    labels_pred = decision_making(symbols)
    return calculate_ber(labels_iq, labels_pred)


def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(10, 6))
    snr_array = np.array(snr_range)
    
    plt.semilogy(snr_array, model_bers, 'o-', label='IS2B (Refinement)')
    
    if len(ref_bers) > 0:
        if len(ref_bers) == len(model_bers):
            plt.semilogy(snr_array, ref_bers, 's--', label='Baseline (MMSE)')
        else:
            limit = min(len(snr_array), len(ref_bers))
            plt.semilogy(snr_array[:limit], ref_bers[:limit], 's--', label='Baseline (MMSE)')

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('QPSK 在 Rayleigh+AWGN 下：IS2B (Refinement) 性能')
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
    
    # 路径
    ckpt_path = fr'IS2B/IS2B_rayleigh_all_h/results/best_model_epoch_with_n_step{n_steps}_x0.pth'
    baseline_csv_path = 'IS2B/IS2B_rayleigh_all_h/ber_result/baseline_ber.csv'
    result_save_path = f'IS2B/IS2B_rayleigh_all_h/ber_result/ber_curve_nsteps{n_steps}_refinement.png'

    # ----- 1. 加载模型 -----
    # 训练时输入是 [x_t, h]，共4通道
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

    # 初始化 IS2B 实例
    is2b_instance = IS2B(model, n_steps=n_steps, min_beta=1e-4, max_beta=0.02, device=device)

    # ----- 2. 数据 -----
    data = QPSKDataset(400000, 500000)
    rx_clean = data.y   # [N,2,L]
    h_np = data.z       # [N,2]
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

    print(f"开始测试 (Refinement Mode)...")
    for snr_db in snr_range:
        ber = run_IS2B_chain(
            labels_iq=labels_iq,
            snr_db_symbol=snr_db,
            is2b_instance=is2b_instance,
            rx_clean=rx_clean,
            h_np=h_np,
            batch_size=batch_size,
            sps=sps
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