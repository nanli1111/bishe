import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
from scipy.special import erfc  # 用于计算理论误码率

# ==========================================
# 导入自定义模块
# ==========================================
# 请根据实际文件路径调整 import
from model.net import SETimeResNet1D 
from ddpm_core import DDPM
from dataset.dataset import QPSKDataset

# 中文字体设置 (防止绘图乱码)
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# ==========================================
# 0. 辅助函数
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

def decision_making(symbols):
    """硬判决: 将复数符号映射为比特"""
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
    return ber

def get_theoretical_qpsk_ber(snr_db_array):
    """
    计算 QPSK 在 AWGN 信道下的理论 BER (格雷码)
    公式: BER = Q(sqrt(Es/N0)) = 0.5 * erfc(sqrt(Es/N0 / 2))
    参数: snr_db_array (SNR per symbol in dB)
    """
    snr_linear = 10 ** (np.array(snr_db_array) / 10.0)
    return 0.5 * erfc(np.sqrt(snr_linear / 2))

def find_matching_timestep(ddpm, snr_db, sps=16):
    """
    根据物理信道 SNR 寻找扩散过程中最匹配的时间步 t
    """
    # 物理 SNR (Sample level, 因为信号是过采样的)
    snr_db_sample = snr_db - 10 * math.log10(sps)
    target_snr = 10 ** (snr_db_sample / 10.0)
    
    # 扩散过程 SNR = alpha_bar / (1 - alpha_bar)
    alpha_bars = ddpm.alpha_bars.cpu().numpy()
    diff_snrs = alpha_bars / (1.0 - alpha_bars + 1e-8)
    
    # 找最近的 t
    diff = np.abs(diff_snrs - target_snr)
    best_t = np.argmin(diff)
    
    return int(best_t), alpha_bars[best_t]

# ==========================================
# 1. 核心恢复逻辑 (One-shot / Refinement)
# ==========================================
def ddpm_restore_one_shot(snr_db_symbol, ddpm_instance, rx_clean, batch_size=256, sps=16):
    """
    使用 DDPM 进行单步恢复 (Refinement)
    """
    device = ddpm_instance.device
    model = ddpm_instance.model
    n, c, L = rx_clean.shape

    # 1. 模拟物理信道 (添加噪声得到 y)
    snr_db_sample = snr_db_symbol - 10 * math.log10(sps)
    rx_noisy = add_awgn_noise_np(rx_clean, snr_db_sample)

    # 2. 转 Tensor
    y_all = torch.from_numpy(rx_noisy).float().to(device)
    
    # 3. 寻找对应的时间步 t
    t_idx, current_alpha_bar = find_matching_timestep(ddpm_instance, snr_db_symbol, sps)
    
    # 4. 幅度缩放: 近似认为 x_t ≈ y * sqrt(alpha_bar)
    scale_factor = math.sqrt(current_alpha_bar)
    x_t_all = y_all * scale_factor

    print(f"SNR={snr_db_symbol}dB -> Matched Step t={t_idx}/{ddpm_instance.n_steps}, Scale={scale_factor:.3f}")

    recovered = []
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc="Inferring"):
            end = min(start + batch_size, n)
            
            # 取出当前批次 [B, 2, L]
            x_t_batch = x_t_all[start:end]
            
            # 构造时间 tensor
            t_tensor = torch.full((x_t_batch.shape[0],), t_idx, device=device, dtype=torch.long)
            
            # === 模型预测 x0 ===
            x0_pred = model(x_t_batch, t_tensor)
            
            # 截断 (Clamping)
            x0_pred = torch.clamp(x0_pred, -2.0, 2.0)
            
            recovered.append(x0_pred.cpu().numpy())

    recovered = np.concatenate(recovered, axis=0)
    
    # 5. 中点采样 (Downsampling)
    mid = L // 2
    sym_i = recovered[:, 0, mid]
    sym_q = recovered[:, 1, mid]
    symbols = sym_i + 1j * sym_q
    
    return symbols

def run_test_chain(labels_iq, snr_db_symbol, ddpm_instance, rx_clean, batch_size=256, sps=16):
    # 执行恢复
    symbols = ddpm_restore_one_shot(
        snr_db_symbol=snr_db_symbol,
        ddpm_instance=ddpm_instance,
        rx_clean=rx_clean,
        batch_size=batch_size,
        sps=sps
    )
    # 判决 & 计算 BER
    labels_pred = decision_making(symbols)
    ber = calculate_ber(labels_iq, labels_pred)
    return ber

def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(10, 6))
    snr_array = np.array(snr_range)
    
    # 绘制模型曲线
    plt.semilogy(snr_array, model_bers, 'o-', color='red', label='Diffcusion Model (Proposed)')
    
    # 绘制理论曲线
    if len(ref_bers) > 0:
        plt.semilogy(snr_array, ref_bers, 'k--', linewidth=2, label='Theoretical QPSK (AWGN)')

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('SNR per symbol (dB)', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('BER Performance Comparison', fontsize=14)
    plt.legend(fontsize=12)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()

# ==========================================
# 2. 主函数
# ==========================================
if __name__ == "__main__":
    # ----- 配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 70       # 需与训练时保持一致
    batch_size = 2048   
    sps = 16            # 每个符号的采样点数
    
    # 路径配置
    result_dir = './results'
    ckpt_path = os.path.join(result_dir, 'best_model.pth') 
    result_save_path = os.path.join(result_dir, 'ber_curve_comparison.png')

    # ----- 1. 加载模型 -----
    print("Initializing Model...")
    # 确保参数与训练时 SETimeResNet1D 一致
    model = SETimeResNet1D(
        in_channels=2, 
        out_channels=2, 
        hidden_dim=128, 
        num_blocks=8, 
        time_emb_dim=128
    ).to(device)

    # 加载权重
    if os.path.exists(ckpt_path):
        print(f"Loading weights from: {ckpt_path}")
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
            print("Weights loaded successfully.")
        except Exception as e:
            print(f"⚠️ 权重加载失败: {e}")
            exit()
    else:
        print(f"⚠️ 警告: 未找到权重文件 {ckpt_path}，测试结果将无意义。")

    # 初始化 DDPM
    ddpm = DDPM(model, n_steps=n_steps, device=device)

    # ----- 2. 准备数据 -----
    print("Loading Test Data...")
    # 取一部分数据进行测试 (例如 20000 个符号)
    data = QPSKDataset(100000, 120000) 
    rx_clean = data.x  # [N, 2, L]
    
    # 生成 Ground Truth 标签
    # 假设 rx_clean 是纯净信号，直接取中心点做判决作为标签
    mid = rx_clean.shape[-1] // 2
    temp_sym = rx_clean[:, 0, mid] + 1j * rx_clean[:, 1, mid]
    labels_iq = decision_making(temp_sym) 

    print(f"Test Data Shape: {rx_clean.shape}")

    # ----- 3. 运行仿真 -----
    snr_range = np.arange(-5, 10, 1) # 测试 -5, -4, ..., 19 dB
    model_bers = []

    print(f"\nStarting BER Test...")
    print(f"{'SNR(dB)':<10} | {'Model BER':<15}")
    print("-" * 30)

    for snr_db in snr_range:
        ber = run_test_chain(
            labels_iq=labels_iq,
            snr_db_symbol=snr_db,
            ddpm_instance=ddpm,
            rx_clean=rx_clean,
            batch_size=batch_size,
            sps=sps
        )
        model_bers.append(ber)
        print(f"{snr_db:<10.1f} | {ber:<15.6e}")

    # ----- 4. 计算并打印理论基准 -----
    ref_bers = get_theoretical_qpsk_ber(snr_range)
    
    print("\n=== Performance Comparison ===")
    print(f"{'SNR(dB)':<10} | {'Model BER':<15} | {'Theory BER':<15}")
    print("-" * 45)
    for i, snr in enumerate(snr_range):
        print(f"{snr:<10.1f} | {model_bers[i]:<15.6e} | {ref_bers[i]:<15.6e}")

    # ----- 5. 绘图保存 -----
    plot_ber(model_bers, ref_bers, snr_range, result_save_path)
    print(f"✅ Test Finished. Results saved to {result_save_path}")