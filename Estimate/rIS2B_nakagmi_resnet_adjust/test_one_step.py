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
from model.resnet_pro import DilatedTimeResNet1D
# 2. 数据集与工具
from dataset.dataset import QPSKDataset
from test_fig_x_pre import add_awgn_noise_torch

# 中文字体设置
rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
rcParams['axes.unicode_minus'] = False

# ==========================================
# 1. 辅助函数
# ==========================================

def decision_making(symbols):
    """QPSK 硬判决"""
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

def save_results_to_csv(snr_list, ber_list, save_path):
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

def plot_ber(model_bers, ref_bers, snr_range, save_path):
    plt.figure(figsize=(10, 6))
    snr_array = np.array(snr_range)
    
    # 绘制当前模型曲线
    plt.semilogy(snr_array, model_bers, 'o-', color='red', label='TimeResNet1D (One-Step Pred)')
    
    # 绘制基准
    if len(ref_bers) > 0:
        limit = min(len(snr_array), len(ref_bers))
        plt.semilogy(snr_array[:limit], ref_bers[:limit], 's--', color='blue', alpha=0.6, label='Baseline (MMSE)')

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('IS2B One-Step Prediction Performance (ResNet)')
    plt.legend()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"BER 曲线已保存至: {save_path}")
    plt.close()

# ==========================================
# 2. 核心：ResNet 一步预测函数
# ==========================================

def predict_one_step_resnet(model, rx_clean, h_np, snr_db_sample, n_steps, device='cuda', batch_size=4096):
    """
    使用 TimeResNet1D 进行一步预测。
    输入：y (含噪信号) + t=1 (对应 index = n_steps-1)
    输出：预测的 x0
    """
    model.eval()
    n, c, L = rx_clean.shape

    # 1. 准备数据
    rx_clean_tensor = torch.from_numpy(rx_clean).float().to(device)
    
    # 2. 生成含噪接收信号 y
    y_all = add_awgn_noise_torch(rx_clean_tensor, snr_db_sample)
    
    # 3. 处理 h
    if h_np.ndim == 2:
        h_expanded = h_np[:, :, np.newaxis]
        h_expanded = np.repeat(h_expanded, L, axis=-1)
    else:
        h_expanded = h_np
    h_all = torch.from_numpy(h_expanded).float().to(device)

    recovered_symbols = []

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"One-Step | SNR={snr_db_sample:.1f}dB", leave=False):
            end = min(start + batch_size, n)
            
            # Batch 数据
            y_batch = y_all[start:end]
            h_batch = h_all[start:end]
            current_batch_size = y_batch.shape[0]

            # === 关键步骤 ===
            # 1. 构造时间索引：设为最大值 (n_steps - 1)
            # 代表输入完全是噪声/衰落状态 (t=1.0)
            t_idx = torch.full((current_batch_size,), n_steps - 1, device=device, dtype=torch.long)

            # 2. 构造网络输入 [y, h]
            net_input = torch.cat([y_batch, h_batch], dim=1)

            # 3. 直接预测 x0
            # TimeResNet1D 接受 (x, t) 作为输入
            pred_x0 = model(net_input, t_idx)
            
            # 4. 提取数据转 Numpy
            pred_x0_np = pred_x0.cpu().numpy()
            
            # 5. 取中间点符号 (Center Sampling)
            mid = L // 2
            sym_complex = pred_x0_np[:, 0, mid] + 1j * pred_x0_np[:, 1, mid]
            recovered_symbols.append(sym_complex)

    return np.concatenate(recovered_symbols, axis=0)

# ==========================================
# 3. 主程序
# ==========================================

if __name__ == "__main__":
    # ----- 配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 20   # 必须与训练时一致
    batch_size = 4096
    sps = 16 
    
    # === 路径配置 ===
    # 指向 TimeResNet 训练出的最佳模型
    ckpt_path = fr'Estimate/rIS2B_nakagmi_resnet_combine/results/best_model_IS2B_resnet_{n_steps}.pth'
    
    # 结果保存
    result_img_path = f'Estimate/rIS2B_nakagmi_resnet_combine/ber_results/ber_curve_onestep_resnet.png'
    result_csv_path = f'Estimate/rIS2B_nakagmi_resnet_combine/ber_results/ber_data_onestep_resnet.csv'
    
    # 基准 BER 文件 (可选)
    baseline_csv_path = 'Estimate/rIS2B_nakagmi_resnet_combine/ber_results/baseline_ber.csv'

    # ----- 1. 加载 TimeResNet1D 模型 -----
    print(f"Building TimeResNet1D on {device}...")
    # 参数必须与 train_resnet.py 中定义的完全一致
    model = DilatedTimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=128,   # 宽度
        num_blocks=12,    # 深度可以加深，例如 12 层
        time_emb_dim=128
    ).to(device)

    if os.path.exists(ckpt_path):
        print(f"加载模型权重: {ckpt_path}")
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
    else:
        raise FileNotFoundError(f"权重文件未找到: {ckpt_path}")

    # ----- 2. 加载数据 -----
    print("Loading Test Data...")
    test_start, test_end = 400000, 500000
    test_data = QPSKDataset(test_start, test_end) 
    rx_clean = test_data.y   # [N,2,L]
    h_np = test_data.z       # [N,2,L]
    
    # 加载标签
    label_path = r'F:\LJN\bishe\bishe\data\nakagmi_data\labels.npy'
    label_all = np.load(label_path)
    label_seg = label_all[test_start:test_end]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    labels_iq = np.array([map_label[int(v)] for v in label_seg], dtype=int)

    # ----- 3. 运行测试 -----
    snr_range = np.arange(0, 19, 1) # 2dB 到 18dB
    model_bers = []

    print(f"开始一步预测 BER 测试 (ResNet)...")
    
    for snr_db in snr_range:
        # 换算 SNR
        snr_db_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)

        # 运行一步预测
        symbols_pred = predict_one_step_resnet(
            model=model,
            rx_clean=rx_clean,
            h_np=h_np,
            snr_db_sample=snr_db_sample,
            n_steps=n_steps,
            device=device,
            batch_size=batch_size
        )
        
        # 判决与计算
        labels_pred = decision_making(symbols_pred)
        ber = calculate_ber(labels_iq, labels_pred)
        
        print(f"SNR(sym)={snr_db}dB | BER={ber:.6e}")
        model_bers.append(ber)

    # ----- 4. 保存 CSV 结果 -----
    save_results_to_csv(snr_range, model_bers, result_csv_path)

    # ----- 5. 读取基准 (如果有) 并绘图 -----
    ref_bers = []
    if os.path.exists(baseline_csv_path):
        try:
            with open(baseline_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                baseline_data = {float(row['snr_db']): float(row['baseline_ber']) for row in reader}
                for snr in snr_range:
                    ref_bers.append(baseline_data.get(snr, 1.0))
        except:
            pass

    plot_ber(model_bers, ref_bers, snr_range, result_img_path)