import os
import math
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import csv

# === 引入项目模块 ===
from model.resnet_pro import DilatedTimeResNet1D  # 确保这里导入的是你的模型类
from IS2B_x_pre import IS2B
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

def IS2B_restore_symbol_hybrid(model, device, n_steps, snr_db_sample, rx_clean, h_np, batch_size=256, guidance_scale=1.0):
    """
    执行 IS2B 混合恢复：One-Step Anchor + Rectified Flow Refinement
    """
    n, c, L = rx_clean.shape
    
    # 临时构建 IS2B 实例
    is2b_instance = IS2B(model, n_steps=n_steps, device=device)

    rx_clean_tensor = torch.from_numpy(rx_clean).float().to(device)
    y_all = add_awgn_noise_torch(rx_clean_tensor, snr_db_sample)
    
    if h_np.ndim == 2:
        h_expanded = h_np[:, :, np.newaxis]
        h_expanded = np.repeat(h_expanded, L, axis=-1)
    else:
        h_expanded = h_np
    h_all = torch.from_numpy(h_expanded).float().to(device)

    recovered = []
    model.eval()

    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"SNR={snr_db_sample:.1f}dB"):
            end = min(start + batch_size, n)
            
            y_batch = y_all[start:end]
            h_batch = h_all[start:end]
            current_batch_size = y_batch.shape[0]
            
            # Step A: Anchor (One-Step)
            net_input_os = torch.cat([y_batch, h_batch], dim=1)
            t_max = torch.full((current_batch_size,), n_steps - 1, device=device, dtype=torch.long)
            anchor_x = model(net_input_os, t_max)
            
            # Step B: Rectified Flow (Hybrid)
            x_rec = is2b_instance.sample(
                y=y_batch,
                h=h_batch,
                guidance_scale=guidance_scale,
                stop_t=0.0,      # 保持截断策略
                anchor=anchor_x  # 传入 Anchor
            )
            recovered.append(x_rec.cpu().numpy())

    recovered = np.concatenate(recovered, axis=0)
    mid = L // 2
    sym_i = recovered[:, 0, mid]
    sym_q = recovered[:, 1, mid]
    symbols = sym_i + 1j * sym_q
    
    return symbols

# ==========================================
# 2. 评估核心函数 (动态模型结构)
# ==========================================
def evaluate_single_model(name, ckpt_path, hidden_dim, num_blocks, device, test_data_tuple, snr_range, sps, batch_size):
    """
    评估单个模型的性能，支持动态传入 hidden_dim 和 num_blocks
    """
    print(f"\n🚀 Evaluating Model: {name}")
    print(f"   Path: {ckpt_path}")
    print(f"   Config: hidden_dim={hidden_dim}, num_blocks={num_blocks}")
    
    # 1. 动态实例化模型
    # time_emb_dim 通常与 hidden_dim 一致或固定，这里假设与 hidden_dim 一致以保持灵活性
    model = DilatedTimeResNet1D(
        in_channels=4, 
        out_channels=2, 
        hidden_dim=hidden_dim,   
        num_blocks=num_blocks,    
        time_emb_dim=hidden_dim # 或者固定为 128，看你训练时的设置
    ).to(device)

    if os.path.exists(ckpt_path):
        try:
            model.load_state_dict(torch.load(ckpt_path, map_location=device))
        except RuntimeError as e:
            print(f"❌ Error loading state dict: {e}")
            print("   (可能是模型结构参数不匹配，请检查 hidden_dim 和 num_blocks)")
            return None
    else:
        print(f"❌ Error: Checkpoint not found: {ckpt_path}")
        return None

    # 解包数据
    rx_clean, h_np, labels_iq = test_data_tuple
    
    # 2. 运行测试
    bers = []
    n_steps = 20 # 假设训练时都是20步
    
    for snr_db in snr_range:
        snr_db_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)
        
        symbols = IS2B_restore_symbol_hybrid(
            model=model,
            device=device,
            n_steps=n_steps,
            snr_db_sample=snr_db_sample,
            rx_clean=rx_clean,
            h_np=h_np,
            batch_size=batch_size,
            guidance_scale=1.0
        )
        
        labels_pred = decision_making(symbols)
        ber = calculate_ber(labels_iq, labels_pred)
        bers.append(ber)
        print(f"   SNR={snr_db}dB | BER={ber:.6e}") # 如果不想刷屏可以注释掉
        
    return bers

# ==========================================
# 3. 绘图函数
# ==========================================
def plot_multi_model_ber(results_dict, ref_resnet, ref_baseline, snr_range, save_path):
    plt.figure(figsize=(12, 8))
    snr_array = np.array(snr_range)
    
    # 1. 绘制各个测试模型的曲线
    markers = ['o-', 's-', 'D-', '^-', 'v-', 'x-', '*-']
    for i, (name, bers) in enumerate(results_dict.items()):
        marker = markers[i % len(markers)]
        plt.semilogy(snr_array, bers, marker, linewidth=2, label=name)
    
    # 2. 绘制参考线 (ResNet)
    if len(ref_resnet) > 0:
        limit = min(len(snr_array), len(ref_resnet))
        plt.semilogy(snr_array[:limit], ref_resnet[:limit], '--', color='gray', alpha=0.6, label='Ref: ResNet (Base)')

    # 3. 绘制参考线 (Baseline MMSE)
    if len(ref_baseline) > 0:
        limit = min(len(snr_array), len(ref_baseline))
        plt.semilogy(snr_array[:limit], ref_baseline[:limit], '-.', color='black', alpha=0.6, label='Ref: MMSE')

    plt.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('Multi-Model BER Comparison (Hybrid Decoding)')
    plt.legend()
    plt.ylim(1e-6, 1.0) 
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f"✅ 绘图完成: {save_path}")
    plt.close()

# ==========================================
# 4. 主程序
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 4096 
    sps = 16 
    
    # === 配置：模型列表 ===
    # 格式：(图例名称, 权重路径, hidden_dim, num_blocks)
    models_to_test = [
        # 模型 1: 原版 (128, 12)
        ("Base (128-12)", 
         r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\results\best_model_IS2B_resnet_pro_20.pth", 
         128, 12),
         
        # 模型 2
        ("L1 Loss (0.5)", 
         r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\results\best_model_IS2B_resnet_pro_0.5L1_20.pth", 
         128, 12),
         
        # 模型 3
        ("L1 Loss (0.7)", 
         r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\results\best_model_IS2B_resnet_pro_0.7L1_20.pth", 
         128, 12),

        # 模型 4
        ("crum Loss", 
          r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\results\best_model_IS2B_resnet_pro_crum_20.pth",
          128, 12),
        # 模型 5
        ("Scope Loss", 
         r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\results\best_model_IS2B_resnet_pro_scope_20.pth",
          128, 12),
          
        # 模型 6
        ("Cosine LR",
            r"F:\LJN\bishe\bishe\IS2B\rIS2B_nakagmi_resnet_adjust\results\best_model_IS2B_resnet_cos.pth",
            256, 8),

    ]
    
    # 结果保存路径
    save_dir = r'IS2B/rIS2B_nakagmi_resnet_adjust/comparison_results'
    baseline_csv_path = 'IS2B/rIS2B_nakagmi_resnet_adjust/ber_results/ber_curve_resnet_values.csv'

    # === 1. 统一加载数据 (只加载一次) ===
    print("Loading Test Data...")
    test_start, test_end = 400000, 500000
    test_data = QPSKDataset(test_start, test_end)
    rx_clean = test_data.y   
    h_np = test_data.z       
    
    label_path = r'F:\LJN\bishe\bishe\data\nakagmi_data\labels.npy'
    label_all = np.load(label_path)
    label_seg = label_all[test_start:test_end]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    labels_iq = np.array([map_label[int(v)] for v in label_seg], dtype=int)
    
    data_tuple = (rx_clean, h_np, labels_iq)

    # === 2. 批量测试 ===
    snr_range = np.arange(0, 19, 1)
    results_dict = {} 
    
    # 解包由4个元素组成的元组
    for name, path, h_dim, n_blk in models_to_test:
        bers = evaluate_single_model(
            name=name, 
            ckpt_path=path, 
            hidden_dim=h_dim,   # 传入特定的宽度
            num_blocks=n_blk,   # 传入特定的深度
            device=device, 
            test_data_tuple=data_tuple, 
            snr_range=snr_range, 
            sps=sps, 
            batch_size=batch_size
        )
        if bers is not None:
            results_dict[name] = bers

    # === 3. 读取参考数据 ===
    ref_resnet_bers = []
    ref_baseline_bers = []
    if os.path.exists(baseline_csv_path):
        try:
            with open(baseline_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                csv_resnet_map = {}
                csv_baseline_map = {}
                for row in reader:
                    try:
                        s = round(float(row['snr_db']), 1)
                        csv_resnet_map[s] = float(row['resnet_ber'])
                        csv_baseline_map[s] = float(row['baseline_ber'])
                    except ValueError: continue

                for snr in snr_range:
                    k = round(float(snr), 1)
                    if k in csv_resnet_map: ref_resnet_bers.append(csv_resnet_map[k])
                    if k in csv_baseline_map: ref_baseline_bers.append(csv_baseline_map[k])
        except Exception as e:
            print(f"Error reading baseline: {e}")

    # === 4. 绘图与保存 ===
    plot_path = os.path.join(save_dir, 'multi_model_arch_comparison.png')
    plot_multi_model_ber(results_dict, ref_resnet_bers, ref_baseline_bers, snr_range, plot_path)
    
    csv_save_path = os.path.join(save_dir, 'multi_model_arch_data.csv')
    with open(csv_save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        header = ['snr_db'] + list(results_dict.keys())
        writer.writerow(header)
        for i, snr in enumerate(snr_range):
            row = [snr]
            for name in results_dict.keys():
                row.append(f"{results_dict[name][i]:.6e}")
            writer.writerow(row)
    print(f"✅ 数据汇总 CSV 已保存: {csv_save_path}") 