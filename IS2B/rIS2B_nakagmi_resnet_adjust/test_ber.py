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
# 2. IS2B 恢复逻辑 (混合解码策略：OneStep + RF)
# ==========================================

def IS2B_restore_symbol_hybrid(snr_db_sample, is2b_instance, rx_clean, h_np, batch_size=256, guidance_scale=1.0):
    """
    执行 IS2B 混合恢复：One-Step Anchor + Rectified Flow Refinement
    """
    device = is2b_instance.device
    model = is2b_instance.model
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
    model.eval()

    # 3. 批量恢复
    with torch.no_grad():
        for start in tqdm(range(0, n, batch_size), desc=f"SNR(sample)={snr_db_sample:.2f} dB"):
            end = min(start + batch_size, n)
            
            y_batch = y_all[start:end]
            h_batch = h_all[start:end]
            current_batch_size = y_batch.shape[0]
            
            # === Step A: 生成 Anchor (One-Step Prediction) ===
            # 构造输入 [y, h]
            net_input_os = torch.cat([y_batch, h_batch], dim=1)
            # t = 1.0 (最大步数索引)
            t_max = torch.full((current_batch_size,), is2b_instance.n_steps - 1, device=device, dtype=torch.long)
            
            # 得到 One-Step 预测结果 (作为锚点)
            anchor_x = model(net_input_os, t_max)
            
            # === Step B: 执行 Rectified Flow 采样 (带 Anchor 约束) ===
            x_rec = is2b_instance.sample(
                y=y_batch,
                h=h_batch,
                guidance_scale=guidance_scale,
                stop_t=0.0,        # 建议配合截断，如 0.1
                anchor=anchor_x    # <--- 关键修改：传入 Anchor
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
    symbols = IS2B_restore_symbol_hybrid(
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

def plot_ber_compare(model_bers, ref_resnet, ref_baseline, snr_range, save_path):
    plt.figure(figsize=(10, 6))
    snr_array = np.array(snr_range)
    
    # 1. 绘制当前模型的曲线 (红色实线)
    plt.semilogy(snr_array, model_bers, 'o-', color='red', linewidth=2, label='Current Model (IS2B)')
    
    # 2. 绘制 CSV 中的 ResNet 参考线 (蓝色虚线)
    if len(ref_resnet) > 0:
        # 截断以防止长度不一致
        limit = min(len(snr_array), len(ref_resnet))
        plt.semilogy(snr_array[:limit], ref_resnet[:limit], 's--', color='blue', alpha=0.7, label='Reference: ResNet')

    # 3. 绘制 CSV 中的 Baseline 参考线 (绿色点划线)
    if len(ref_baseline) > 0:
        limit = min(len(snr_array), len(ref_baseline))
        plt.semilogy(snr_array[:limit], ref_baseline[:limit], '^-.', color='green', alpha=0.7, label='Reference: Baseline (MMSE)')

    plt.grid(True, which='both', linestyle='--', alpha=0.7)
    plt.xlabel('SNR per symbol (dB)')
    plt.ylabel('BER')
    plt.title('BER Performance Comparison')
    plt.legend()
 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
# ==========================================
# 4. 主程序
# ==========================================

if __name__ == "__main__":
    # ----- 配置 -----
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_steps = 20   # 必须与训练时的设置一致
    batch_size = 4096 
    sps = 16 
    
    # 即使之前没训 CFG，这里的 guidance_scale 设大一点 (e.g. 1.5) 也没事了
    # 因为有了 anchor 约束，不怕跑偏
    guidance_scale = 1.0 

    # === 路径配置 ===
    ckpt_path = fr'IS2B/rIS2B_nakagmi_resnet_adjust/results/best_model_IS2B_resnet_{n_steps}.pth'
    
    # 保存路径 (文件名改一下以区分)
    result_save_path = f'IS2B/rIS2B_nakagmi_resnet_adjust/ber_results/ber_curve_resnet_hybrid.png'
    result_csv_path = f'IS2B/rIS2B_nakagmi_resnet_adjust/ber_results/ber_data_resnet_hybrid.csv'
    
    baseline_csv_path = 'IS2B/rIS2B_nakagmi_resnet_adjust/ber_results/ber_curve_resnet_values.csv'

    # ----- 1. 加载 TimeResNet1D 模型 -----
    print(f"Building TimeResNet1D on {device}...")
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
    test_start, test_end = 400000, 500000
    test_data = QPSKDataset(test_start, test_end)
    
    rx_clean = test_data.y   # [N,2,L]
    h_np = test_data.z       # [N,2,L]
    n_win = rx_clean.shape[0]

    # ----- 3. 加载标签 -----
    label_path = r'F:\LJN\bishe\bishe\data\nakagmi_data\labels.npy'
    print(f"Loading Labels from {label_path}...")
    label_all = np.load(label_path)
    label_seg = label_all[test_start:test_end]
    map_label = {0: (0, 0), 1: (0, 1), 2: (1, 1), 3: (1, 0)}
    labels_iq = np.array([map_label[int(v)] for v in label_seg], dtype=int)

    # ----- 4. 运行仿真 -----
    snr_range = np.arange(0, 19, 1) # 2 ~ 18 dB
    model_bers = []

    print(f"开始测试 (Hybrid: OneStep Anchor + Rectified Flow)...")
    for snr_db in snr_range:
        print(f"\n=== SNR (Symbol): {snr_db} dB ===")
        snr_db_sample = snr_db - 10 * math.log10(sps) + 10 * math.log10(2)

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
    # ----- 6. 读取基准对比 (读取两列参考数据) -----
    ref_resnet_bers = []
    ref_baseline_bers = []
    
    if os.path.exists(baseline_csv_path):
        try:
            with open(baseline_csv_path, 'r', newline='') as f:
                reader = csv.DictReader(f)
                
                # 使用字典暂存数据，方便按 SNR 查找
                # key: snr, value: ber
                csv_resnet_map = {}
                csv_baseline_map = {}
                
                for row in reader:
                    try:
                        # 统一保留1位小数，防止浮点数匹配错误
                        s = round(float(row['snr_db']), 1)
                        csv_resnet_map[s] = float(row['resnet_ber'])
                        csv_baseline_map[s] = float(row['baseline_ber'])
                    except ValueError:
                        continue

                # 根据当前的 snr_range 提取对应的数据
                for snr in snr_range:
                    query_key = round(float(snr), 1)
                    
                    # 获取 ResNet 数据
                    val_res = csv_resnet_map.get(query_key, None)
                    if val_res is not None:
                        ref_resnet_bers.append(val_res)
                    
                    # 获取 Baseline 数据
                    val_base = csv_baseline_map.get(query_key, None)
                    if val_base is not None:
                        ref_baseline_bers.append(val_base)
                        
            print(f"成功加载参考数据: ResNet ({len(ref_resnet_bers)}点), Baseline ({len(ref_baseline_bers)}点)")
            
        except Exception as e:
            print(f"Error reading CSV: {e}")
    else:
        print(f"未找到基准 CSV 文件: {baseline_csv_path}")

    # ----- 7. 绘图 -----
    # 注意参数顺序：当前结果，ResNet参考，Baseline参考，SNR范围，保存路径
    plot_ber_compare(model_bers, ref_resnet_bers, ref_baseline_bers, snr_range, result_save_path)
    print(f"Done. Saved Image to {result_save_path}")