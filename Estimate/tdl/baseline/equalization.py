import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv
import tdl_utils as utils  # 导入工具库

def main():
    data_file = utils.CONFIG['output_file']
    if not os.path.exists(data_file):
        print(f"Error: Intermediate file {data_file} not found. Run step1_estimation.py first.")
        return

    # 1. 确保结果保存目录存在
    os.makedirs(utils.CONFIG['save_dir'], exist_ok=True)
    
    print(f"Loading data from {data_file}...")
    loaded_data = np.load(data_file)
    
    # 提取公共数据
    snr_range = loaded_data['snr_range']
    labels = loaded_data['labels']
    nmse_ls_list = loaded_data['nmse_ls']
    nmse_lmmse_list = loaded_data['nmse_lmmse']
    # 新增：从 npz 文件加载 h_dataset_est
    h_dataset_est_full = loaded_data['h_dataset_est'] 
    
    # 【优化步骤 A】我们需要计算信号功率来推导 N0 (复用 Step 1 的逻辑)
    # 临时加载 clean data 计算功率基准
    total_needed = utils.CONFIG['train_samples'] + utils.CONFIG['test_samples']
    loader = utils.TDLDatasetLoader(utils.CONFIG['data_dir'], limit=total_needed)
    N_train = utils.CONFIG['train_samples']
    N_test = utils.CONFIG['test_samples']
    # 注意这里 get_data 的返回值顺序
    rx_test_clean, _, _, _ = loader.get_data(N_train, N_train+N_test) 
    signal_power_test = np.mean((np.abs(rx_test_clean[:,0,:]) ** 2) + (np.abs(rx_test_clean[:,1,:]) ** 2))
    print(f"Signal Power calculated: {signal_power_test:.4f}")

    # 生成 RRC 滤波器
    rrc_filter = utils.generate_rrc_filter(utils.CONFIG['sps'], utils.CONFIG['num_taps'], utils.CONFIG['alpha'])
    
    ber_ls_list = []
    ber_lmmse_list = []
    ber_dataset_est_list = [] # 新增：保存数据集估计的 BER
    
    eval_size = len(labels) 
    lbl_subset = labels[:eval_size]

    print("\n--- Step 2: Running RRC + MMSE Scalar Equalization + BER ---")
    
    for snr in tqdm(snr_range):
        key_prefix = f"snr_{snr}"
        
        # 1. 读取数据
        h_ls = loaded_data[f"{key_prefix}_h_ls"][:eval_size]
        h_lmmse = loaded_data[f"{key_prefix}_h_lmmse"][:eval_size]
        rx_noisy = loaded_data[f"{key_prefix}_rx"][:eval_size]
        
        # 从完整的数据集中提取对应 snr 的 h_dataset_est (如果是固定值，这里也是一样的)
        h_dataset_est_current = h_dataset_est_full[:eval_size]
        
        # 2. 计算当前 SNR 下的噪声功率 N0 (用于 MMSE 均衡)
        EbN0_linear = 10**(snr/10)
        N0 = signal_power_test / EbN0_linear
        
        # 3. RRC 匹配滤波 + 下采样
        rx_symbols = utils.apply_rrc_and_downsample(rx_noisy, rrc_filter, utils.CONFIG['sps'])
        
        # 4. MMSE 标量均衡 (对三条曲线都执行)
        # 提取中心点
        h_ls_center = h_ls[:, 7]
        h_lmmse_center = h_lmmse[:, 7]
        h_dataset_est_center = h_dataset_est_current[:, 7] # 新增：数据集估计的中心点
        sym_rx_center = rx_symbols[:, 7]
        
        # LS 均衡 (MMSE Scalar: y * h* / (|h|^2 + N0))
        num_ls = sym_rx_center * np.conj(h_ls_center)
        den_ls = (np.abs(h_ls_center) ** 2) + N0
        sym_eq_ls = num_ls / den_ls
        
        # LMMSE 均衡
        num_lmmse = sym_rx_center * np.conj(h_lmmse_center)
        den_lmmse = (np.abs(h_lmmse_center) ** 2) + N0
        sym_eq_lmmse = num_lmmse / den_lmmse
        
        # 新增：数据集估计均衡
        num_dataset_est = sym_rx_center * np.conj(h_dataset_est_center)
        den_dataset_est = (np.abs(h_dataset_est_center) ** 2) + N0 # 同样使用当前 SNR 的 N0
        sym_eq_dataset_est = num_dataset_est / den_dataset_est
        
        # 5. 判决与计算 BER
        preds_ls = utils.qpsk_decision(sym_eq_ls)
        preds_lmmse = utils.qpsk_decision(sym_eq_lmmse)
        preds_dataset_est = utils.qpsk_decision(sym_eq_dataset_est) # 新增
        
        ber_ls = np.mean(preds_ls != lbl_subset)
        ber_lmmse = np.mean(preds_lmmse != lbl_subset)
        ber_dataset_est = np.mean(preds_dataset_est != lbl_subset) # 新增
        
        ber_ls_list.append(ber_ls)
        ber_lmmse_list.append(ber_lmmse)
        ber_dataset_est_list.append(ber_dataset_est) # 新增

    # 保存
    csv_path = os.path.join(utils.CONFIG['save_dir'], 'final_separated_results_mmse_eq_with_dataset_est.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['snr', 'nmse_ls', 'nmse_lmmse', 'ber_ls', 'ber_lmmse', 'ber_dataset_est'])
        writer.writerows(zip(snr_range, nmse_ls_list, nmse_lmmse_list, ber_ls_list, ber_lmmse_list, ber_dataset_est_list))
        
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 左图：NMSE (直接使用 Step 1 计算好的)
    plt.subplot(1, 2, 1)
    plt.plot(snr_range, nmse_ls_list, 'r--o', label='LS (Simulated)')
    plt.plot(snr_range, nmse_lmmse_list, 'b-^', label='LMMSE (Simulated)')
    
    # --- 【核心修改】计算数据集自带估计信道的 NMSE (只计算中心点) ---
    nmse_dataset_est_list = []
    h_true_for_nmse = loaded_data['h_true']
    
    # 提取中心点 (Index 7)
    center_idx = 7
    h_est_center_full = h_dataset_est_full[:, center_idx]
    h_true_center_full = h_true_for_nmse[:, center_idx]
    
    # 计算均方误差 (标量)
    err_dataset_center = np.abs(h_est_center_full - h_true_center_full)**2
    pow_true_center = np.abs(h_true_center_full)**2
    nmse_dataset_val = 10*np.log10(np.mean(err_dataset_center) / np.mean(pow_true_center))
    
    print(f"Dataset Estimated H (Center Tap Only) NMSE: {nmse_dataset_val:.2f} dB")
    
    # 生成列表用于绘图
    nmse_dataset_est_list = [nmse_dataset_val] * len(snr_range)
    
    plt.plot(snr_range, nmse_dataset_est_list, 'g-x', label='H_Dataset_Est (Center Only)')
    
    plt.xlabel('SNR (dB)'); plt.ylabel('NMSE (dB)'); plt.grid(True)
    plt.title('Step 1 Result: NMSE Comparison')
    plt.legend()
    
    # 右图：BER (Step 2 计算的)
    plt.subplot(1, 2, 2)
    plt.semilogy(snr_range, ber_ls_list, 'r--o', label='LS (Simulated) + MMSE Eq')
    plt.semilogy(snr_range, ber_lmmse_list, 'b-^', label='LMMSE (Simulated) + MMSE Eq')
    plt.semilogy(snr_range, ber_dataset_est_list, 'g-x', label='H_Dataset_Est + MMSE Eq')
    
    plt.xlabel('SNR (dB)'); plt.ylabel('BER'); plt.grid(True, which='both', linestyle=':')
    plt.title('Step 2 Result: BER (Optimized) Comparison')
    plt.legend()
    
    img_path = os.path.join(utils.CONFIG['save_dir'], 'final_separated_plot_mmse_eq_with_dataset_est.png')
    plt.tight_layout()
    plt.savefig(img_path)
    print(f"\nAll done! Optimized results saved to {utils.CONFIG['save_dir']}")
    plt.show()

if __name__ == "__main__":
    main()