import os
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import tdl_utils as utils  # 导入工具库


def main():
    # 确保中间数据保存目录存在 (Estimate/tdl/intermediate)
    output_dir = os.path.dirname(utils.CONFIG['output_file'])
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保最终结果/图片保存目录存在 (Estimate/tdl/results/final)
    os.makedirs(utils.CONFIG['save_dir'], exist_ok=True)
    
    # 1. 加载数据
    total_needed = utils.CONFIG['train_samples'] + utils.CONFIG['test_samples']
    loader = utils.TDLDatasetLoader(utils.CONFIG['data_dir'], limit=total_needed)
    
    N_train = utils.CONFIG['train_samples']
    N_test = utils.CONFIG['test_samples']
    
    # 注意这里 get_data 的返回值顺序
    rx_train_clean, h_train_true, h_dataset_est_train, _ = loader.get_data(0, N_train) # 训练集只用于 LMMSE fit
    rx_test_clean, h_test_true, h_dataset_est_test, lbl_test = loader.get_data(N_train, N_train+N_test)
    
    # 2. 计算功率基准
    signal_power_train = np.mean((np.abs(rx_train_clean[:,0,:]) ** 2) + (np.abs(rx_train_clean[:,1,:]) ** 2))
    signal_power_test  = np.mean((np.abs(rx_test_clean[:,0,:]) ** 2) + (np.abs(rx_test_clean[:,1,:]) ** 2))
    
    # 准备存储结构
    saved_data = {
        'snr_range': utils.CONFIG['snr_range'],
        'labels': lbl_test,          # 测试集标签
        'h_true': h_test_true,       # 测试集真实信道 (用于后续对比)
        'h_dataset_est': h_dataset_est_test, # 保存数据集自带的估计信道
        'nmse_ls': [],
        'nmse_lmmse': []
    }
    
    print("\n--- Step 1: Running Estimation & Saving Data ---")
    
    for snr in tqdm(utils.CONFIG['snr_range']):
        EbN0_linear = 10**(snr/10)
        
        # 计算噪声参数
        N0_train = signal_power_train / EbN0_linear
        noise_std_train = np.sqrt(N0_train/2)
        N0_test = signal_power_test / EbN0_linear
        noise_std_test = np.sqrt(N0_test/2)
        
        # 生成含噪信道向量 (用于训练和测试)
        h_train_ls = utils.generate_noisy_channel_vec(h_train_true, noise_std_train)
        h_test_ls  = utils.generate_noisy_channel_vec(h_test_true,  noise_std_test)
        
        # 生成含噪波形
        rx_test_noisy, _ = utils.add_awgn_noise_np(rx_test_clean, snr)
        
        # LMMSE 训练与预测
        lmmse = utils.LMMSE_Denoiser()
        # LMMSE 训练用的是我们模拟加噪的 h_train_ls
        lmmse.fit(h_train_ls, h_train_true) 
        h_test_lmmse = lmmse.predict(h_test_ls)
        
        # 计算 LS 和 LMMSE 的 NMSE (全向量计算，因为它们是稠密的)
        err_ls = np.sum(np.abs(h_test_ls - h_test_true)**2, axis=1)
        pow_true = np.sum(np.abs(h_test_true)**2, axis=1)
        nmse_ls_val = 10*np.log10(np.mean(err_ls / pow_true))
        
        err_lmmse = np.sum(np.abs(h_test_lmmse - h_test_true)**2, axis=1)
        nmse_lmmse_val = 10*np.log10(np.mean(err_lmmse / pow_true))
        
        saved_data['nmse_ls'].append(nmse_ls_val)
        saved_data['nmse_lmmse'].append(nmse_lmmse_val)
        
        # 保存当前 SNR 下的中间结果
        key_prefix = f"snr_{snr}"
        saved_data[f"{key_prefix}_h_ls"] = h_test_ls
        saved_data[f"{key_prefix}_h_lmmse"] = h_test_lmmse
        saved_data[f"{key_prefix}_rx"] = rx_test_noisy

    # 3. 保存数据到磁盘
    np.savez_compressed(utils.CONFIG['output_file'], **saved_data)
    print(f"\nEstimation done. Data saved to: {utils.CONFIG['output_file']}")
    
    # 4. 绘图并保存
    plt.figure(figsize=(8, 6))
    plt.plot(utils.CONFIG['snr_range'], saved_data['nmse_ls'], 'r--o', label='LS (Simulated)')
    plt.plot(utils.CONFIG['snr_range'], saved_data['nmse_lmmse'], 'b-^', label='LMMSE (Simulated)')
    
    # --- 【核心修改】绘制数据集自带估计信道的 NMSE (只计算中心点) ---
    # 原因：h_dataset_est 是稀疏的，只有中心点有值。如果计算全向量误差，未知的点会拉低 NMSE 到 0dB。
    # 我们只关心那个存在的导频点的估计质量。
    
    # 提取中心点 (Index 7)
    center_idx = 7
    h_est_center = saved_data['h_dataset_est'][:, center_idx]
    h_true_center = saved_data['h_true'][:, center_idx]
    
    # 计算中心点的均方误差
    err_dataset_center = np.abs(h_est_center - h_true_center)**2
    pow_true_center = np.abs(h_true_center)**2
    
    # 计算 NMSE (dB) - 这是一个定值，不随循环中的 SNR 变化
    nmse_dataset_val = 10*np.log10(np.mean(err_dataset_center) / np.mean(pow_true_center))
    
    print(f"Dataset Estimated H (Center Tap Only) NMSE: {nmse_dataset_val:.2f} dB")
    
    # 为了绘图，生成一个常数列表
    nmse_dataset_est_list = [nmse_dataset_val] * len(utils.CONFIG['snr_range'])
    
    plt.plot(utils.CONFIG['snr_range'], nmse_dataset_est_list, 'g-x', label='H_Dataset_Est (Center Only)')
    
    plt.xlabel('SNR (dB)')
    plt.ylabel('NMSE (dB)')
    plt.grid(True)
    plt.legend()
    plt.title('Step 1 Check: NMSE Performance')
    
    img_path = os.path.join(utils.CONFIG['save_dir'], 'step1_nmse_check.png')
    plt.savefig(img_path)
    print(f"NMSE plot saved to: {img_path}")
    
    plt.show() 
    plt.close() 

if __name__ == "__main__":
    main()