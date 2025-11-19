import torch
import torch.nn as nn
import numpy as np

class CDDM():
    """
    Channel Denoising Diffusion Models (CDDM) - 信道去噪扩散模型
    基于论文: CDDM: Channel Denoising Diffusion Models for Wireless Semantic Communications
    
    核心思想: 将扩散模型应用于无线通信信道去噪，通过学习信道输入信号的分布来去除信道噪声
    """

    def __init__(self,
                 device,
                 n_steps: int,
                 channel_type: str = 'rayleigh',  # 'rayleigh' or 'awgn'
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        """
        初始化CDDM模型
        参数:
            device: 计算设备 (cuda/cpu)
            n_steps: 扩散过程总步数
            channel_type: 信道类型 ('rayleigh'瑞利衰落或'awgn'加性高斯白噪声)
            min_beta, max_beta: 噪声调度参数，控制噪声添加的速率
        """
        
        # ==================== 基础扩散参数初始化 ====================
        # 创建线性增长的beta序列，控制每个时间步的噪声水平
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas  # alpha = 1 - beta
        alpha_bars = torch.empty_like(alphas)
        
        # 计算累积乘积 alpha_bar_t = ∏_{i=1}^t alpha_i
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        
        # 存储参数
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        self.device = device
        self.channel_type = channel_type
        
        # ==================== CDDM专用参数 ====================
        # 计算前一个时间步的alpha_bar
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]  # 前一时间步的累积乘积
        alpha_prev[0] = 1  # 初始时间步
        
        # 逆向采样过程中的系数 (用于公式29)
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def compute_W_matrices(self, h_c, sigma):
        """
        计算W_s和W_n矩阵 (论文公式6)
        这些矩阵编码了信道特性和噪声信息，是CDDM的核心
        
        参数:
            h_c: 信道估计 [batch_size, k] 复数，表示每个子载波的信道响应
            sigma: 噪声标准差
            
        返回:
            W_s: 信号权重矩阵 [batch_size, 2k, 2k]
            W_n: 噪声权重矩阵 [batch_size, 2k, 2k] 
            h_r: 实数值信道响应 [batch_size, 2k]
        """
        # 确保输入是二维的
        if h_c.dim() == 1:
            h_c = h_c.unsqueeze(0)
        
        batch_size, k = h_c.shape
        # 将复数信道转换为实数值表示 (I路和Q路)
        h_r = torch.cat([torch.abs(h_c), torch.abs(h_c)], dim=1)  # [batch_size, 2k]
        
        # 计算W_s和W_n的对角元素 (公式6)
        h_sq = torch.abs(h_c)**2  # 信道功率
        W_s_diag = h_sq / (h_sq + 2 * sigma**2)  # 信号权重
        W_n_diag = torch.abs(h_c) / (h_sq + 2 * sigma**2)  # 噪声权重
        
        # 构建对角矩阵 - 每个批次样本有自己的W矩阵
        W_s = torch.diag_embed(torch.cat([W_s_diag, W_s_diag], dim=1))
        W_n = torch.diag_embed(torch.cat([W_n_diag, W_n_diag], dim=1))
        
        return W_s, W_n, h_r

    def get_sampling_steps(self, sigma, t_max=None):
        """
        根据信道噪声确定采样步数 (论文公式21, 41)
        找到与当前信道噪声水平最匹配的扩散步数
        
        参数:
            sigma: 噪声标准差
            t_max: 最大采样步数限制 (防止过度采样)
            
        返回:
            m: 最优采样步数
        """
        if t_max is None:
            t_max = self.n_steps
        
        # 目标比值: 2 * sigma^2 对应信道噪声水平
        target_ratio = 2 * sigma**2
        
        # 计算所有时间步的噪声水平与目标水平的差异
        differences = torch.abs(target_ratio - (1 - self.alpha_bars) / self.alpha_bars)
        
        # 找到最接近的时间步
        m = torch.argmin(differences).item()
        
        # 应用最大步数限制
        return min(t_max, m)

    def sample_forward(self, x_0, t, h_c, sigma, eps=None):
        """
        CDDM前向扩散过程 (论文公式17)
        模拟信号通过无线信道的过程：在原始信号上逐步添加噪声
        
        参数:
            x_0: 原始信号 W_s * x [batch_size, 2, 48]
            t: 时间步 [batch_size]
            h_c: 信道估计 [batch_size, k] 复数
            sigma: 噪声标准差
            eps: 可选的外部噪声
            
        返回:
            x_t: 加噪后的信号
            eps: 使用的噪声
        """
        # 计算信道相关的W矩阵
        W_s, W_n, h_r = self.compute_W_matrices(h_c, sigma)
        
        # 获取当前时间步的alpha_bar
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1)  # 调整形状以匹配x_0
              
        # CDDM前向过程: x_t = sqrt(alpha_bar) * x_0 + sqrt(1-alpha_bar) * W_n * eps
        # 这里W_n * eps 模拟了信道特定的噪声特性
        noise_component = torch.sqrt(1 - alpha_bar) * torch.matmul(W_n, eps.unsqueeze(-1)).squeeze(-1)
        signal_component = torch.sqrt(alpha_bar) * x_0
        
        x_t = signal_component + noise_component
        return x_t

    def sample_backward(self, y_r, net, h_c, sigma, t_max=None, clip_x0=True):
        """
        CDDM逆向采样过程 (论文Algorithm 2)
        从接收信号开始，逐步去除噪声，恢复原始信号
        
        参数:
            y_r: 接收信号 (均衡后) [batch_size, 2, 48]
            net: 噪声预测网络 (CDDMConvNet 或 CDDMUNet)
            h_c: 信道估计 [batch_size, k] 复数
            sigma: 噪声标准差
            t_max: 最大采样步数
            clip_x0: 是否对估计的x_0进行裁剪
            
        返回:
            y: 去噪后的信号 [batch_size, 2, 48]
        """
        # 确定最优采样步数
        m = self.get_sampling_steps(sigma, t_max)
        print(f"使用采样步数: {m}")
        
        # 从接收信号开始逆向过程 (x_m = y_r)
        x_t = y_r.clone()
        net = net.to(self.device)
        
        # 计算信道矩阵
        W_s, W_n, h_r = self.compute_W_matrices(h_c, sigma)
        
        # 逆向采样过程: 从步数m逐步回到1
        for t in range(m, 0, -1):
            x_t = self.sample_backward_step(x_t, t, net, W_n, h_r, clip_x0)
        
        # 最后一步 t=1: 直接估计原始信号
        t_tensor = torch.tensor([1] * x_t.shape[0], dtype=torch.long).to(self.device).unsqueeze(1)
        eps_pred = net(x_t, t_tensor, h_r)  # 预测噪声
        z = torch.matmul(W_n, eps_pred.unsqueeze(-1)).squeeze(-1)  # 应用噪声权重
        y = (x_t - torch.sqrt(1 - self.alpha_bars[1]) * z) / torch.sqrt(self.alpha_bars[1])
        
        return y

    def sample_backward_step(self, x_t, t, net, W_n, h_r, clip_x0=True):
        """
        单步逆向采样 (论文公式29)
        从一个噪声版本 x_t 生成更干净的版本 x_{t-1}
        
        参数:
            x_t: 当前时间步的加噪信号 [batch_size, 2, 48]
            t: 当前时间步
            net: 噪声预测网络
            W_n: 噪声权重矩阵
            h_r: 实数值信道响应
            clip_x0: 是否裁剪估计值
            
        返回:
            x_t_minus_1: 下一个时间步的信号
        """
        n = x_t.shape[0]
        # 创建时间步张量
        t_tensor = torch.tensor([t] * n, dtype=torch.long).to(self.device).unsqueeze(1)
        
        # 使用网络预测噪声
        eps_pred = net(x_t, t_tensor, h_r)
        
        # 计算 z = W_n * eps_pred (信道加权的预测噪声)
        z = torch.matmul(W_n, eps_pred.unsqueeze(-1)).squeeze(-1)
        
        if t > 1:
            # 估计原始信号 x_0
            x_0_est = (x_t - torch.sqrt(1 - self.alpha_bars[t]) * z) / torch.sqrt(self.alpha_bars[t])
            
            # 可选: 对估计值进行裁剪，保持数值稳定性
            if clip_x0:
                x_0_est = torch.clip(x_0_est, -1, 1)
            
            # 计算 x_{t-1} 的均值 (公式29)
            mean = (torch.sqrt(self.alpha_bars[t-1]) * 
                   (x_t - torch.sqrt(1 - self.alpha_bars[t]) * z) / torch.sqrt(self.alpha_bars[t]) +
                   torch.sqrt(1 - self.alpha_bars[t-1]) * z)
            
            # 添加随机噪声 (只在t>1时)
            noise = torch.randn_like(x_t)
            var = self.betas[t]
            x_t_minus_1 = mean + torch.sqrt(var) * noise
        else:
            # t=1时直接使用当前值
            x_t_minus_1 = x_t
        
        return x_t_minus_1

