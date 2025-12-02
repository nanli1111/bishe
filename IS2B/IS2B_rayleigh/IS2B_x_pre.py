import torch
import torch.nn as nn
import numpy as np

class IS2B(nn.Module):
    def __init__(self, model, n_steps=100, min_beta=1e-4, max_beta=0.02, device='cuda'):
        """
        IS2B 核心类 (适配 x0-prediction)
        
        Args:
            model: UNet 模型 (注意：训练目标必须是预测 x0，而非 epsilon)
            n_steps: 扩散步数
            device: 设备
        """
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.device = device

        # ======================================
        # 1. 扩散调度器 (Linear Schedule)
        # ======================================
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # alpha_prev (t-1), t=0 时设为 1.0
        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]])

        # 注册为 buffer，会自动随模型 .to(device)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)

        # ======================================
        # 2. 预计算系数 (用于训练和采样)
        # ======================================
        # 用于 q_sample (前向加噪): x_t = sqrt(ab) * x0 + sqrt(1-ab) * eps
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alpha_bars))
        
        # 用于后验均值计算 (标准 DDPM)
        self.register_buffer('posterior_mean_coef1', 
                             betas * torch.sqrt(alpha_bars_prev) / (1.0 - alpha_bars))
        self.register_buffer('posterior_mean_coef2', 
                             (1.0 - alpha_bars_prev) * torch.sqrt(alphas) / (1.0 - alpha_bars))
        self.register_buffer('posterior_variance', 
                             betas * (1.0 - alpha_bars_prev) / (1.0 - alpha_bars))

    # ----------------------------------------------------------------
    # 采样器 A: 纯条件采样 (推荐)
    # 适用于: x0-prediction, 显式信道注入 (Explicit CSI), CFG
    # 特点: 波形最光滑，无额外噪声引入
    # ----------------------------------------------------------------
    @torch.no_grad()
    def sample_x0_prediction(self, y, h, eta=0.0, guidance_scale=1.0):
        """
        基于预测 x0 的 DDIM 采样
        """
        device = self.device
        y = y.to(device)
        h = h.to(device)
        B, C, L = y.shape

        # 1. 扩展 h 维度 [B, 2] -> [B, 2, L]
        if h.dim() == 2:
            h_expanded = h.unsqueeze(-1).repeat(1, 1, L)
        else:
            h_expanded = h

        # 2. 随机初始化 x_T
        x = torch.randn_like(y)
        
        # 3. 准备时间步 (倒序 T-1 -> 0)
        step_indices = list(range(self.n_steps - 1, -1, -1))

        # 4. 逆扩散循环
        for t_idx in step_indices:
            t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # --- (1) 构造 CFG 输入 ---
            # x: [2B, 2, L]
            x_in = torch.cat([x, x], dim=0)
            # y: Cond=[y], Uncond=[0]
            y_in = torch.cat([y, torch.zeros_like(y)], dim=0)
            # h: Cond=[h], Uncond=[0]
            h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
            # t: [2B]
            t_in = torch.cat([t_tensor, t_tensor], dim=0)
            
            # 6通道输入
            net_input = torch.cat([x_in, y_in, h_in], dim=1)
            
            # --- (2) 预测 x0 (注意: 模型直接输出 clean signal) ---
            # 这一步非常关键：因为是 x0-pred，所以不需要除以 sqrt(alpha_bar)，数值极其稳定
            x0_pred_double = self.model(net_input, t_in)
            x0_cond, x0_uncond = x0_pred_double.chunk(2, dim=0)
            
            # CFG 组合
            pred_x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)

            # --- (3) 幅度截断 (Clamping) ---
            # 强制约束物理范围，消除毛刺
            pred_x0 = torch.clamp(pred_x0, -2.0, 2.0)

            # --- (4) DDIM 更新 (基于 x0 的公式) ---
            # x_{t-1} = sqrt(ab_prev) * x0_pred + dir_xt * eps_implied
            alpha_bar = self.alpha_bars[t_idx]
            alpha_bar_prev = self.alpha_bars_prev[t_idx]
            
            # 反推隐含的噪声 epsilon (用于确定方向)
            # eps = (x_t - sqrt(ab) * x0) / sqrt(1-ab)
            eps_implied = (x - torch.sqrt(alpha_bar) * pred_x0) / torch.sqrt(1.0 - alpha_bar)
            
            # 计算方向 (eta=0 为确定性采样)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma_t**2) * eps_implied
            
            # 随机噪声 (如果 eta > 0)
            noise = sigma_t * torch.randn_like(x)
            
            # 更新 x
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise

        return x

    # ----------------------------------------------------------------
    # 采样器 B: 物理引导恢复 (可选)
    # 适用于: x0-prediction, 需要强物理约束 (Data Consistency) 的场景
    # ----------------------------------------------------------------
    @torch.no_grad()
    def restore_x0_prediction(self, y, h, sigma_y, sigma_p=1.0, guidance_scale=1.0):
        """
        适配 x0-prediction 的 IS2B 物理约束采样
        """
        device = self.device
        y = y.to(device)
        h = h.to(device)
        B, C, L = y.shape

        if h.dim() == 2:
            h_expanded = h.unsqueeze(-1).repeat(1, 1, L)
        else:
            h_expanded = h

        # 准备物理常数
        sigma_y2 = float(sigma_y) ** 2
        sigma_p2 = float(sigma_p) ** 2
        
        h_r = h_expanded[:, 0, :]
        h_i = h_expanded[:, 1, :]
        abs_h2 = h_r**2 + h_i**2
        y_r = y[:, 0, :]
        y_i = y[:, 1, :]

        # 初始化
        x = torch.randn_like(y)
        step_indices = list(range(self.n_steps - 1, -1, -1))

        for t_idx in step_indices:
            t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # 1. 神经网络预测 x0 (带 CFG)
            x_in = torch.cat([x, x], dim=0)
            y_in = torch.cat([y, torch.zeros_like(y)], dim=0)
            h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
            t_in = torch.cat([t_tensor, t_tensor], dim=0)
            
            net_input = torch.cat([x_in, y_in, h_in], dim=1)
            x0_pred_double = self.model(net_input, t_in)
            x0_cond, x0_uncond = x0_pred_double.chunk(2, dim=0)
            
            pred_x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            pred_x0 = torch.clamp(pred_x0, -2.0, 2.0)

            # 2. Data Consistency (物理修正)
            # 在 x0 空间直接融合: x0_new = (sigma_y^2 * x0_net + sigma_p^2 * y/h)
            # 计算 h* y
            hy_r = h_r * y_r + h_i * y_i
            hy_i = h_r * y_i - h_i * y_r
            
            denom = sigma_y2 + sigma_p2 * abs_h2
            
            x0_dc_r = (sigma_y2 * pred_x0[:, 0, :] + sigma_p2 * hy_r) / (denom + 1e-8)
            x0_dc_i = (sigma_y2 * pred_x0[:, 1, :] + sigma_p2 * hy_i) / (denom + 1e-8)
            x0_dc = torch.stack([x0_dc_r, x0_dc_i], dim=1)

            # 3. 使用修正后的 x0_dc 进行更新
            alpha_bar = self.alpha_bars[t_idx]
            alpha_bar_prev = self.alpha_bars_prev[t_idx]
            
            # 重新计算 implied epsilon (基于修正后的 x0)
            eps_implied = (x - torch.sqrt(alpha_bar) * x0_dc) / torch.sqrt(1.0 - alpha_bar)
            
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev) * eps_implied
            x = torch.sqrt(alpha_bar_prev) * x0_dc + dir_xt

        return x