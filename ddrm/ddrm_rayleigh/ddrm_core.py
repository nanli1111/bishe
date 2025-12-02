import torch
import torch.nn as nn
import torch.nn.functional as F

class DDRM(nn.Module):
    def __init__(self, model, n_steps=100, min_beta=1e-4, max_beta=0.02, device='cuda'):
        """
        model: UNet (in_channels=6, out_channels=2)
        n_steps: 扩散步数
        min_beta, max_beta: beta调度范围
        device: 'cuda' 或 'cpu'
        """
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.device = device

        # ======== 调度器 (标准 DDPM) ========
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        alpha_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]])
        
        # 记录关键系数
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_prev', alpha_prev)
        
        # 预计算 posterior 均值系数 (用于 q_posterior_mean)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alpha_bars))
        
        # 用于标准 DDPM 采样的系数
        self.register_buffer('coef1', torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars))
        self.register_buffer('coef2', torch.sqrt(alpha_prev) * betas / (1 - alpha_bars))

    # ------------------ 辅助函数 ------------------
    def extract(self, a, t, x_shape):
        """提取系数并广播到 x 的维度"""
        batch_size = t.shape[0]
        out = a.gather(-1, t)
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))

    # ------------------ 训练用接口 (手动构造输入) ------------------
    # 注意：原本的 p_losses 只接受 x0，现在我们是在 train.py 中手动实现训练循环
    # 所以这里不需要 p_losses，或者可以留作参考但不会被 train_ddrm_cfg 调用。
    # 我们保留一个简单的 forward 作为兼容
    def forward(self, x_t, y, h, t):
        """
        前向传播：构造 6 通道输入并预测噪声
        x_t: [B, 2, L] 当前噪声状态
        y:   [B, 2, L] 接收信号条件
        h:   [B, 2, L] 信道条件
        """
        net_input = torch.cat([x_t, y, h], dim=1)
        return self.model(net_input, t)
    # ------------------ 纯条件 DDIM 采样 (推荐用于显式条件模型) ------------------
    # ddrm_core.py 中的 DDRM 类新增方法

    @torch.no_grad()
    def sample_pure_conditional(self, y, h, eta=0.0, guidance_scale=1.0):
        """
        DDRM 纯条件采样 (适配显式信道注入 + CFG)
        y: [B, 2, L] 含噪观测
        h: [B, 2] 或 [B, 2, L] 信道
        guidance_scale: CFG 强度
        """
        device = self.device
        y = y.to(device)
        h = h.to(device)
        B, _, L = y.shape

        # 1. 扩展 h 维度
        if h.dim() == 2:
            h_expanded = h.unsqueeze(-1).repeat(1, 1, L)
        else:
            h_expanded = h

        # 2. 随机初始化 x_T
        x = torch.randn_like(y)
        
        # 3. 准备时间步 (倒序 T-1 -> 0)
        step_indices = list(range(self.n_steps - 1, -1, -1))

        # 4. 采样循环
        for t_idx in step_indices:
            t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # --- A. 构造 CFG 输入 (Concat) ---
            # Conditional: [x, y, h]
            # Unconditional: [x, 0, 0]
            x_in = torch.cat([x, x], dim=0)
            y_in = torch.cat([y, torch.zeros_like(y)], dim=0)
            h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
            t_in = torch.cat([t_tensor, t_tensor], dim=0)
            
            # 6通道输入 [2B, 6, L]
            net_input = torch.cat([x_in, y_in, h_in], dim=1)
            
            # --- B. 预测噪声 ---
            noise_pred_double = self.model(net_input, t_in)
            eps_cond, eps_uncond = noise_pred_double.chunk(2, dim=0)
            
            # CFG 组合
            eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)

            # --- C. DDIM 更新 ---
            alpha_bar = self.alpha_bars[t_idx]
            alpha_bar_prev = self.alpha_bars[t_idx - 1] if t_idx > 0 else torch.tensor(1.0).to(device)
            
            # 预测 x0
            pred_x0 = (x - torch.sqrt(1 - alpha_bar) * eps) / torch.sqrt(alpha_bar)
            
            # 幅度截断 (防止毛刺)
            pred_x0 = torch.clamp(pred_x0, -2.0, 2.0)
            
            # 指向 x_{t-1} 的方向 (eta=0, sigma=0)
            dir_xt = torch.sqrt(1 - alpha_bar_prev) * eps
            
            # 更新 x
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt

        return x
    # ------------------ DDRM 恢复 (关键逻辑) ------------------
    @torch.no_grad()
    def restore(self,
                y,
                h,
                sigma_y,
                sigma_p=1.0,
                eta=0.0,
                steps=None,
                init_from='y',
                guidance_scale=0.0): # 支持 CFG (可选)
        """
        DDRM 信道恢复采样器
        结合了 显式条件输入 (Explicit Input) 和 DDRM 的测量一致性 (Measurement Consistency)

        Args:
            y: [B, 2, L]  含噪观测信号 (Rayleigh + AWGN)
            h: [B, 2] 或 [B, 2, L] 信道系数
            sigma_y: 标量，噪声标准差
            guidance_scale: CFG 强度 (如果训练使用了 CFG)
        """
        device = self.device
        y = y.to(device)
        h = h.to(device)

        B, C, L = y.shape
        assert C == 2

        # 1. 处理 h 维度: 确保是 [B, 2, L]
        if h.dim() == 2:
            h_expanded = h.unsqueeze(-1).repeat(1, 1, L) # 用于拼接
        else:
            h_expanded = h
            
        # 2. 准备时间步
        T = int(self.n_steps)
        step_indices = list(range(T - 1, -1, -1)) # 倒序 T-1 -> 0

        # 3. 初始化 x_T
        if init_from == 'y':
            # 从观测信号初始化 (通常对去噪任务有效)
            x_t = y.clone() 
        elif init_from == 'rand':
            # 标准高斯噪声初始化
            x_t = torch.randn_like(y)
        else:
            raise ValueError(f"Unknown init_from: {init_from}")

        # 4. 预计算常数 (用于 DDRM 更新公式)
        sigma_y2 = float(sigma_y) ** 2
        sigma_p2 = float(sigma_p) ** 2
        
        # h 的复数形式处理 (用于 Data Consistency)
        # 假设 h_expanded 是 [B, 2, L]
        h_r_map = h_expanded[:, 0, :] # [B, L]
        h_i_map = h_expanded[:, 1, :] # [B, L]
        abs_h2_map = h_r_map**2 + h_i_map**2 # [B, L]

        # y 的复数形式处理
        y_r_map = y[:, 0, :]
        y_i_map = y[:, 1, :]

        # 5. 逆扩散循环
        for t_idx in step_indices:
            t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)
            
            # --- A. 预测噪声 (CFG 支持) ---
            if guidance_scale > 0:
                # 构造 Unconditional Input (丢弃 y 和 h)
                x_in = torch.cat([x_t, x_t], dim=0)
                y_in = torch.cat([y, torch.zeros_like(y)], dim=0)
                h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)
                
                # 拼接输入 [2B, 6, L]
                net_input = torch.cat([x_in, y_in, h_in], dim=1)
                
                noise_pred = self.model(net_input, t_in)
                eps_cond, eps_uncond = noise_pred.chunk(2, dim=0)
                
                eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
            else:
                # 标准 Conditional Input
                # 必须拼接 6 通道: [x_t, y, h]
                net_input = torch.cat([x_t, y, h_expanded], dim=1)
                eps = self.model(net_input, t_tensor)

            # --- B. DDRM 更新逻辑 (利用物理模型修正 x0) ---
            
            # 1. Tweaked x0 estimation (从 eps 反推 x0)
            # x0_hat = (x_t - sqrt(1-alpha_bar) * eps) / sqrt(alpha_bar)
            alpha_bar_t = self.alpha_bars[t_idx]
            sqrt_ab = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab = torch.sqrt(1 - alpha_bar_t)
            
            x0_hat = (x_t - sqrt_one_minus_ab * eps) / sqrt_ab
            
            # 2. Data Consistency (在 x0 空间融合观测 y)
            # 公式: x0_new = (sigma_y^2 * x0_hat + sigma_p^2 * h^* y) / (sigma_y^2 + sigma_p^2 * |h|^2)
            
            # 分离 I/Q
            x0_r = x0_hat[:, 0, :]
            x0_i = x0_hat[:, 1, :]
            
            # 计算 h^* y (复数乘法: (hr - j hi)(yr + j yi) = (hr yr + hi yi) + j(hr yi - hi yr))
            hy_r = h_r_map * y_r_map + h_i_map * y_i_map
            hy_i = h_r_map * y_i_map - h_i_map * y_r_map
            
            # 分母
            denom = sigma_y2 + sigma_p2 * abs_h2_map # [B, L]
            
            # 融合
            x0_dc_r = (sigma_y2 * x0_r + sigma_p2 * hy_r) / (denom + 1e-8) # 加 epsilon 防除零
            x0_dc_i = (sigma_y2 * x0_i + sigma_p2 * hy_i) / (denom + 1e-8)
            
            x0_dc = torch.stack([x0_dc_r, x0_dc_i], dim=1) # [B, 2, L]
            
            # 3. 采样下一步 x_{t-1} (DDPM 后验均值)
            # mean = coef1 * x_t + coef2 * x0_dc
            c1 = self.coef1[t_idx]
            c2 = self.coef2[t_idx]
            beta_t = self.betas[t_idx]
            
            mean = c1 * x_t + c2 * x0_dc
            
            if t_idx > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta_t) * (1.0 if eta == 1.0 else 0.0) # 简单处理 eta
                x_t = mean + sigma_t * noise
            else:
                x_t = mean

        return x_t