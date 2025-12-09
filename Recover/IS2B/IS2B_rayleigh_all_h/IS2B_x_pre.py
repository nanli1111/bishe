import torch
import torch.nn as nn
import numpy as np

class IS2B(nn.Module):
    def __init__(self, model, n_steps=100, min_beta=1e-4, max_beta=0.02, device='cuda'):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.device = device

        # 1. 扩散调度器 (Gaussian Diffusion)
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        
        # alpha_prev (t-1), t=0 时设为 1.0
        alpha_bars_prev = torch.cat([torch.tensor([1.0], device=device), alpha_bars[:-1]])

        # === 注册基础 Buffer ===
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)

        # === 关键修复：添加训练代码需要的系数 ===
        # 你的 train.py 中使用了 extract(IS2B.sqrt_alphas_cumprod...)
        # 所以必须在这里定义它们，否则训练会报错 AttributeError
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alpha_bars))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alpha_bars))

    @torch.no_grad()
    def sample_ddim_from_intermediate(self, h, start_x, start_step, eta=0.0):
        """
        从中间时间步开始的 DDIM 采样 (适配对比实验)
        
        Args:
            h: [B, 2] or [B, 2, L] 信道条件
            start_x: [B, 2, L] 起始信号 (被视为 x_t)
            start_step: int, 起始时间步 t (0 <= t < n_steps)
        """
        device = self.device
        B, C, L = start_x.shape
        x = start_x.clone()

        # 1. 扩展 h 维度
        if h.dim() == 2:
            h_expanded = h.unsqueeze(-1).repeat(1, 1, L)
        else:
            h_expanded = h

        # 2. 构造倒序时间步列表: [start_step, start_step-1, ..., 0]
        step_indices = list(range(start_step, -1, -1))

        # 3. DDIM 逆向循环
        for t_idx in step_indices:
            t_tensor = torch.full((B,), t_idx, device=device, dtype=torch.long)

            # --- A. 构造输入 [x_t, h] ---
            # 这里的输入必须与训练时一致 (4通道)
            net_input = torch.cat([x, h_expanded], dim=1)

            # --- B. 预测 x0 ---
            # 训练目标是 x0-prediction
            pred_x0 = self.model(net_input, t_tensor)
            
            # 幅度截断 (防止数值发散)
            pred_x0 = torch.clamp(pred_x0, -2.5, 2.5)

            # --- C. DDIM 更新公式 ---
            alpha_bar = self.alpha_bars[t_idx]
            alpha_bar_prev = self.alpha_bars_prev[t_idx]
            
            # 1. 反推隐含的噪声 epsilon
            # eps = (x_t - sqrt(alpha_bar) * x0) / sqrt(1 - alpha_bar)
            # 注意防止分母为0 (虽然后期 alpha_bar 离 1 很远，但加上 1e-8 更安全)
            eps_implied = (x - torch.sqrt(alpha_bar) * pred_x0) / torch.sqrt(1.0 - alpha_bar + 1e-8)
            
            # 2. 计算方向 (DDIM, eta=0 为确定性)
            sigma_t = eta * torch.sqrt((1 - alpha_bar_prev) / (1 - alpha_bar) * (1 - alpha_bar / alpha_bar_prev))
            dir_xt = torch.sqrt(1.0 - alpha_bar_prev - sigma_t**2) * eps_implied
            
            # 3. 迈向下一步 x_{t-1}
            noise = sigma_t * torch.randn_like(x)
            x = torch.sqrt(alpha_bar_prev) * pred_x0 + dir_xt + noise

        return x