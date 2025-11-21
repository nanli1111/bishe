import torch
import torch.nn as nn
import torch.nn.functional as F

class DDRM(nn.Module):
    def __init__(self, model, n_steps=100, min_beta=1e-4, max_beta=0.02, device='cuda'):
        """
        model: UNet 或 ConvNet
        n_steps: 扩散步数
        min_beta, max_beta: beta调度范围
        device: 'cuda' 或 'cpu'
        """
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.device = device

        # ======== 调度器（保持和原DDPM一致） ========
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[:-1]
        alpha_prev[0] = 1

        self.register_buffer('betas', betas)              # β_t
        self.register_buffer('alphas', alphas)            # α_t
        self.register_buffer('alpha_bars', alpha_bars)    # \barα_t
        self.register_buffer('alpha_prev', alpha_prev)    # \barα_{t-1}
        # 这两个 coef1/coef2 是你之前写的那套 DDPM 形式
        self.register_buffer('coef1', torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars))
        self.register_buffer('coef2', torch.sqrt(alpha_prev) * betas / (1 - alpha_bars))

    # ------------------ 训练用前向扩散 & loss ------------------
    def q_sample(self, x0, t, noise=None):
        """前向加噪: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        # t: [B]，alpha_bar: [B,1,1]
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def p_losses(self, x0, t):
        """训练损失: 预测噪声 ε"""
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    # ------------------ 原来的无条件去噪（先验采样器） ------------------
    @torch.no_grad()
    def denoise(self, x_noisy):
        """
        纯先验 DDPM 逆扩散去噪（不使用信道观测）
        x_noisy: 输入带噪信号，shape [B, 2, L]
        """
        x = x_noisy.to(self.device)

        for t in reversed(range(self.n_steps)):
            n = x.shape[0]
            t_tensor = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self.model(x, t_tensor)   # 预测噪声 ε_θ(x_t, t)

            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0

            # 从 ε 反推 x0 估计
            x0_pred = (x - torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alpha_bars[t])
            x0_pred = torch.clip(x0_pred, -1, 1)

            # 标准 DDPM 形式的后验均值
            mean = self.coef1[t] * x + self.coef2[t] * x0_pred
            x = mean + torch.sqrt(self.betas[t]) * z

        return x

    # ------------------ 新增：带信道约束的 DDRM 恢复 ------------------
    @torch.no_grad()
    def restore(self,
                y,
                h,
                sigma_y,
                sigma_p=1.0,
                eta=0.0,
                steps=None,
                init_from='y'):
        """
        DDRM 信道恢复采样器：利用观测 y 和信道 h 约束，恢复干净 x0

        y: [B, 2, L]  I/Q 波形
        h: [B, 2]     (real, imag)
        """
        device = self.device
        y = y.to(device)
        h = h.to(device)

        B, C, L = y.shape
        assert C == 2, f"expect 2 channels (IQ), got {C}"

        T = int(self.n_steps)
        if (steps is None) or (steps > T):
            steps = T

        # 构造一个纯 Python 的时间步列表，避免 GPU linspace 花活
        if steps == T:
            step_indices = list(range(T - 1, -1, -1))
        else:
            import numpy as np
            step_indices = np.linspace(T - 1, 0, steps, dtype=np.int64).tolist()
            step_indices = sorted(set(int(t) for t in step_indices), reverse=True)

        alpha_bars = self.alpha_bars    # [T]
        betas      = self.betas         # [T]
        coef1      = self.coef1         # [T]
        coef2      = self.coef2         # [T]

        # 初始化 x_T
        if init_from == 'y':
            x_t = y.clone()             # [B,2,L]
        elif init_from == 'rand':
            x_t = torch.randn_like(y)
        else:
            raise ValueError(f"init_from must be 'y' or 'rand', got {init_from}")

        sigma_y2 = float(sigma_y) ** 2
        sigma_p2 = float(sigma_p) ** 2

        # h: [B,2] -> [B,1,1]
        h_r = h[:, 0].view(B, 1, 1)
        h_i = h[:, 1].view(B, 1, 1)
        abs_h2 = h_r ** 2 + h_i ** 2       # [B,1,1]

        # y: [B,2,L] -> [B,1,L]
        y_r = y[:, 0, :].unsqueeze(1)      # [B,1,L]
        y_i = y[:, 1, :].unsqueeze(1)      # [B,1,L]

        for t_int in step_indices:
            alpha_bar_t = alpha_bars[t_int]          # 标量 tensor
            sqrt_ab_t = torch.sqrt(alpha_bar_t)
            sqrt_one_minus_ab_t = torch.sqrt(1.0 - alpha_bar_t)

            # 1) 先验步：x_t → x0_hat
            t_tensor = torch.full((B,), t_int, device=device, dtype=torch.long)
            eps = self.model(x_t, t_tensor)          # [B,2,L]

            x0_hat = (x_t - sqrt_one_minus_ab_t * eps) / sqrt_ab_t   # [B,2,L]
            x0_hat = torch.clamp(x0_hat, -1., 1.)

            x0_r = x0_hat[:, 0, :]    # [B,L]
            x0_i = x0_hat[:, 1, :]    # [B,L]

            # 2) 数据一致性步：在 x0 空间做高斯后验 -> x0_dc
            # conj(h)*y
            hy_r = h_r * y_r + h_i * y_i          # [B,1,L]
            hy_i = h_r * y_i - h_i * y_r          # [B,1,L]

            hy_r = hy_r.squeeze(1)                # [B,L]
            hy_i = hy_i.squeeze(1)                # [B,L]

            # denom: [B,1]，只在“用户”维度上不同，时间维统一
            denom = (sigma_y2 + sigma_p2 * abs_h2).view(B, 1)   # [B,1]

            x0_dc_r = (sigma_y2 * x0_r + sigma_p2 * hy_r) / denom  # [B,L]
            x0_dc_i = (sigma_y2 * x0_i + sigma_p2 * hy_i) / denom  # [B,L]

            x0_dc = torch.stack([x0_dc_r, x0_dc_i], dim=1)   # [B,2,L]
            x0_dc = torch.clamp(x0_dc, -1., 1.)

            # 关键断言：防止再出现 [B,2,B,L] 这种形状
            assert x_t.shape == x0_dc.shape, \
                f"x_t shape {x_t.shape}, x0_dc shape {x0_dc.shape}"

            # 3) 从 x0_dc 走一步到 x_{t-1}
            beta_t  = betas[t_int]
            coef1_t = coef1[t_int]
            coef2_t = coef2[t_int]

            mean = coef1_t * x_t + coef2_t * x0_dc   # [B,2,L]

            if t_int > 0:
                if eta > 0:
                    sigma_t = eta * torch.sqrt(beta_t)
                else:
                    sigma_t = torch.sqrt(beta_t)
                noise = sigma_t * torch.randn_like(x_t)
                x_t = mean + noise
            else:
                x_t = mean

        return x_t
