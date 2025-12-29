import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPM(nn.Module):
    def __init__(self, model, n_steps=1000, min_beta=1e-4, max_beta=0.02, device='cuda'):
        """
        x_0-prediction DDPM
        model: 输入 (x_t, t) -> 输出 x_0 (预测的纯净信号)
        """
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.device = device

        # ======== 1. Beta 调度 ========
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1. - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_prev = F.pad(alpha_bars[:-1], (1, 0), value=1.0)

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_bars_prev', alpha_bars_prev)

        # ======== 2. 后验均值系数 ========
        self.posterior_mean_coef1 = (
            betas * torch.sqrt(alpha_bars_prev) / (1. - alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1. - alpha_bars_prev) * torch.sqrt(alphas) / (1. - alpha_bars)
        )
        
        # ======== 3. 后验方差 ========
        self.posterior_variance = (
            betas * (1. - alpha_bars_prev) / (1. - alpha_bars)
        )
        
        # === 修复点在这里 ===
        # 将第一个非零方差复制到 t=0 处以防止 log(0) 错误
        # view(1) 确保它是 1D 张量，与 posterior_variance[1:] 维度匹配
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1].view(1), self.posterior_variance[1:]])
        )

    
    def q_sample(self, x0, t, noise=None):
        """前向加噪 (与标准 DDPM 一致)"""
        if noise is None:
            noise = torch.randn_like(x0)
        
        sqrt_alpha_bar_t = torch.sqrt(self.alpha_bars[t]).view(-1, 1, 1)
        sqrt_one_minus_alpha_bar_t = torch.sqrt(1. - self.alpha_bars[t]).view(-1, 1, 1)
        
        return sqrt_alpha_bar_t * x0 + sqrt_one_minus_alpha_bar_t * noise

    def p_losses(self, x0, t, noise=None):
        """
        训练损失: 这里的核心改变是 Target 变为 x0
        """
        if noise is None:
            noise = torch.randn_like(x0)
            
        # 1. 加噪
        x_noisy = self.q_sample(x0, t, noise)
        
        # 2. 预测 x0 (而不是噪声)
        x0_pred = self.model(x_noisy, t)
        
        # 3. 计算 MSE Loss: || x0_pred - x0 ||^2
        return F.mse_loss(x0_pred, x0)

    def q_posterior(self, x0_pred, x_t, t):
        """
        计算后验分布 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        # 获取当前时刻的系数
        coef1 = self.posterior_mean_coef1[t].view(-1, 1, 1)
        coef2 = self.posterior_mean_coef2[t].view(-1, 1, 1)
        
        # 核心公式: 直接利用预测的 x0 和当前的 xt 计算均值
        posterior_mean = coef1 * x0_pred + coef2 * x_t
        
        posterior_variance = self.posterior_variance[t]
        posterior_log_variance_clipped = self.posterior_log_variance_clipped[t]
        
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    @torch.no_grad()
    def sample(self, shape, clip_denoised=True):
        """
        采样过程
        clip_denoised: 是否将预测的 x0 截断到 [-1, 1]。
                       在波形恢复中，如果信号归一化了，建议开启。
        """
        x = torch.randn(shape, device=self.device)

        for t in reversed(range(self.n_steps)):
            # 1. 预测 x0
            t_tensor = torch.full((shape[0],), t, device=self.device, dtype=torch.long)
            x0_pred = self.model(x, t_tensor)

            # 2. (可选) 截断 x0 预测值以稳定生成
            if clip_denoised:
                x0_pred = torch.clamp(x0_pred, -1., 1.)

            # 3. 计算后验均值 (Posterior Mean)
            # 这里直接使用 x0_pred，而不是像 epsilon-pred 那样需要转换
            mean, variance, log_variance = self.q_posterior(x0_pred, x, t)
            
            # 4. 采样 x_{t-1}
            noise = torch.randn_like(x)
            if t > 0:
                # x_{t-1} = mean + sigma * z
                x = mean + torch.exp(0.5 * log_variance) * noise
            else:
                x = mean
                
        return x