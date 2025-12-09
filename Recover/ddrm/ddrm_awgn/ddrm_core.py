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

        self.register_buffer('betas', betas)
        self.register_buffer('alphas', alphas)
        self.register_buffer('alpha_bars', alpha_bars)
        self.register_buffer('alpha_prev', alpha_prev)
        self.register_buffer('coef1', torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars))
        self.register_buffer('coef2', torch.sqrt(alpha_prev) * betas / (1 - alpha_bars))

    def q_sample(self, x0, t, noise=None):
        """前向加噪: x_t = sqrt(alpha_bar_t) * x0 + sqrt(1-alpha_bar_t) * noise"""
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t].view(-1, 1, 1)
        return torch.sqrt(alpha_bar) * x0 + torch.sqrt(1 - alpha_bar) * noise

    def p_losses(self, x0, t):
        """训练损失: 预测噪声"""
        noise = torch.randn_like(x0)
        x_noisy = self.q_sample(x0, t, noise)
        noise_pred = self.model(x_noisy, t)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def denoise(self, x_noisy):
        """
        DDRM逆扩散去噪
        x_noisy: 输入带噪信号
        """
        x = x_noisy.to(self.device)

        for t in reversed(range(self.n_steps)):
            n = x.shape[0]
            t_tensor = torch.full((n,), t, device=self.device, dtype=torch.long)
            eps = self.model(x, t_tensor)

            # DDRM公式去噪
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0

            x0_pred = (x - torch.sqrt(1 - self.alpha_bars[t]) * eps) / torch.sqrt(self.alpha_bars[t])
            x0_pred = torch.clip(x0_pred, -1, 1)

            mean = self.coef1[t] * x + self.coef2[t] * x0_pred
            x = mean + torch.sqrt(self.betas[t]) * z

        return x
