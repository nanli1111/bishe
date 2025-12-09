import torch
import torch.nn as nn
import numpy as np

class IS2B(nn.Module):
    def __init__(self, model, n_steps=100, device='cuda'):
        super().__init__()
        self.model = model
        self.n_steps = n_steps
        self.device = device

    @torch.no_grad()
    def sample_rectified_flow(self, y, h, guidance_scale=1.0, use_heun=False):
        """
        IS2B / Rectified Flow 采样器 (数值稳定版)
        
        Args:
            y: (Batch, 2, L) 含噪信号 (对应 t=1.0)
            h: (Batch, 2, L) 信道估计
            guidance_scale: CFG 强度
            use_heun: 是否使用二阶 Heun 方法 (推荐开启)
        """
        device = self.device
        y = y.to(device).float()
        h = h.to(device).float()
        B, C, L = y.shape

        # 1. 扩展 h 维度
        if h.dim() == 2:
            h_expanded = h.unsqueeze(-1).repeat(1, 1, L)
        else:
            h_expanded = h

        # 2. 初始化: 从 y 开始 (t=1.0)
        x = y.clone()
        
        # 3. 时间步设置: 从 1.0 -> 0.0
        timesteps = np.linspace(1.0, 0.0, self.n_steps + 1)

        # === 内部辅助函数: 计算速度 v 和预测 x0 ===
        def get_velocity_and_x0(x_curr, t_scalar):
            # 映射时间 t 到离散索引 [0, n_steps-1]
            # 限制范围防止越界
            t_idx_val = min(int(round(t_scalar * (self.n_steps - 1))), self.n_steps - 1)
            t_idx_val = max(0, t_idx_val)
            
            t_tensor = torch.full((B,), t_idx_val, device=device, dtype=torch.long)

            # CFG 处理
            if guidance_scale != 1.0:
                # Double Batch: [Cond, Uncond]
                x_in = torch.cat([x_curr, x_curr], dim=0)
                h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0) # t 也需要 double
                
                net_input = torch.cat([x_in, h_in], dim=1)
                
                out = self.model(net_input, t_in)
                x0_cond, x0_uncond = out.chunk(2, dim=0)
                
                # CFG 公式
                pred_x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                # 标准前向
                net_input = torch.cat([x_curr, h_expanded], dim=1)
                pred_x0 = self.model(net_input, t_tensor)

            # 截断保护
            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

            # 计算速度 v = (x_t - x_0) / t
            # 当 t 非常小时，除法不稳定，这里加一个极小值 epsilon 或在外部处理
            if t_scalar < 1e-5:
                v = torch.zeros_like(x_curr) # 占位，实际上不会用到
            else:
                v = (x_curr - pred_x0) / t_scalar
            
            return v, pred_x0

        # === 4. 逆向求解 ODE Loop ===
        for i in range(self.n_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            dt = t_curr - t_next # 正数，表示步长

            # --- 特殊情况：最后一步 (t_next -> 0) ---
            # 在 t 接近 0 时，Heun 方法需要的 d_next 会除以 0，导致数值爆炸。
            # 因此，如果下一步是 0，我们强制使用一阶插值直接跳到 x0。
            if t_next < 1e-6:
                # 获取当前预测的 x0
                _, pred_x0 = get_velocity_and_x0(x, t_curr)
                # 直接信任模型预测，跳到终点
                x = pred_x0
                break

            # --- 正常步 ---
            # 1. 计算当前速度 d_curr
            d_curr, pred_x0_curr = get_velocity_and_x0(x, t_curr)

            if use_heun:
                # === Heun Step (二阶) ===
                # A. 试探步 (Euler): 走到 x_probe
                x_probe = x - dt * d_curr
                
                # B. 计算试探点的速度 d_next
                d_next, _ = get_velocity_and_x0(x_probe, t_next)
                
                # C. 取平均速度更新
                d_avg = (d_curr + d_next) / 2.0
                x = x - dt * d_avg
            else:
                # === Stable Interpolation Step (一阶) ===
                # 相比 x = x - dt * d_curr，插值形式数值更稳
                # x_{next} = (t_{next}/t_{curr}) * x + (1 - t_{next}/t_{curr}) * x0
                ratio = t_next / t_curr
                x = ratio * x + (1 - ratio) * pred_x0_curr

        return x

    def sample(self, y, h, guidance_scale=1.0):
        # 默认推荐使用 Heun，因为它能修正曲线轨迹，显著降低底噪
        return self.sample_rectified_flow(y, h, guidance_scale, use_heun=True)