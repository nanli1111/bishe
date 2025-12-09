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
    def sample_rectified_flow(self, y, h, guidance_scale=1.0, use_heun=False, stop_t=0.0):
        """
        IS2B / Rectified Flow 采样器 (支持截断采样)
        
        Args:
            y: (Batch, 2, L) 含噪信号 (对应 t=1.0)
            h: (Batch, 2, L) 信道估计
            guidance_scale: CFG 强度
            use_heun: 是否使用二阶 Heun 方法
            stop_t: 截断时间阈值 (float). 
                    - 默认 0.0 (不截断，一直走到 t=0).
                    - 建议尝试 0.1 或 0.05。当 t < stop_t 时，直接输出预测的 x0 并结束。
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
            t_idx_val = min(int(round(t_scalar * (self.n_steps - 1))), self.n_steps - 1)
            t_idx_val = max(0, t_idx_val)
            
            t_tensor = torch.full((B,), t_idx_val, device=device, dtype=torch.long)

            # CFG 处理
            if guidance_scale != 1.0:
                # Double Batch: [Cond, Uncond]
                x_in = torch.cat([x_curr, x_curr], dim=0)
                h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)
                
                net_input = torch.cat([x_in, h_in], dim=1)
                out = self.model(net_input, t_in)
                x0_cond, x0_uncond = out.chunk(2, dim=0)
                
                # CFG 公式
                pred_x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                net_input = torch.cat([x_curr, h_expanded], dim=1)
                pred_x0 = self.model(net_input, t_tensor)

            # 截断保护
            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

            # 计算速度 v
            if t_scalar < 1e-5:
                v = torch.zeros_like(x_curr)
            else:
                v = (x_curr - pred_x0) / t_scalar
            
            return v, pred_x0

        # === 4. 逆向求解 ODE Loop ===
        for i in range(self.n_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            dt = t_curr - t_next 

            # ===== 核心修改：截断采样逻辑 =====
            # 如果下一步的时间 t_next 小于设定的停止阈值 stop_t，
            # 说明我们已经走到了足够深的地方，不需要再微调了。
            # 直接获取当前对 x0 的最佳预测，并作为最终结果返回。
            # 注意：这里用 1e-6 兜底，防止 stop_t=0 时除零错误。
            threshold = max(stop_t, 1e-6)
            
            if t_next < threshold:
                # 获取当前状态下预测的 x0
                _, pred_x0 = get_velocity_and_x0(x, t_curr)
                # 直接跳到终点
                x = pred_x0
                break
            # ===============================

            # --- 正常更新步 ---
            d_curr, pred_x0_curr = get_velocity_and_x0(x, t_curr)

            if use_heun:
                # Heun Step
                x_probe = x - dt * d_curr
                d_next, _ = get_velocity_and_x0(x_probe, t_next)
                d_avg = (d_curr + d_next) / 2.0
                x = x - dt * d_avg
            else:
                # Stable Interpolation Step
                ratio = t_next / t_curr
                x = ratio * x + (1 - ratio) * pred_x0_curr

        return x

    def sample(self, y, h, guidance_scale=1.0, stop_t=0.0):
        # 暴露 stop_t 接口，默认 0.0 (不截断)
        # 建议测试：stop_t = 0.1 或 0.05
        return self.sample_rectified_flow(y, h, guidance_scale, use_heun=True, stop_t=stop_t)