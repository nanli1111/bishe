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
    def sample_rectified_flow(self, y, h, guidance_scale=1.0, use_heun=False, stop_t=0.0, anchor=None):
        """
        IS2B / Rectified Flow 采样器 (仅在终点进行整体符号翻转)
        
        策略：让 ODE 自由演化，生成最自然的波形。
             在采样结束时，如果最终结果的象限与 Anchor (One-Step) 不一致，
             则认为模型生成了一个"形状完美但方向相反"的波形，将其整体乘以 -1 修正回来。
        """
        device = self.device
        y = y.to(device).float()
        h = h.to(device).float()
        B, C, L = y.shape

        if h.dim() == 2:
            h_expanded = h.unsqueeze(-1).repeat(1, 1, L)
        else:
            h_expanded = h

        x = y.clone()
        timesteps = np.linspace(1.0, 0.0, self.n_steps + 1)
        mid_idx = L // 2

        # === 内部辅助函数 ===
        def get_velocity_and_x0(x_curr, t_scalar):
            t_idx_val = min(int(round(t_scalar * (self.n_steps - 1))), self.n_steps - 1)
            t_idx_val = max(0, t_idx_val)
            t_tensor = torch.full((B,), t_idx_val, device=device, dtype=torch.long)

            if guidance_scale != 1.0:
                x_in = torch.cat([x_curr, x_curr], dim=0)
                h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)
                net_input = torch.cat([x_in, h_in], dim=1)
                
                out = self.model(net_input, t_in)
                x0_cond, x0_uncond = out.chunk(2, dim=0)
                pred_x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                net_input = torch.cat([x_curr, h_expanded], dim=1)
                pred_x0 = self.model(net_input, t_tensor)

            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

            if t_scalar < 1e-5:
                v = torch.zeros_like(x_curr)
            else:
                v = (x_curr - pred_x0) / t_scalar
            
            return v, pred_x0

        # === 1. 执行完整的采样循环 ===
        for i in range(self.n_steps):
            t_curr = timesteps[i]
            t_next = timesteps[i+1]
            dt = t_curr - t_next 

            # 截断采样逻辑
            threshold = max(stop_t, 1e-6)
            if t_next < threshold:
                # 获取最后的预测值作为终点
                _, pred_x0 = get_velocity_and_x0(x, t_curr)
                x = pred_x0
                break # 跳出循环，去执行下面的最终检查

            # 正常更新步
            d_curr, pred_x0_curr = get_velocity_and_x0(x, t_curr)

            if use_heun:
                x_probe = x - dt * d_curr
                d_next, _ = get_velocity_and_x0(x_probe, t_next)
                d_avg = (d_curr + d_next) / 2.0
                x = x - dt * d_avg
            else:
                ratio = t_next / t_curr
                x = ratio * x + (1 - ratio) * pred_x0_curr
            
            # 注意：此处循环中不再对 x 进行任何干预

        # === 2. 采样结束：最终一致性检查与翻转 ===
        if anchor is not None:
            # 1. 获取 One-Step 认为的正确符号 [B, 2]
            anchor_sign = torch.sign(anchor[:, :, mid_idx])
            
            # 2. 获取多步采样最终结果的符号 [B, 2]
            final_sign = torch.sign(x[:, :, mid_idx])
            
            # 3. 判断是否冲突 (1表示反了，0表示一致)
            flip_mask = (anchor_sign != final_sign).float() # [B, 2]
            
            # 4. 构造翻转系数: 一致乘以 1.0，反了乘以 -1.0
            # [B, 2] -> [B, 2, 1]
            correction_factor = (1.0 - 2.0 * flip_mask).unsqueeze(-1)
            
            # 5. 执行整体翻转
            # 这会把那些"形状完美但跑错象限"的波形直接折叠回正确的象限
            x = x * correction_factor

        return x

    def sample(self, y, h, guidance_scale=1.0, stop_t=0.0, anchor=None):
        return self.sample_rectified_flow(
            y, h, 
            guidance_scale=guidance_scale, 
            use_heun=True, 
            stop_t=stop_t, 
            anchor=anchor
        )