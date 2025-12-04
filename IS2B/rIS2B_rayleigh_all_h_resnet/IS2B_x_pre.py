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
        IS2B / Rectified Flow 采样器
        
        Args:
            y: 接收到的含噪信号 (Batch, 2, Seq_Len) [t=1.0 起点]
            h: 信道估计 (Batch, 2, Seq_Len) 或 (Batch, 2)
            guidance_scale: CFG 强度 (1.0 为不使用 CFG)
            use_heun: 是否使用二阶 Heun 方法 (精度更高，速度慢一倍)
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
        
        # 定义获取预测结果的辅助函数 (包含 CFG 逻辑)
        def get_model_prediction(x_input, t_now_idx):
            # 构造 Time Embedding
            # 注意：t_idx 需要扩展为与 x_input 的 Batch Size 一致
            current_batch = x_input.shape[0]
            
            # 如果是 CFG，x_input 已经是 2*B，这里需要判断
            # 简单起见，传入原始 Batch 的 t_idx，然后在内部 cat
            t_tensor = torch.full((B,), t_now_idx, device=device, dtype=torch.long)

            if guidance_scale != 1.0:
                # 构造 Batch: [Conditional, Unconditional]
                # x_input 期望在外部处理好是否 double，或者在这里处理
                # 为了一致性，我们假设外部传入的是单倍 x，在这里做 cat
                
                x_in = torch.cat([x_input, x_input], dim=0)
                h_in = torch.cat([h_expanded, torch.zeros_like(h_expanded)], dim=0)
                t_in = torch.cat([t_tensor, t_tensor], dim=0)
                
                # 4通道输入
                net_input = torch.cat([x_in, h_in], dim=1)
                
                # 预测
                out = self.model(net_input, t_in)
                x0_cond, x0_uncond = out.chunk(2, dim=0)
                
                # CFG 公式: Uncond + w * (Cond - Uncond)
                pred_x0 = x0_uncond + guidance_scale * (x0_cond - x0_uncond)
            else:
                # 标准推断
                net_input = torch.cat([x_input, h_expanded], dim=1)
                pred_x0 = self.model(net_input, t_tensor)
            
            return pred_x0

        # 3. 逆向循环: 从 t=1.0 走到 t=0.0
        # 使用 linspace 生成时间步，更精确
        timesteps = np.linspace(1.0, 0.0, self.n_steps + 1)
        
        for i in range(self.n_steps):
            t_curr = timesteps[i]     # 当前时间 (例如 1.00)
            t_next = timesteps[i+1]   # 下一步时间 (例如 0.99)
            
            # 对应的离散 Time Index (用于输入网络)
            # t=1.0 -> idx=99; t -> 0 -> idx=0
            # 简单的映射：(t * (N-1)) 四舍五入
            t_idx = min(int(round(t_curr * (self.n_steps - 1))), self.n_steps - 1)

            # === A. 预测 x0 (当前点的切线方向目标) ===
            pred_x0 = get_model_prediction(x, t_idx)
            
            # 幅度截断 (防止数值爆炸)
            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

            # === B. 更新步 (Update Step) ===
            
            # dt 是正数，表示时间步长
            dt = t_curr - t_next 

            if use_heun and i < self.n_steps - 1:
                # --- 方法 1: Heun's Method (二阶，更准) ---
                # 1. 计算当前点的速度 d_curr
                # 直线方程: x_t = (1-t)x0 + t*x1 => d_curr = (x_t - x0) / t
                if t_curr < 1e-6: d_curr = torch.zeros_like(x) # 避免除零
                else: d_curr = (x - pred_x0) / t_curr
                
                # 2. 试探性地走一步 (Euler Step) 到 x_next_probe
                x_probe = x - dt * d_curr
                
                # 3. 在新位置预测 x0_probe
                t_next_idx = min(int(round(t_next * (self.n_steps - 1))), self.n_steps - 1)
                pred_x0_probe = get_model_prediction(x_probe, t_next_idx)
                pred_x0_probe = torch.clamp(pred_x0_probe, -3.0, 3.0)
                
                # 4. 计算新位置的速度 d_next
                if t_next < 1e-6: d_next = d_curr # 避免除零
                else: d_next = (x_probe - pred_x0_probe) / t_next
                
                # 5. 取平均速度更新
                d_avg = (d_curr + d_next) / 2.0
                x = x - dt * d_avg
                
            else:
                # --- 方法 2: Stable Interpolation (一阶，更稳) ---
                # 不使用 (x - pred_x0) * dt/t 这种微分形式
                # 而是使用几何加权平均形式 (Rectified Flow 特性)
                # x_next = (t_next / t_curr) * x_curr + (1 - t_next / t_curr) * pred_x0
                
                if t_curr < 1e-6:
                    # 最后一步直接到达 x0
                    x = pred_x0
                else:
                    ratio = t_next / t_curr
                    x = ratio * x + (1 - ratio) * pred_x0

        return x

    def sample(self, y, h, guidance_scale=1.0):
        # 默认推荐使用 Heun 方法以获得更好的恢复质量，尤其是高步数时
        # 如果追求速度，可以设为 False
        return self.sample_rectified_flow(y, h, guidance_scale, use_heun=True)