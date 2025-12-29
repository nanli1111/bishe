import torch
import numpy as np

class I2SBJointSampler:
    """
    双head I2SB 采样器（匹配你的新方案）
    - 网络输入（每步）：concat([x_t, est_h_expanded], dim=1) -> [B,12,L]
    - 网络输出：x0_hat [B,2,L], h_hat [B,10,S]
    - 更新规则：x_{t_next} = (1-t_next)*x0_hat + t_next*y  （用 y 锁回线段）
    """

    def __init__(self, model, n_steps=20, sps=16, device="cuda"):
        self.model = model
        self.n_steps = n_steps
        self.sps = sps
        self.device = device

    def _expand_h(self, est_h, L):
        # est_h: [B,10,S] -> [B,10,L]
        return torch.repeat_interleave(est_h, repeats=self.sps, dim=-1)

    @torch.no_grad()
    def one_step(self, y_noisy, est_h):
        """
        t=1 (idx=n_steps-1) 的一步预测：直接得到 x0_hat 和 h_hat
        """
        y_noisy = y_noisy.to(self.device).float()   # [B,2,L]
        est_h = est_h.to(self.device).float()       # [B,10,S]
        B, _, L = y_noisy.shape

        est_h_exp = self._expand_h(est_h, L)        # [B,10,L]
        net_in = torch.cat([y_noisy, est_h_exp], dim=1)  # [B,12,L]

        t_idx = torch.full((B,), self.n_steps - 1, device=self.device, dtype=torch.long)
        x0_hat, h_hat = self.model(net_in, t_idx)
        return x0_hat, h_hat

    @torch.no_grad()
    def sample(self, y_noisy, est_h, return_h_from="first"):
        """
        多步采样：
        return_h_from:
          - "first": 返回 t=1 的 h_hat（通常最稳，且你现在 one-step 很准）
          - "last" : 返回最后一步的 h_hat
        """
        y_noisy = y_noisy.to(self.device).float()
        est_h = est_h.to(self.device).float()
        B, _, L = y_noisy.shape

        est_h_exp = self._expand_h(est_h, L)  # [B,10,L]

        # 初始 x_{t=1} = y
        x = y_noisy

        h_first = None
        h_last = None

        # 直接用离散 idx：n_steps-1 -> 1
        for idx in range(self.n_steps - 1, 0, -1):
            t_next = (idx - 1) / (self.n_steps - 1)  # in [0,1)

            net_in = torch.cat([x, est_h_exp], dim=1)  # [B,12,L]
            t_idx = torch.full((B,), idx, device=self.device, dtype=torch.long)

            x0_hat, h_hat = self.model(net_in, t_idx)

            if h_first is None:
                h_first = h_hat
            h_last = h_hat

            # ✅ 用 y 锁回线段（关键修复点）
            x = (1.0 - t_next) * x0_hat + t_next * y_noisy

        # idx=1 更新后，x 已经是 t=0 的估计（理论上就是 x0）
        if return_h_from == "first":
            return x, h_first
        else:
            return x, h_last
