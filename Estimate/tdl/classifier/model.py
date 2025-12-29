import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1) 复数工具：用 (Re,Im) 做粗等化
# ==========================================
def complex_equalize(x_wave, h_center, eps=1e-6):
    xr, xi = x_wave[:, 0, :], x_wave[:, 1, :]
    hr, hi = h_center[:, 0], h_center[:, 1]   # [B]
    denom = (hr * hr + hi * hi + eps)         # [B]

    yr = (xr * hr.unsqueeze(-1) + xi * hi.unsqueeze(-1)) / denom.unsqueeze(-1)
    yi = (xi * hr.unsqueeze(-1) - xr * hi.unsqueeze(-1)) / denom.unsqueeze(-1)
    return torch.stack([yr, yi], dim=1)

# ==========================================
# 2) FiLM 条件调制（用 h_center 生成 gamma/beta）
# ==========================================
class FiLM(nn.Module):
    def __init__(self, cond_dim, channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cond_dim, channels * 2),
            nn.SiLU(),
            nn.Linear(channels * 2, channels * 2)
        )

    def forward(self, x, cond):
        gb = self.net(cond)               # [B, 2C]
        gamma, beta = gb.chunk(2, dim=-1) # [B, C], [B, C]
        gamma = gamma.unsqueeze(-1)
        beta = beta.unsqueeze(-1)
        return x * (1.0 + gamma) + beta

# ==========================================
# 3) 深度可分离空洞卷积残差块（大感受野 + 轻量）
# ==========================================
class TCNFiLMBlock(nn.Module):
    def __init__(self, channels, cond_dim, dilation=1, kernel_size=5, dropout=0.1):
        super().__init__()
        pad = (kernel_size // 2) * dilation

        self.norm1 = nn.GroupNorm(8, channels)
        self.act1 = nn.GELU()
        self.film1 = FiLM(cond_dim, channels)

        self.dw = nn.Conv1d(channels, channels, kernel_size=kernel_size,
                            padding=pad, dilation=dilation, groups=channels)

        self.norm2 = nn.GroupNorm(8, channels)
        self.act2 = nn.GELU()
        self.film2 = FiLM(cond_dim, channels)

        self.pw = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, cond):
        residual = x
        h = self.norm1(x)
        h = self.film1(h, cond)
        h = self.act1(h)
        h = self.dw(h)

        h = self.norm2(h)
        h = self.film2(h, cond)
        h = self.act2(h)
        h = self.dropout(h)
        h = self.pw(h)

        return h + residual

# ==========================================
# 4) Center-Query Attention Pooling（中心导向）
# ==========================================
class CenterQueryAttention(nn.Module):
    def __init__(self, channels, attn_dim=128):
        super().__init__()
        self.q = nn.Linear(channels, attn_dim, bias=False)
        self.k = nn.Linear(channels, attn_dim, bias=False)
        self.v = nn.Linear(channels, attn_dim, bias=False)
        self.out = nn.Linear(attn_dim, channels, bias=False)

    def forward(self, x, center_idx):
        B, C, L = x.shape
        xc = x[:, :, center_idx]          # [B, C]

        q = self.q(xc)                    # [B, D]
        xt = x.transpose(1, 2)            # [B, L, C]
        k = self.k(xt)                    # [B, L, D]
        v = self.v(xt)                    # [B, L, D]

        attn = torch.sum(k * q.unsqueeze(1), dim=-1) / (k.shape[-1] ** 0.5)  # [B, L]
        w = F.softmax(attn, dim=-1).unsqueeze(-1)                            # [B, L, 1]

        pooled = torch.sum(v * w, dim=1)     # [B, D]
        pooled = self.out(pooled)            # [B, C]
        return pooled, w.squeeze(-1)         # w: [B, L]

# ==========================================
# 5) BER-friendly：输出 2-bit logits (I/Q)
# ==========================================
class BetterQPSKNet(nn.Module):
    def __init__(self,
                 num_classes=2,          # ✅ 改成 2：I/Q 两个 bit 的 logits
                 hidden_dim=192,
                 num_blocks=10,
                 dropout=0.15,
                 sps=16):
        super().__init__()
        assert num_classes == 2, "2-bit BCE 训练请设置 num_classes=2"
        self.sps = sps

        # wave(2) + eq(2) + h_up(2) = 6
        self.entry = nn.Conv1d(6, hidden_dim, kernel_size=3, padding=1)

        self.cond_mlp = nn.Sequential(
            nn.Linear(4, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        cond_dim = hidden_dim

        dilations = [1, 2, 4, 8, 16, 32, 1, 2, 4, 8]
        dilations = dilations[:num_blocks]

        self.blocks = nn.ModuleList([
            TCNFiLMBlock(hidden_dim, cond_dim, dilation=dilations[i], dropout=dropout)
            for i in range(num_blocks)
        ])

        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.GELU()

        self.center_attn = CenterQueryAttention(hidden_dim, attn_dim=128)

        # ✅ 只改最后输出维度：2 logits -> [I_logit, Q_logit]
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x_wave, x_h):
        """
        x_wave: [B, 2, L]   (Re,Im)
        x_h:    [B, 2, S] or [B,2,L]
        return: logits [B,2]  (I/Q bit logits, 用 BCEWithLogitsLoss)
        """
        B, _, L = x_wave.shape

        # 1) H 扩到采样级
        if x_h.shape[-1] != L:
            scale = L // x_h.shape[-1]
            x_h_up = torch.repeat_interleave(x_h, repeats=scale, dim=-1)
        else:
            x_h_up = x_h

        # 2) h_center
        if x_h.shape[-1] != L:          # 符号级
            center_sym = x_h.shape[-1] // 2
            h_center = x_h[:, :, center_sym]   # [B,2]
        else:                           # 采样级
            h_center = x_h[:, :, L // 2]       # [B,2]

        # 3) 粗等化
        x_eq = complex_equalize(x_wave, h_center)

        # 4) cond：Re,Im,|h|,angle
        hr, hi = h_center[:, 0], h_center[:, 1]
        h_mag = torch.sqrt(hr * hr + hi * hi + 1e-6)
        h_ang = torch.atan2(hi, hr)
        cond_raw = torch.stack([hr, hi, h_mag, h_ang], dim=-1)  # [B,4]
        cond = self.cond_mlp(cond_raw)                          # [B, hidden_dim]

        # 5) fusion
        x = torch.cat([x_wave, x_eq, x_h_up], dim=1)            # [B,6,L]
        h = self.entry(x)

        for blk in self.blocks:
            h = blk(h, cond)

        h = self.final_norm(h)
        h = self.final_act(h)

        # 6) center pooling
        center_idx = L // 2
        pooled_attn, _ = self.center_attn(h, center_idx)        # [B,C]

        half = self.sps // 2
        st = max(0, center_idx - half)
        ed = min(L, center_idx + half)
        pooled_local = torch.mean(h[:, :, st:ed], dim=-1)       # [B,C]

        feat = torch.cat([pooled_attn, pooled_local], dim=-1)   # [B,2C]
        logits = self.classifier(feat)                          # ✅ [B,2]
        return logits

# 兼容你现有脚本：from model import OptimizedQPSKNet
OptimizedQPSKNet = BetterQPSKNet

if __name__ == "__main__":
    B, L = 8, 80
    wave = torch.randn(B, 2, L)
    h7 = torch.randn(B, 2, 7)
    model = BetterQPSKNet(num_classes=2)
    out = model(wave, h7)
    print(out.shape)  # [B,2]