import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ==========================================
# 1) 基础组件
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

# ==========================================
# 2) SE-Block
# ==========================================
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y

# ==========================================
# 3) 残差块
# ==========================================
class SEAdaGNResBlock1D(nn.Module):
    def __init__(self, channels, time_emb_dim, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)

        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)

        self.act = nn.GELU()
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels * 2))
        self.dropout = nn.Dropout(dropout)

        self.se = SEBlock1D(channels, reduction=4)

    def forward(self, x, t_emb):
        residual = x

        h = self.norm1(x)
        t_proj = self.time_proj(t_emb).unsqueeze(-1)  # [B,2C,1]
        scale, shift = t_proj.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv1(h)

        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)

        h = self.se(h)
        return h + residual

# ==========================================
# 4) 双 head：波形恢复 + 信道补全（按你的新方案）
# ==========================================
class SETimeResNet1D(nn.Module):
    """
    输入:
      x: [B, 12, L] = concat([x_t(2), est_h_expanded(10)], dim=1)
      t: [B] (long idx / float 都行)

    输出(固定双输出):
      x0_hat: [B, 2, L]
      h_hat:  [B, 10, S]
    """
    def __init__(
        self,
        in_channels=12,        # ✅ 2 + 10
        out_wave_channels=2,
        out_h_channels=10,     # 5条路径的 Re/Im => 10
        hidden_dim=128,
        num_blocks=12,
        time_emb_dim=128,
        sps=16,                # 176/11=16
        h_symbols=11           # S=11
    ):
        super().__init__()
        self.sps = sps
        self.h_symbols = h_symbols

        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        self.entry = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)

        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)
            self.blocks.append(SEAdaGNResBlock1D(hidden_dim, time_emb_dim, dilation=dilation))

        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.GELU()

        # 波形 head：采样级输出
        self.wave_head = nn.Conv1d(hidden_dim, out_wave_channels, kernel_size=1)

        # 信道 head：符号级输出（先 pool 到 S 再 1x1）
        self.h_head = nn.Conv1d(hidden_dim, out_h_channels, kernel_size=1)

        # zero-init（对 x-pre 很友好）
        self.wave_head.weight.data.zero_()
        self.wave_head.bias.data.zero_()
        self.h_head.weight.data.zero_()
        self.h_head.bias.data.zero_()

    def forward(self, x, t):
        t_emb = self.time_mlp(t.float())

        h = self.entry(x.float())
        for blk in self.blocks:
            h = blk(h, t_emb)

        h = self.final_norm(h)
        h = self.final_act(h)

        # ---- wave ----
        x0_hat = self.wave_head(h)  # [B,2,L]

        # ---- channel ----
        # L=176, sps=16 => S=11
        h_sym_feat = F.avg_pool1d(h, kernel_size=self.sps, stride=self.sps)  # [B,hidden,S]
        # 这里不做插值兜底，按你的“别太防呆”风格：直接假设数据对齐
        h_hat = self.h_head(h_sym_feat)  # [B,10,S]

        return x0_hat, h_hat


if __name__ == "__main__":
    B, L, S = 4, 176, 11
    x_t = torch.randn(B, 2, L)
    est_h_sym = torch.zeros(B, 10, S)
    est_h_sym[:, :, S // 2] = torch.randn(B, 10)  # 仅中心导频非零

    est_h_exp = torch.repeat_interleave(est_h_sym, repeats=L // S, dim=-1)  # [B,10,L]
    net_input = torch.cat([x_t, est_h_exp], dim=1)  # [B,12,L]

    t_idx = torch.randint(0, 20, (B,))  # 假设 n_steps=20

    net = SETimeResNet1D(in_channels=12, sps=16, h_symbols=11)
    x0_hat, h_hat = net(net_input, t_idx)

    print("x0_hat:", x0_hat.shape)  # [B,2,176]
    print("h_hat:", h_hat.shape)    # [B,10,11]
