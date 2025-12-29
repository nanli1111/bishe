
import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 0. 复数工具：用 (real, imag) 两通道表示
# ==========================================
def complex_conj(a):
    # a: [B,2,*]
    return torch.stack([a[:, 0], -a[:, 1]], dim=1)

def complex_mul(x, h):
    # x,h: [B,2,L]
    xr, xi = x[:, 0], x[:, 1]
    hr, hi = h[:, 0], h[:, 1]
    yr = xr * hr - xi * hi
    yi = xr * hi + xi * hr
    return torch.stack([yr, yi], dim=1)

def complex_div(x, h, eps=1e-8):
    """
    x / h = x * conj(h) / (|h|^2 + eps)
    x: [B,2,L], h: [B,2,1] 或 [B,2,L]
    """
    hr, hi = h[:, 0], h[:, 1]
    denom = hr * hr + hi * hi + eps
    h_conj = complex_conj(h)
    y = complex_mul(x, h_conj)
    y = torch.stack([y[:, 0] / denom, y[:, 1] / denom], dim=1)
    return y

# ==========================================
# 1. Pilot 提取：从稀疏 h_est 里找非零点（或最大幅度点）
# ==========================================
def extract_pilot_from_sparse_h(x_h, eps=1e-12):
    """
    x_h: [B,2,S] (S=window_symbols)，稀疏导频，只有少数点非零
    返回 pilot: [B,2,1]
    """
    mag = (x_h[:, 0, :] ** 2 + x_h[:, 1, :] ** 2)
    idx = torch.argmax(mag, dim=-1)  # [B]
    idx_expand = idx.view(-1, 1, 1).repeat(1, 2, 1)
    pilot = torch.gather(x_h, dim=2, index=idx_expand)

    pilot_mag = (pilot[:, 0, 0] ** 2 + pilot[:, 1, 0] ** 2)
    bad = (pilot_mag < eps).view(-1, 1, 1)
    if bad.any():
        pilot = pilot.clone()
        pilot[bad.expand_as(pilot)] = 1.0
    return pilot

# ==========================================
# 2. Symbol Token Transformer (中心符号 2-bit 判决)
# ==========================================
class PilotAidedSymbolTransformer(nn.Module):
    """
    输入:
      x_wave: [B,2,L]   (L = window_symbols * sps)
      x_h:    [B,2,S]   (S = window_symbols, 稀疏导频: 只有中心非零)
    输出:
      logits_bits: [B,2]  (I_bit_logit, Q_bit_logit)  -> 2-bit BCEWithLogitsLoss
    """
    def __init__(self,
                 num_bits=2,              # ✅ 改为 2-bit
                 window_symbols=9,
                 sps=16,
                 stem_dim=96,
                 d_model=192,
                 nhead=4,
                 num_layers=6,
                 dropout=0.12):
        super().__init__()
        self.window_symbols = window_symbols
        self.sps = sps

        # --- 2.2 Wave Stem：raw(2) + eq(2) = 4 channels
        self.stem = nn.Sequential(
            nn.Conv1d(4, stem_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, stem_dim),
            nn.GELU(),
            nn.Conv1d(stem_dim, stem_dim, kernel_size=5, padding=2),
            nn.GroupNorm(8, stem_dim),
            nn.GELU(),
        )

        # --- 2.3 Tokenize：采样级 -> 符号级 token (stride=sps)
        self.to_tokens = nn.Conv1d(stem_dim, d_model, kernel_size=sps, stride=sps)

        # --- 2.4 条件分支：h token + pilot mask token
        self.h_proj = nn.Conv1d(2, d_model, kernel_size=1)
        self.m_proj = nn.Conv1d(1, d_model, kernel_size=1)

        # --- 2.5 位置编码
        self.pos_emb = nn.Parameter(torch.randn(1, window_symbols, d_model) * 0.02)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # --- 2.6 分类头：输出 2 个 bit logits
        self.head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(d_model, num_bits)   # ✅ 从 4 改为 2
        )

    def forward(self, x_wave, x_h):
        """
        x_wave: [B,2,L]
        x_h:    [B,2,S] or [B,2,L]
        """
        B, _, L = x_wave.shape

        # --- 把 x_h 变成 [B,2,S] ---
        if x_h.shape[-1] != self.window_symbols:
            if x_h.shape[-1] == L:
                x_h = F.avg_pool1d(x_h, kernel_size=self.sps, stride=self.sps)
            else:
                x_h = F.interpolate(x_h, size=self.window_symbols, mode="linear", align_corners=False)

        # --- 1) 提取 pilot 并做均衡 ---
        pilot = extract_pilot_from_sparse_h(x_h)          # [B,2,1]
        x_eq = complex_div(x_wave, pilot)                 # [B,2,L]

        # --- 2) wave stem ---
        x = torch.cat([x_wave, x_eq], dim=1)              # [B,4,L]
        feat = self.stem(x)                               # [B,stem_dim,L]

        # --- 3) tokenize ---
        tok = self.to_tokens(feat)                        # [B,d_model,S]
        if tok.shape[-1] != self.window_symbols:
            tok = F.interpolate(tok, size=self.window_symbols, mode="linear", align_corners=False)

        # --- 4) 条件融合：h token + pilot mask token ---
        h_tok = self.h_proj(x_h)                          # [B,d_model,S]
        mask = ((x_h[:, 0:1, :] ** 2 + x_h[:, 1:2, :] ** 2) > 0).float()  # [B,1,S]
        m_tok = self.m_proj(mask)                         # [B,d_model,S]
        tok = tok + h_tok + m_tok                         # [B,d_model,S]

        # --- 5) Transformer ---
        tok = tok.transpose(1, 2)                         # [B,S,d_model]
        tok = tok + self.pos_emb[:, :tok.shape[1], :]
        tok = self.encoder(tok)                           # [B,S,d_model]

        # --- 6) 中心判决：输出 2-bit logits ---
        center = self.window_symbols // 2
        feat_center = tok[:, center, :]
        feat_mean = tok.mean(dim=1)
        logits_bits = self.head(torch.cat([feat_center, feat_mean], dim=-1))  # [B,2]
        return logits_bits


# 兼容你当前工程：from model import OptimizedQPSKNet
OptimizedQPSKNet = PilotAidedSymbolTransformer


if __name__ == "__main__":
    B = 8
    window_symbols = 9
    sps = 16
    L = window_symbols * sps

    x_wave = torch.randn(B, 2, L)
    x_h = torch.zeros(B, 2, window_symbols)
    x_h[:, :, window_symbols // 2] = torch.randn(B, 2)  # 中心 pilot

    model = PilotAidedSymbolTransformer(window_symbols=window_symbols, sps=sps)
    y = model(x_wave, x_h)
    print("logits_bits:", y.shape)  # [B,2]
