import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# Time embedding (learned + MLP)
# -------------------------
class TimeEmbed(nn.Module):
    def __init__(self, n_steps: int, dim: int):
        super().__init__()
        self.emb = nn.Embedding(n_steps, dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim * 4),
        )
        self.out_dim = dim * 4

    def forward(self, t_idx: torch.Tensor):
        # t_idx: [B] long
        x = self.emb(t_idx)     # [B,dim]
        x = self.mlp(x)         # [B,4dim]
        return x


# -------------------------
# FiLM modulation from time embedding
# -------------------------
class FiLM(nn.Module):
    def __init__(self, time_dim: int, channels: int):
        super().__init__()
        self.to_gb = nn.Linear(time_dim, channels * 2)

    def forward(self, x, t_emb):
        # x: [B,C,L], t_emb: [B,time_dim]
        gb = self.to_gb(t_emb)          # [B,2C]
        g, b = gb.chunk(2, dim=-1)      # [B,C],[B,C]
        g = g.unsqueeze(-1)
        b = b.unsqueeze(-1)
        return x * (1 + g) + b


# -------------------------
# ResBlock1D with time FiLM
# -------------------------
class ResBlock1D(nn.Module):
    def __init__(self, channels: int, time_dim: int, dropout=0.0, dilation=1):
        super().__init__()
        pad = dilation
        self.norm1 = nn.GroupNorm(8, channels)
        self.film1 = FiLM(time_dim, channels)
        self.conv1 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)

        self.norm2 = nn.GroupNorm(8, channels)
        self.film2 = FiLM(time_dim, channels)
        self.conv2 = nn.Conv1d(channels, channels, 3, padding=pad, dilation=dilation)

        self.act = nn.SiLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        r = x
        x = self.norm1(x)
        x = self.film1(x, t_emb)
        x = self.act(x)
        x = self.conv1(x)

        x = self.norm2(x)
        x = self.film2(x, t_emb)
        x = self.act(x)
        x = self.drop(x)
        x = self.conv2(x)
        return x + r


# -------------------------
# Cross Attention: wave tokens attend to symbol tokens
# -------------------------
class CrossAttn(nn.Module):
    def __init__(self, d_wave: int, d_sym: int, nhead: int = 8, dropout=0.0):
        super().__init__()
        self.q = nn.Linear(d_wave, d_wave, bias=False)
        self.k = nn.Linear(d_sym, d_wave, bias=False)
        self.v = nn.Linear(d_sym, d_wave, bias=False)
        self.attn = nn.MultiheadAttention(embed_dim=d_wave, num_heads=nhead, dropout=dropout, batch_first=True)
        self.out = nn.Sequential(
            nn.Linear(d_wave, d_wave),
            nn.SiLU(),
            nn.Linear(d_wave, d_wave),
        )
        self.ln = nn.LayerNorm(d_wave)

    def forward(self, wave_feat, sym_tok):
        """
        wave_feat: [B,C,L]  -> tokens [B,L,C]
        sym_tok:   [B,S,Ds]
        """
        B, C, L = wave_feat.shape
        w = wave_feat.transpose(1, 2)          # [B,L,C]
        w0 = self.ln(w)

        q = self.q(w0)                         # [B,L,C]
        k = self.k(sym_tok)                    # [B,S,C]
        v = self.v(sym_tok)                    # [B,S,C]

        attn_out, _ = self.attn(q, k, v)       # [B,L,C]
        w = w + self.out(attn_out)
        return w.transpose(1, 2)               # [B,C,L]


# -------------------------
# Symbol Transformer encoder (S=11 tokens)
# -------------------------
class SymbolEncoder(nn.Module):
    def __init__(self, in_sym_ch: int, d_model: int, n_layers: int = 4, nhead: int = 8, dropout: float = 0.1, max_S: int = 64):
        super().__init__()
        self.proj = nn.Conv1d(in_sym_ch, d_model, kernel_size=1)

        self.pos = nn.Parameter(torch.randn(1, max_S, d_model) * 0.02)

        enc = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True
        )
        self.enc = nn.TransformerEncoder(enc, num_layers=n_layers)

    def forward(self, sym_in):
        """
        sym_in: [B, Cin, S]  (Cin = 10(h) + 1(mask) = 11)
        return: [B, S, d_model]
        """
        B, _, S = sym_in.shape
        x = self.proj(sym_in)                 # [B,d,S]
        x = x.transpose(1, 2)                 # [B,S,d]
        x = x + self.pos[:, :S, :]
        x = self.enc(x)
        return x


# =========================================================
# ✅ Strong Joint Model (适配你的 IS2B_Joint)
# =========================================================
class StrongJointUNetRF(nn.Module):
    """
    输入:
      net_input: [B, 13, L] = concat([x_t(2), h_sparse_exp(10), mask_exp(1)], dim=1)
      t_idx:     [B] long in [0, n_steps-1]

    输出:
      dict:
        "x0": [B,2,L]
        "h":  [B,10,S]   (S = L//sps, 你的数据是 11)
    """
    def __init__(
        self,
        n_steps: int = 100,
        sps: int = 16,
        in_ch: int = 13,
        h_ch: int = 10,
        base: int = 256,
        sym_d: int = 256,
        sym_layers: int = 4,
        sym_heads: int = 8,
        wave_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.sps = sps
        self.h_ch = h_ch

        # time
        self.t_embed = TimeEmbed(n_steps, base)
        time_dim = self.t_embed.out_dim

        # -------- Symbol branch (from expanded h+mask -> down to symbol tokens -> Transformer) --------
        # sym input channels: 10(h) + 1(mask) = 11
        self.sym_enc = SymbolEncoder(in_sym_ch=h_ch + 1, d_model=sym_d, n_layers=sym_layers, nhead=sym_heads, dropout=dropout, max_S=64)
        self.sym_time = nn.Linear(time_dim, sym_d)

        # -------- Wave UNet --------
        self.in_conv = nn.Conv1d(in_ch, base, 3, padding=1)

        # down
        self.down1 = nn.ModuleList([
            ResBlock1D(base, time_dim, dropout=dropout, dilation=1),
            ResBlock1D(base, time_dim, dropout=dropout, dilation=2),
        ])
        self.ds1 = nn.Conv1d(base, base * 2, 4, stride=2, padding=1)

        self.down2 = nn.ModuleList([
            ResBlock1D(base * 2, time_dim, dropout=dropout, dilation=1),
            ResBlock1D(base * 2, time_dim, dropout=dropout, dilation=2),
        ])
        self.ds2 = nn.Conv1d(base * 2, base * 4, 4, stride=2, padding=1)

        self.down3 = nn.ModuleList([
            ResBlock1D(base * 4, time_dim, dropout=dropout, dilation=1),
            ResBlock1D(base * 4, time_dim, dropout=dropout, dilation=2),
        ])
        self.ds3 = nn.Conv1d(base * 4, base * 4, 4, stride=2, padding=1)

        # bottleneck + cross attention
        self.mid1 = ResBlock1D(base * 4, time_dim, dropout=dropout, dilation=4)
        self.xattn = CrossAttn(d_wave=base * 4, d_sym=sym_d, nhead=wave_heads, dropout=dropout)
        self.mid2 = ResBlock1D(base * 4, time_dim, dropout=dropout, dilation=4)

        # up
        self.us3 = nn.ConvTranspose1d(base * 4, base * 4, 4, stride=2, padding=1)
        self.up3 = nn.ModuleList([
            ResBlock1D(base * 8, time_dim, dropout=dropout, dilation=1),
            ResBlock1D(base * 8, time_dim, dropout=dropout, dilation=2),
        ])
        self.red3 = nn.Conv1d(base * 8, base * 2, 1)

        self.us2 = nn.ConvTranspose1d(base * 2, base * 2, 4, stride=2, padding=1)
        self.up2 = nn.ModuleList([
            ResBlock1D(base * 4, time_dim, dropout=dropout, dilation=1),
            ResBlock1D(base * 4, time_dim, dropout=dropout, dilation=2),
        ])
        self.red2 = nn.Conv1d(base * 4, base, 1)

        self.us1 = nn.ConvTranspose1d(base, base, 4, stride=2, padding=1)
        self.up1 = nn.ModuleList([
            ResBlock1D(base * 2, time_dim, dropout=dropout, dilation=1),
            ResBlock1D(base * 2, time_dim, dropout=dropout, dilation=2),
        ])
        self.red1 = nn.Conv1d(base * 2, base, 1)

        self.out_norm = nn.GroupNorm(8, base)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv1d(base, 2, 1)   # x0_hat

        # -------- h head (symbol-level) --------
        self.h_fuse = nn.Sequential(
            nn.Linear(sym_d + base * 4, sym_d),
            nn.SiLU(),
            nn.Linear(sym_d, sym_d),
        )
        self.h_head = nn.Sequential(
            nn.Conv1d(sym_d, sym_d, 3, padding=1),
            nn.GELU(),
            nn.Conv1d(sym_d, h_ch, 1),
        )

    def forward(self, net_input, t_idx):
        """
        net_input: [B,13,L] = [x_t(2), h_sparse_exp(10), mask_exp(1)]
        t_idx: [B]
        """
        B, Cin, L = net_input.shape
        t_emb = self.t_embed(t_idx)  # [B, time_dim]

        # ----- build symbol tokens from expanded h+mask -----
        # net_input layout (和你的采样器一致)
        #   x_t:   0:2
        #   h_exp: 2:12
        #   mask:  12:13
        h_exp = net_input[:, 2:12, :]       # [B,10,L]
        m_exp = net_input[:, 12:13, :]      # [B,1,L]

        # downsample to symbol-level by avg_pool over sps
        h_sym = F.avg_pool1d(h_exp, kernel_size=self.sps, stride=self.sps)  # [B,10,S]
        m_sym = F.avg_pool1d(m_exp, kernel_size=self.sps, stride=self.sps)  # [B,1,S]
        m_sym = (m_sym > 0.5).float()

        sym_in = torch.cat([h_sym, m_sym], dim=1)  # [B,11,S]
        sym_tok = self.sym_enc(sym_in)             # [B,S,sym_d]
        sym_tok = sym_tok + self.sym_time(t_emb).unsqueeze(1)

        # ----- Wave UNet -----
        x = self.in_conv(net_input)  # [B,base,L]

        # down1
        for blk in self.down1:
            x = blk(x, t_emb)
        skip1 = x
        x = self.ds1(x)              # [B,2base,L/2]

        # down2
        for blk in self.down2:
            x = blk(x, t_emb)
        skip2 = x
        x = self.ds2(x)              # [B,4base,L/4]

        # down3
        for blk in self.down3:
            x = blk(x, t_emb)
        skip3 = x
        x = self.ds3(x)              # [B,4base,L/8]

        # mid + cross-attn
        x = self.mid1(x, t_emb)
        x = self.xattn(x, sym_tok)   # fuse h tokens into wave bottleneck
        x = self.mid2(x, t_emb)

        # also make wave global for h head
        wave_global = F.adaptive_avg_pool1d(x, 1).squeeze(-1)  # [B,4base]

        # up3
        x = self.us3(x)
        if x.shape[-1] != skip3.shape[-1]:
            x = F.interpolate(x, size=skip3.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip3], dim=1)  # [B,8base,*]
        for blk in self.up3:
            x = blk(x, t_emb)
        x = self.red3(x)                 # [B,2base,*]

        # up2
        x = self.us2(x)
        if x.shape[-1] != skip2.shape[-1]:
            x = F.interpolate(x, size=skip2.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip2], dim=1)  # [B,4base,*]
        for blk in self.up2:
            x = blk(x, t_emb)
        x = self.red2(x)                 # [B,base,*]

        # up1
        x = self.us1(x)
        if x.shape[-1] != skip1.shape[-1]:
            x = F.interpolate(x, size=skip1.shape[-1], mode="linear", align_corners=False)
        x = torch.cat([x, skip1], dim=1)  # [B,2base,*]
        for blk in self.up1:
            x = blk(x, t_emb)
        x = self.red1(x)                 # [B,base,L]

        x = self.out_act(self.out_norm(x))
        x0_hat = self.out_conv(x)        # [B,2,L]

        # ----- h head (symbol-level) -----
        # fuse sym_tok with wave_global
        S = sym_tok.shape[1]
        wg = wave_global.unsqueeze(1).expand(B, S, wave_global.shape[-1])  # [B,S,4base]
        h_fused = self.h_fuse(torch.cat([sym_tok, wg], dim=-1))            # [B,S,sym_d]
        h_fused = h_fused.transpose(1, 2)                                  # [B,sym_d,S]
        h_hat = self.h_head(h_fused)                                       # [B,10,S]

        return {"x0": x0_hat, "h": h_hat}
