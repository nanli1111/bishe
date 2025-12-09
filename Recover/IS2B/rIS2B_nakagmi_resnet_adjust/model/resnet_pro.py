import torch
import torch.nn as nn
import math
import torch.nn.functional as F

class SinusoidalPosEmb(nn.Module):
    """正弦位置编码 (保持不变)"""
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

class AdaGNResBlock1D(nn.Module):
    """
    升级版残差块：
    1. 支持空洞卷积 (Dilation) -> 扩大感受野
    2. 支持 AdaGN (时间步注入) -> 增强时间控制
    """
    def __init__(self, channels, time_emb_dim, dilation=1, dropout=0.1):
        super().__init__()
        
        # 1. 卷积层 (使用空洞卷积)
        # padding = dilation 保证输出长度不变
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        
        # 2. 归一化层 (GroupNorm 比 BatchNorm 在小 Batch 下更稳)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # 3. 激活函数
        self.act = nn.GELU()
        
        # 4. 时间步映射 (AdaGN 核心)
        # 将 time_emb 映射为 (scale, shift) 两个参数
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * 2) # 输出 scale 和 shift
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        """
        x: [B, C, L]
        t_emb: [B, time_dim]
        """
        residual = x
        
        # --- Block 1: 包含时间注入 ---
        h = self.norm1(x)
        
        # AdaGN: 计算时间带来的 Scale 和 Shift
        # t_proj: [B, 2*C] -> [B, 2*C, 1]
        t_proj = self.time_proj(t_emb).unsqueeze(-1)
        scale, shift = t_proj.chunk(2, dim=1)
        
        # 核心公式: h = h * (1 + scale) + shift
        h = h * (1 + scale) + shift
        
        h = self.act(h)
        h = self.conv1(h)
        
        # --- Block 2 ---
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual

class DilatedTimeResNet1D(nn.Module):
    """
    Pro 版 ResNet:
    - 更深: hidden_dim 加大
    - 更广: 空洞卷积
    - 更强: AdaGN 时间注入
    """
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=128, num_blocks=8, time_emb_dim=128):
        super().__init__()
        
        # 1. 时间嵌入 (MLP)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 2. 头部卷积
        self.entry = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 3. 堆叠残差块 (使用循环空洞率)
        # 例如: 1, 2, 4, 8, 1, 2, 4, 8...
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 4) # 循环膨胀率: 1, 2, 4, 8...
            self.blocks.append(
                AdaGNResBlock1D(hidden_dim, time_emb_dim, dilation=dilation)
            )
        
        # 4. 尾部输出
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.GELU()
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)
        
        # 初始化权重 (可选，有助于收敛)
        self.exit.weight.data.zero_()
        self.exit.bias.data.zero_()

    def forward(self, x, t):
        """
        x: [B, 4, L]
        t: [B] (Long or Float)
        """
        # 1. 计算时间嵌入
        t_emb = self.time_mlp(t.float()) # [B, time_emb_dim]
        
        # 2. 入口
        x = x.float()
        h = self.entry(x)
        
        # 3. 通过所有残差块 (每层都注入 t)
        for block in self.blocks:
            h = block(h, t_emb)
            
        # 4. 出口
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.exit(h)
        
        return out