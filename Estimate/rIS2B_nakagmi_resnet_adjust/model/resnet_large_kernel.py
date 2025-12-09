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

class LargeKernelResBlock(nn.Module):
    """
    大卷积核残差块 (Large Kernel ResBlock)
    特点：
    1. 使用较大的 kernel_size (如 7, 9)
    2. padding 自动计算以保持维度不变
    3. AdaGN 时间注入
    """
    def __init__(self, channels, time_emb_dim, kernel_size=7, dropout=0.1):
        super().__init__()
        
        # 自动计算 padding 以保持序列长度 L 不变
        # 公式: padding = (kernel_size - 1) // 2
        padding = (kernel_size - 1) // 2
        
        # 1. 第一层卷积 (大核)
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        
        # 2. 第二层卷积 (大核)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=kernel_size, padding=padding)
        
        # 3. 归一化层 (GroupNorm)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # 4. 激活函数
        self.act = nn.GELU()
        
        # 5. 时间步映射 (AdaGN)
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, channels * 2) # 输出 scale 和 shift
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        residual = x
        
        # --- Part 1: Norm + Time Injection + Conv ---
        h = self.norm1(x)
        
        # AdaGN 注入时间信息
        t_proj = self.time_proj(t_emb).unsqueeze(-1)
        scale, shift = t_proj.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        
        h = self.act(h)
        h = self.conv1(h)
        
        # --- Part 2: Norm + Act + Dropout + Conv ---
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + residual

class LargeKernelTimeResNet1D(nn.Module):
    """
    基于大卷积核的时间条件 ResNet
    """
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=256, num_blocks=8, kernel_size=7, time_emb_dim=128):
        super().__init__()
        
        # 1. 时间嵌入 MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 2. 头部特征提取 (使用较小的 3x3 卷积先做初步映射)
        self.entry = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 3. 堆叠大卷积核残差块
        self.blocks = nn.ModuleList([
            LargeKernelResBlock(
                channels=hidden_dim, 
                time_emb_dim=time_emb_dim, 
                kernel_size=kernel_size  # 所有层使用统一的大卷积核
            ) for _ in range(num_blocks)
        ])
        
        # 4. 尾部输出
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.GELU()
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)
        
        # 初始化输出层为 0，使初始预测接近 0 (有助于稳定训练初期)
        self.exit.weight.data.zero_()
        self.exit.bias.data.zero_()

    def forward(self, x, t):
        """
        x: [B, 4, L]
        t: [B]
        """
        # 1. 计算时间嵌入
        t_emb = self.time_mlp(t.float())
        
        # 2. 入口
        x = x.float()
        h = self.entry(x)
        
        # 3. 通过大核残差块
        for block in self.blocks:
            h = block(h, t_emb)
            
        # 4. 出口
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.exit(h)
        
        return out