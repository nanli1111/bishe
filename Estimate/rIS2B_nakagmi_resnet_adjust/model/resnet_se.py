import torch
import torch.nn as nn
import math
import torch.nn.functional as F

# ==========================================
# 1. 基础组件 (保持不变)
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
# 2. SE-Block (通道注意力模块)
# ==========================================
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        """
        Squeeze-and-Excitation Block for 1D signals
        reduction: 降维比例，通常取 4 或 8，越小参数越多但表达力越强
        """
        super().__init__()
        # Squeeze: 全局平均池化，将 [B, C, L] -> [B, C, 1]
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        
        # Excitation: 学习通道权重
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid() # 输出 0~1 之间的权重
        )

    def forward(self, x):
        b, c, _ = x.size()
        
        # 1. Squeeze
        y = self.avg_pool(x).view(b, c)
        
        # 2. Excitation
        y = self.fc(y).view(b, c, 1) # 重塑为 [B, C, 1] 以便广播
        
        # 3. Scale (Reweight)
        return x * y

# ==========================================
# 3. 集成 SE 的残差块
# ==========================================
class SEAdaGNResBlock1D(nn.Module):
    """
    集成了 SE-Block 的 AdaGN 残差块
    结构: Norm1 -> Time -> Conv1 -> Norm2 -> Conv2 -> SE -> Residual
    """
    def __init__(self, channels, time_emb_dim, dilation=1, dropout=0.1):
        super().__init__()
        
        # 卷积层
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        
        # 归一化
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        
        # 激活
        self.act = nn.GELU()
        
        # 时间注入
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels * 2))
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # === 新增 SE 模块 ===
        # reduction=4 意味着如果 channels=128，中间层只有 32，非常轻量
        self.se = SEBlock1D(channels, reduction=4)

    def forward(self, x, t_emb):
        residual = x
        
        # Block 1
        h = self.norm1(x)
        t_proj = self.time_proj(t_emb).unsqueeze(-1)
        scale, shift = t_proj.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv1(h)
        
        # Block 2
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        # === 核心：应用 SE 通道加权 ===
        h = self.se(h)
        
        return h + residual

# ==========================================
# 4. 主网络: SETimeResNet1D
# ==========================================
class SETimeResNet1D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=128, num_blocks=8, time_emb_dim=128):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 头部卷积
        self.entry = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 堆叠 SE-Residual Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)
            self.blocks.append(
                SEAdaGNResBlock1D(hidden_dim, time_emb_dim, dilation=dilation)
            )
        
        # 尾部输出
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.GELU()
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)
        
        # Zero Init (让初始输出接近0，加速收敛)
        self.exit.weight.data.zero_()
        self.exit.bias.data.zero_()

    def forward(self, x, t):
        # 1. Time Embedding
        t_emb = self.time_mlp(t.float())
        
        # 2. Entry
        x = x.float()
        h = self.entry(x)
        
        # 3. Backbone
        for block in self.blocks:
            h = block(h, t_emb)
            
        # 4. Exit
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.exit(h)
        
        return out