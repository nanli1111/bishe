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
# 2. SE-Block (通道注意力模块 - 保持不变)
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
# 3. 集成 SE 的残差块 (保持不变)
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
        
        # SE
        h = self.se(h)
        
        return h + residual

# ==========================================
# 4. 主网络: SETimeResNet1D (已修改)
# ==========================================
class SETimeResNet1D(nn.Module):
    def __init__(self, 
                 in_channels=2,   # <--- 修改：默认为2 (对应 I/Q 两路)
                 out_channels=2,  # <--- 修改：默认为2 (对应 I/Q 两路)
                 hidden_dim=128, 
                 num_blocks=8, 
                 time_emb_dim=128):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 头部卷积: 将 2 通道映射到 128 通道
        self.entry = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 堆叠 SE-Residual Blocks
        self.blocks = nn.ModuleList()
        for i in range(num_blocks):
            dilation = 2 ** (i % 4)
            self.blocks.append(
                SEAdaGNResBlock1D(hidden_dim, time_emb_dim, dilation=dilation)
            )
        
        # 尾部输出: 将 128 通道映射回 2 通道
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.GELU()
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)
        
        # Zero Init (让初始输出接近0，加速扩散模型收敛)
        self.exit.weight.data.zero_()
        self.exit.bias.data.zero_()

    def forward(self, x, t):
        # x shape: [Batch, 2, Length]
        # t shape: [Batch]
        
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
        out = self.exit(h) # shape: [Batch, 2, Length]
        
        return out

# ==========================================
# 5. 测试代码
# ==========================================
if __name__ == "__main__":
    # 配置
    batch_size = 16
    seq_len = 1024
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 实例化模型 (使用默认的 2入2出)
    model = SETimeResNet1D(in_channels=2, out_channels=2, hidden_dim=64).to(device)

    # 模拟输入数据: [Batch, 2, Length] -> 代表 I路 和 Q路
    x = torch.randn(batch_size, 2, seq_len).to(device)
    
    # 模拟时间步: [Batch]
    t = torch.randint(0, 1000, (batch_size,)).to(device)

    # 前向传播
    y = model(x, t)

    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # 验证维度是否未变
    assert x.shape == y.shape, "输入输出维度不匹配！"
    print("Check passed: 2-Channel In/Out successful.")