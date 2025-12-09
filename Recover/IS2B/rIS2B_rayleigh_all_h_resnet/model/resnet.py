import torch
import torch.nn as nn
import math

class SinusoidalPosEmb(nn.Module):
    """
    正弦位置编码，用于将时间步 t (Scalar) 转换为向量 (Vector)
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [Batch_Size]
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class TimeResBlock1D(nn.Module):
    """
    标准的 1D 残差块：Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(channels)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += residual
        out = self.relu(out)
        return out

class TimeResNet1D(nn.Module):
    """
    支持时间条件 (Time-Conditional) 的 1D ResNet
    用于替换 U-Net 进行信号恢复任务
    """
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=128, num_blocks=8, time_emb_dim=64):
        super().__init__()
        
        # 1. 时间嵌入层 (MLP)
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 2. 头部卷积: 映射输入到特征空间
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 3. 中间残差堆叠 (深度特征提取)
        self.blocks = nn.ModuleList([
            TimeResBlock1D(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 4. 尾部卷积: 映射回 IQ 信号
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x, t):
        """
        Args:
            x: [B, 4, L] 输入信号 (Noisy Y + H)
            t: [B] 时间步索引 (LongTensor or FloatTensor)
        """
        # 1. 计算时间嵌入
        t_emb = self.time_mlp(t.float()) # [B, hidden_dim]
        # 扩展维度以便广播: [B, hidden_dim, 1]
        t_emb = t_emb.unsqueeze(-1) 
        
        # 2. 初始特征提取
        x = x.float()
        h = self.entry(x)
        
        # 3. 注入时间信息 (Time Injection)
        # 将时间特征直接加到信号特征上，让整个网络感知当前的 t
        h = h + t_emb 
        
        # 4. 通过残差块
        for block in self.blocks:
            h = block(h)
            
        # 5. 输出预测
        out = self.exit(h)
        return out