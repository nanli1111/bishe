import torch
import torch.nn as nn

class ResBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        # 保持维度不变: kernel_size=3, padding=1
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
        # 残差连接
        out += residual
        out = self.relu(out)
        return out

class SimpleResNet1D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=64, num_blocks=4):
        """
        Args:
            in_channels: 4 (Real_Y, Imag_Y, Real_H, Imag_H)
            out_channels: 2 (Real_X, Imag_X)
            hidden_dim: 中间层通道数，64 或 128 足够
            num_blocks: 残差块数量，4-8 层通常足够
        """
        super().__init__()
        
        # 1. 头部卷积：将输入映射到特征空间
        self.entry = nn.Sequential(
            nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # 2. 中间残差堆叠
        self.blocks = nn.ModuleList([
            ResBlock1D(hidden_dim) for _ in range(num_blocks)
        ])
        
        # 3. 尾部卷积：映射回 IQ 信号
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)

    def forward(self, x, t=None):
        """
        x: [Batch, 4, Length]
        t: 为了兼容之前的训练代码保留的参数，这里直接忽略
        """
        # 确保输入是 float
        x = x.float()
        
        out = self.entry(x)
        
        for block in self.blocks:
            out = block(out)
            
        out = self.exit(out)
        
        return out

# 简单的构建函数，保持和你之前代码接口一致
def build_simple_network(device='cuda'):
    return SimpleResNet1D().to(device)