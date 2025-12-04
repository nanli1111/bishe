import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary

# -----------------------------
# SE 模块 (Squeeze-and-Excitation)
# -----------------------------
class SEBlock1D(nn.Module):
    def __init__(self, channels, reduction=4):
        super(SEBlock1D, self).__init__()
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)

    def forward(self, x):
        b, c, l = x.size()
        y = x.mean(dim=2)  # Global Average Pooling (GAP)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y)).view(b, c, 1)
        return x * y.expand_as(x)

# -----------------------------
# 改进 ResNet1D (SE 可选)
# -----------------------------
class RESNET_model(nn.Module):
    def __init__(self, num_classes=2, dropout=0.2, use_se=True):
        super(RESNET_model, self).__init__()
        self.use_se = use_se

        # BLOCK 1
        # 输入为 signal(2) + channel(2) = 4通道
        self.conv1 = nn.Conv1d(4, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.conv1b = nn.Conv1d(32, 32, kernel_size=3, padding=1)
        self.bn1b = nn.BatchNorm1d(32)
        self.shortcut1 = nn.Conv1d(4, 32, kernel_size=1)
        self.bn1s = nn.BatchNorm1d(32)
        if use_se: self.se1 = SEBlock1D(32)
        self.dropout1 = nn.Dropout(dropout)

        # BLOCK 2 (下采样)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm1d(64)
        self.conv2b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn2b = nn.BatchNorm1d(64)
        self.shortcut2 = nn.Conv1d(32, 64, kernel_size=1, stride=2)
        self.bn2s = nn.BatchNorm1d(64)
        if use_se: self.se2 = SEBlock1D(64)
        self.dropout2 = nn.Dropout(dropout)

        # BLOCK 3
        self.conv3 = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(64)
        self.conv3b = nn.Conv1d(64, 64, kernel_size=3, padding=1)
        self.bn3b = nn.BatchNorm1d(64)
        self.shortcut3 = nn.Identity()
        if use_se: self.se3 = SEBlock1D(64)
        self.dropout3 = nn.Dropout(dropout)

        # GAP + FC
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, signal, channel_estimate):
        # 拼接输入
        x = torch.cat([signal, channel_estimate], dim=1)
        
        # BLOCK 1
        out = F.gelu(self.bn1(self.conv1(x)))
        out = F.gelu(self.bn1b(self.conv1b(out)))
        shortcut = self.bn1s(self.shortcut1(x))
        # 尺寸对齐检查
        if shortcut.size(-1) != out.size(-1):
            shortcut = F.interpolate(shortcut, size=out.size(-1), mode='linear', align_corners=False)
        out = F.gelu(out + shortcut)
        if self.use_se: out = self.se1(out)
        out = self.dropout1(out)

        # BLOCK 2
        out2 = F.gelu(self.bn2(self.conv2(out)))
        out2 = F.gelu(self.bn2b(self.conv2b(out2)))
        shortcut = self.bn2s(self.shortcut2(out))
        if shortcut.size(-1) != out2.size(-1):
            shortcut = F.interpolate(shortcut, size=out2.size(-1), mode='linear', align_corners=False)
        out = F.gelu(out2 + shortcut)
        if self.use_se: out = self.se2(out)
        out = self.dropout2(out)

        # BLOCK 3
        out3 = F.gelu(self.bn3(self.conv3(out)))
        out3 = F.gelu(self.bn3b(self.conv3b(out3)))
        shortcut = self.shortcut3(out)
        if shortcut.size(-1) != out3.size(-1):
            shortcut = F.interpolate(shortcut, size=out3.size(-1), mode='linear', align_corners=False)
        out = F.gelu(out3 + shortcut)
        if self.use_se: out = self.se3(out)
        out = self.dropout3(out)

        # HEAD
        out = self.gap(out).squeeze(-1)
        out = self.fc(out)
        return out

# -----------------------------
# 主程序执行
# -----------------------------
if __name__ == "__main__":
    # 实例化模型
    model = RESNET_model(num_classes=2)
    
    # 模拟输入数据
    # Batch Size = 1, Channels = 2 (I/Q), Length = 48
    signal = torch.randn(1, 2, 48)  
    channel_estimate = torch.randn(1, 2, 48)  

    print("Model Summary:")
    try:
        # 使用 input_data 列表传递位置参数，对应 forward(self, signal, channel_estimate)
        summary(model, input_data=[signal, channel_estimate])
    except Exception as e:
        print(f"Summary failed: {e}")
        print("提示: 请确保安装了 torchinfo (pip install torchinfo)")

    # 测试实际前向传播
    output = model(signal, channel_estimate)
    print("\nExecution Test:")
    print(f"Input Signal Shape: {signal.shape}")
    print(f"Input Channel Est Shape: {channel_estimate.shape}")
    print(f"Output Shape: {output.shape}") # 应该是 [1, 2]