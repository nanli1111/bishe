import torch
import torch.nn as nn
import torch.nn.functional as F

def get_signal_shape():
    """获取信号形状定义
    返回: (通道数, 信号长度)
    在CDDM中，这通常对应I/Q两路信号
    """
    return (2, 48)  # 2个通道(I路和Q路)，长度为48的信号

class PositionalEncoding(nn.Module):
    """位置编码模块
    为时间步t生成位置编码，帮助网络理解不同的扩散步骤
    基于Transformer的sin-cos位置编码
    """
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        
        # 确保嵌入维度是偶数，因为我们要分成sin和cos两部分
        assert d_model % 2 == 0

        # 创建位置编码矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)  # 位置序列
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)    # 维度序列
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        
        # 计算sin和cos位置编码
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))      # 偶数维度: sin
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))    # 奇数维度: cos
        
        # 合并sin和cos编码
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        # 使用Embedding层存储位置编码
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)  # 冻结参数，不参与训练

    def forward(self, t):
        """前向传播
        参数:
            t: 时间步张量 [batch_size, 1]
        返回:
            位置编码 [batch_size, d_model]
        """
        return self.embedding(t)


class ChannelAwareResidualBlock(nn.Module):
    """信道感知残差块
    在标准残差块基础上增加信道信息条件
    CDDM关键组件：将信道估计信息融入噪声预测
    """
    def __init__(self, in_c: int, out_c: int, channel_embed_dim: int = 32):
        super().__init__()
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(in_c, out_c, 3, 1, 1)  # 1D卷积，保持尺寸
        self.bn1 = nn.BatchNorm1d(out_c)               # 批归一化
        self.activation1 = nn.ReLU()                   # 激活函数
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.activation2 = nn.ReLU()
        
        # 信道信息投影层 - CDDM核心：将信道信息映射到特征空间
        self.channel_proj = nn.Sequential(
            nn.Linear(channel_embed_dim, out_c),  # 线性变换
            nn.ReLU(),                            # 非线性激活
            nn.Linear(out_c, out_c)               # 再次变换
        )
        
        # 捷径连接：处理输入输出通道数不匹配的情况
        if in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_c, out_c, 1),        # 1x1卷积调整通道
                nn.BatchNorm1d(out_c)
            ) 
        else:
            self.shortcut = nn.Identity()         # 恒等映射

    def forward(self, input, channel_embed=None):
        """前向传播
        参数:
            input: 输入特征 [batch, in_c, length]
            channel_embed: 信道嵌入 [batch, channel_embed_dim]
        返回:
            输出特征 [batch, out_c, length]
        """
        # 第一层卷积
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.activation1(x)
        
        # 第二层卷积
        x = self.conv2(x)
        x = self.bn2(x)
        
        # CDDM关键步骤：加入信道信息条件
        if channel_embed is not None:
            # 将信道信息投影并调整形状以匹配特征图
            channel_effect = self.channel_proj(channel_embed).unsqueeze(-1)  # [batch, out_c, 1]
            x = x + channel_effect  # 广播相加
        
        # 残差连接
        x += self.shortcut(input)
        x = self.activation2(x)
        return x


class CDDMConvNet(nn.Module):
    """CDDM卷积网络
    基于ConvNet架构，专门为CDDM任务设计
    同时处理：加噪信号 + 时间步信息 + 信道信息
    """
    def __init__(self,
                 n_steps,
                 intermediate_channels=[10, 20, 40],
                 pe_dim=10,
                 channel_embed_dim=32,
                 insert_t_to_all_layers=False):
        super().__init__()

        C, H = get_signal_shape()
        self.pe = PositionalEncoding(n_steps, pe_dim)  # 时间步位置编码
        
        # 信道信息嵌入层 - 将原始信道信息映射到嵌入空间
        self.channel_embed = nn.Sequential(
            nn.Linear(H, 64),                    # 输入: 信号长度维度
            nn.ReLU(),
            nn.Linear(64, channel_embed_dim)     # 输出: 信道嵌入维度
        )

        # 时间步信息处理
        self.pe_linears = nn.ModuleList()
        self.all_t = insert_t_to_all_layers
        
        # 如果不将所有层都加入时间信息，只在第一层加入
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, C))

        # 构建残差块序列
        self.residual_blocks = nn.ModuleList()
        prev_channel = C
        for channel in intermediate_channels:
            # 每个残差块都能接收信道信息
            self.residual_blocks.append(
                ChannelAwareResidualBlock(prev_channel, channel, channel_embed_dim)
            )
            # 如果要在所有层加入时间信息，为每层创建线性变换
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
            
        # 输出层：将特征映射回原始信号空间
        self.output_layer = nn.Conv1d(prev_channel, C, 3, 1, 1)  

    def forward(self, x, t, h_r):
        """CDDM前向传播
        参数:
            x: 加噪信号 [batch, 2, 48] - 当前时间步的含噪信号
            t: 时间步 [batch, 1] - 扩散过程的时间步
            h_r: 信道信息 [batch, 48] - 实数值信道响应
        返回:
            预测的噪声 [batch, 2, 48] - 网络预测的加入噪声
        """
        n = t.shape[0]
        t_embed = self.pe(t)  # 获取时间步的位置编码
        
        # 处理信道信息：将原始信道响应映射到嵌入空间
        channel_embed = self.channel_embed(h_r)  # [batch, channel_embed_dim]
        
        # 逐层处理
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            # 如果该层需要加入时间信息
            if m_t is not None:
                pe = m_t(t_embed).reshape(n, -1, 1)  # 调整形状以匹配特征图
                x = x + pe  # 加入时间信息
            
            # 传递信道信息到残差块 - CDDM核心操作
            x = m_x(x, channel_embed)
            
        # 最终输出层
        x = self.output_layer(x)
        return x


class CDDMUnetBlock(nn.Module):
    """CDDM UNet块
    在标准UNet块基础上增加信道信息条件
    用于构建CDDM的UNet架构
    """
    def __init__(self, shape, in_c, out_c, channel_embed_dim=32, residual=False):
        super().__init__()
        self.ln = nn.LayerNorm(shape)  # 层归一化，稳定训练
        self.conv1 = nn.Conv1d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual  # 是否使用残差连接
        
        # 信道信息投影
        self.channel_proj = nn.Sequential(
            nn.Linear(channel_embed_dim, out_c),
            nn.ReLU(),
            nn.Linear(out_c, out_c)
        )
        
        # 残差连接
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(in_c, out_c, 1)

    def forward(self, x, channel_embed=None):
        """前向传播"""
        out = self.ln(x)  # 层归一化
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        
        # 加入信道信息条件
        if channel_embed is not None:
            channel_effect = self.channel_proj(channel_embed).unsqueeze(-1)
            out = out + channel_effect
            
        # 残差连接
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out


class CDDMUNet(nn.Module):
    """CDDM UNet网络
    基于UNet架构的CDDM实现，包含编码器-解码器结构
    适合处理需要多尺度信息的信号去噪任务
    """
    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 channel_embed_dim=32,
                 residual=False) -> None:
        super().__init__()
        
        C, H = get_signal_shape()
        layers = len(channels)
        
        # 计算各层特征图的高度（长度）
        Hs = [H]
        cH = H
        for _ in range(layers - 1):
            cH //= 2  # 下采样，长度减半
            Hs.append(cH)

        self.pe = PositionalEncoding(n_steps, pe_dim)
        
        # 信道信息嵌入
        self.channel_embed = nn.Sequential(
            nn.Linear(H, 64),
            nn.ReLU(),
            nn.Linear(64, channel_embed_dim)
        )

        # 编码器部分
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()  # 编码器时间步线性层
        self.pe_linears_de = nn.ModuleList()  # 解码器时间步线性层
        self.downs = nn.ModuleList()          # 下采样层
        self.ups = nn.ModuleList()            # 上采样层
        
        # 构建编码器
        prev_channel = C
        for channel, cH in zip(channels[0:-1], Hs[0:-1]):
            # 时间步信息处理
            self.pe_linears_en.append(
                nn.Sequential(
                    nn.Linear(pe_dim, prev_channel), 
                    nn.ReLU(),
                    nn.Linear(prev_channel, prev_channel)
                )
            )
            # 编码器块序列
            self.encoders.append(
                nn.Sequential(
                    CDDMUnetBlock((prev_channel, cH), prev_channel, channel, 
                                 channel_embed_dim, residual=residual),
                    CDDMUnetBlock((channel, cH), channel, channel, 
                                 channel_embed_dim, residual=residual)
                )
            )
            # 下采样：长度减半，通道数不变
            self.downs.append(nn.Conv1d(channel, channel, 2, 2))
            prev_channel = channel

        # 中间层（瓶颈层）
        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            CDDMUnetBlock((prev_channel, Hs[-1]), prev_channel, channel, 
                         channel_embed_dim, residual=residual),
            CDDMUnetBlock((channel, Hs[-1]), channel, channel, 
                         channel_embed_dim, residual=residual),
        )
        prev_channel = channel
        
        # 构建解码器
        for channel, cH in zip(channels[-2::-1], Hs[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            # 上采样：长度加倍，通道数减少
            self.ups.append(nn.ConvTranspose1d(prev_channel, channel, 2, 2))
            # 解码器块序列
            self.decoders.append(
                nn.Sequential(
                    CDDMUnetBlock((channel * 2, cH), channel * 2, channel, 
                                 channel_embed_dim, residual=residual),
                    CDDMUnetBlock((channel, cH), channel, channel, 
                                 channel_embed_dim, residual=residual)
                )
            )
            prev_channel = channel

        # 输出层
        self.conv_out = nn.Conv1d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t, h_r):
        """CDDM UNet前向传播
        参数:
            x: 加噪信号 [batch, 2, 48]
            t: 时间步 [batch, 1] 
            h_r: 信道信息 [batch, 48]
        返回:
            预测的噪声 [batch, 2, 48]
        """
        n = t.shape[0]
        t_embed = self.pe(t)
        
        # 处理信道信息
        channel_embed = self.channel_embed(h_r)  # [batch, channel_embed_dim]
        
        # 编码器路径
        encoder_outs = []  # 保存编码器输出用于跳跃连接
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders, self.downs):
            # 加入时间信息
            pe = pe_linear(t_embed).reshape(n, -1, 1)
            x = encoder[0](x + pe, channel_embed)  # 第一个UNet块
            x = encoder[1](x, channel_embed)       # 第二个UNet块
            encoder_outs.append(x)                 # 保存用于跳跃连接
            x = down(x)                           # 下采样
        
        # 中间层（瓶颈）
        pe = self.pe_mid(t_embed).reshape(n, -1, 1)
        x = self.mid[0](x + pe, channel_embed)
        x = self.mid[1](x, channel_embed)
        
        # 解码器路径
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de, self.decoders, 
                                                     self.ups, encoder_outs[::-1]):
            pe = pe_linear(t_embed).reshape(n, -1, 1)
            x = up(x)  # 上采样

            # 填充处理尺寸不匹配
            pad_x = encoder_out.shape[2] - x.shape[2]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2))
            
            # 跳跃连接：合并编码器和解码器特征
            x = torch.cat((encoder_out, x), dim=1)
            
            # 解码器块处理
            x = decoder[0](x + pe, channel_embed)
            x = decoder[1](x, channel_embed)
            
        # 最终输出
        x = self.conv_out(x)
        return x


# ============================ 网络配置 ============================

# CDDM专用配置 - 不同复杂度的网络架构

cddm_convnet_small_cfg = {
    'type': 'CDDMConvNet',
    'intermediate_channels': [10, 20],  # 中间层通道数
    'pe_dim': 128,                      # 位置编码维度
    'channel_embed_dim': 32             # 信道嵌入维度
}

cddm_convnet_medium_cfg = {
    'type': 'CDDMConvNet',
    'intermediate_channels': [10, 10, 20, 20, 40, 40, 80, 80],  # 更深的网络
    'pe_dim': 256,
    'channel_embed_dim': 64,
    'insert_t_to_all_layers': True  # 在所有层加入时间信息
}

cddm_unet_cfg = {
    'type': 'CDDMUNet', 
    'channels': [10, 20, 40, 80],  # UNet各层通道数
    'pe_dim': 128,
    'channel_embed_dim': 32
}

cddm_unet_res_cfg = {
    'type': 'CDDMUNet',
    'channels': [10, 20, 40, 80],
    'pe_dim': 128,
    'channel_embed_dim': 32,
    'residual': True  # 使用残差连接
}

def build_cddm_network(config: dict, n_steps):
    """构建CDDM网络工厂函数
    参数:
        config: 网络配置字典
        n_steps: 总扩散步数
    返回:
        CDDM网络实例
    """
    network_type = config.pop('type')
    if network_type == 'CDDMConvNet':
        network_cls = CDDMConvNet
    elif network_type == 'CDDMUNet':
        network_cls = CDDMUNet
    else:
        raise ValueError(f"Unknown network type: {network_type}")

    network = network_cls(n_steps, **config)
    return network


# ============================ 测试函数 ============================

def test_cddm_network():
    """测试CDDM网络功能
    验证网络输入输出形状和基本功能
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 测试CDDM ConvNet
    print("Testing CDDM ConvNet...")
    net = build_cddm_network(cddm_convnet_small_cfg, n_steps=1000)
    net.to(device)
    
    batch_size = 4
    x = torch.randn(batch_size, 2, 48).to(device)  # 加噪信号 [batch, 2, 48]
    t = torch.randint(0, 1000, (batch_size, 1)).to(device)  # 时间步 [batch, 1]
    h_r = torch.randn(batch_size, 48).to(device)  # 信道信息 [batch, 48]
    
    output = net(x, t, h_r)
    print(f"CDDM ConvNet:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Time steps:   {t.shape}")
    print(f"  Channel info: {h_r.shape}")
    print(f"  Output shape: {output.shape}")
    
    # 测试CDDM UNet
    print("\nTesting CDDM UNet...")
    net_unet = build_cddm_network(cddm_unet_cfg, n_steps=1000)
    net_unet.to(device)
    
    output_unet = net_unet(x, t, h_r)
    print(f"CDDM UNet:")
    print(f"  Input shape:  {x.shape}")
    print(f"  Output shape: {output_unet.shape}")
    
    # 验证输入输出一致性
    assert output.shape == x.shape, "Output shape should match input shape"
    assert output_unet.shape == x.shape, "UNet output shape should match input shape"
    print("\nAll tests passed! Network architectures are correct.")


if __name__ == '__main__':
    test_cddm_network()