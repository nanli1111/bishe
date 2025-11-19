import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
def get_signal_shape():
    """返回信号的形状定义。
    在 CDDM 中，通常是 2 个通道 (I 路和 Q 路)，长度为 48 的信号。
    返回值:
        tuple: (通道数, 信号长度)
    """
    return (2, 48)  # 2 个通道 (I 路和 Q 路)，长度为 48 的信号

class PositionalEncoding(nn.Module):
    """位置编码模块，用于为时间步 t 生成位置编码，帮助网络理解不同的扩散步骤。
    基于 Transformer 的 sin-cos 位置编码方法。
    """
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        
        assert d_model % 2 == 0  # 确保嵌入维度是偶数

        # 创建位置编码矩阵 [max_seq_len, d_model]
        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)  # 位置序列
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)    # 维度序列
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        
        # 计算 sin 和 cos 位置编码
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))      # 偶数维度: sin
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))    # 奇数维度: cos
        
        # 合并 sin 和 cos 编码
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        # 使用 Embedding 层存储位置编码
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)  # 冻结参数，不参与训练

    def forward(self, t):
        """前向传播，用于生成时间步对应的编码。
        参数:
            t: 时间步张量 [batch_size, 1]
        返回:
            位置编码 [batch_size, d_model]
        """
        return self.embedding(t)


class CDDMUnetBlock(nn.Module):
    """CDDM UNet块，在标准UNet块基础上增加信道信息条件
    用于构建 CDDM 的 UNet 架构
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
    """CDDM UNet 网络，基于 UNet 架构，包含编码器-解码器结构。
    适合处理需要多尺度信息的信号去噪任务。
    """
    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 channel_embed_dim=32,
                 residual=False) -> None:
        super().__init__()
        
        C, H = get_signal_shape()  # 获取信号形状
        layers = len(channels)  # 网络的层数
        
        # 计算各层特征图的高度（长度）
        Hs = [H]
        cH = H
        for _ in range(layers - 1):
            cH //= 2  # 下采样，长度减半
            Hs.append(cH)

        self.pe = PositionalEncoding(n_steps, pe_dim)  # 时间步位置编码
        
        # 信道信息嵌入
        self.channel_embed = nn.Sequential(
            nn.Linear(H, 64),  # 输入: 信号长度维度
            nn.ReLU(),
            nn.Linear(64, channel_embed_dim)  # 输出: 信道嵌入维度
        )

        # 编码器部分
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()  # 编码器时间步线性层
        self.pe_linears_de = nn.ModuleList()  # 解码器时间步线性层
        self.downs = nn.ModuleList()  # 下采样层
        self.ups = nn.ModuleList()  # 上采样层
        
        # 构建编码器
        prev_channel = C
        for channel, cH in zip(channels[0:-1], Hs[0:-1]):
            self.pe_linears_en.append(
                nn.Sequential(
                    nn.Linear(pe_dim, prev_channel), 
                    nn.ReLU(),
                    nn.Linear(prev_channel, prev_channel)
                )
            )
            self.encoders.append(
                nn.Sequential(
                    CDDMUnetBlock((prev_channel, cH), prev_channel, channel, 
                                 channel_embed_dim, residual=residual),
                    CDDMUnetBlock((channel, cH), channel, channel, 
                                 channel_embed_dim, residual=residual)
                )
            )
            self.downs.append(nn.Conv1d(channel, channel, 2, 2))  # 下采样
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
            self.ups.append(nn.ConvTranspose1d(prev_channel, channel, 2, 2))  # 上采样
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
        self.conv_out = nn.Conv1d(prev_channel, C, 3, 1, 1)  # 将特征映射回原始信号空间

    def forward(self, x, t, h_r):
        """CDDM UNet 前向传播
        参数:
            x: 加噪信号 [batch, 2, 48]
            t: 时间步 [batch, 1] 
            h_r: 信道信息 [batch, 48]
        返回:
            预测的噪声 [batch, 2, 48]
        """
        n = t.shape[0]
        t_embed = self.pe(t)  # 获取时间步的位置编码
        
        # 处理信道信息
        channel_embed = self.channel_embed(h_r)  # [batch, channel_embed_dim]
        
        # 编码器路径
        encoder_outs = []  # 保存编码器输出用于跳跃连接
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders, self.downs):
            pe = pe_linear(t_embed).reshape(n, -1, 1)
            x = encoder[0](x + pe, channel_embed)  # 第一个UNet块
            x = encoder[1](x, channel_embed)       # 第二个UNet块
            encoder_outs.append(x)  # 保存用于跳跃连接
            x = down(x)  # 下采样
        
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



def build_cddm_network(config: dict, n_steps: int):
    """构建 CDDM 网络工厂函数
    参数:
        config: 网络配置字典
        n_steps: 总扩散步数
    返回:
        CDDM 网络实例
    """
    network_type = config.pop('type')
    
    network_cls = CDDMUNet


    # 创建网络实例
    network = network_cls(n_steps, **config)
    return network


# ============================ 测试函数 ============================
if __name__ == '__main__':
    # 配置参数
    cddm_unet_cfg = {
        'type': 'CDDMUNet', 
        'channels': [10, 20, 40, 80],  # UNet各层通道数
        'pe_dim': 128,
        'channel_embed_dim': 32
    }

    # 创建网络实例
    n_steps = 1000  # 假设有1000步扩散过程
    model = build_cddm_network(cddm_unet_cfg, n_steps)

    # 模拟输入数据
    batch_size = 8  # 假设批量大小是8
    x = torch.randn(batch_size, 2, 48)  # 模拟的加噪信号 [batch_size, 2, 48]
    t = torch.randint(0, n_steps, (batch_size, 1))  # 随机生成时间步 [batch_size, 1]
    h_r = torch.randn(batch_size, 48)  # 随机生成信道信息 [batch_size, 48]

    # 使用 summary 展示网络结构
    summary(model, input_data=(x, t, h_r))  # 仅传递输入形状，不传递额外的 t 和 h_r