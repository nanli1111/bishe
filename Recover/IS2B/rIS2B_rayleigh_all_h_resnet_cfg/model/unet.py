import torch
import torch.nn as nn
import torch.nn.functional as F
# from dataset.dataset import get_signal_shape # 移除或仅作为默认值

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()
        assert d_model % 2 == 0
        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        
        # 修改这里：增加 indexing='ij'
        pos, two_i = torch.meshgrid(i_seq, j_seq, indexing='ij')
        
        pe_2i = torch.sin(pos / 10000**(two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000**(two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)
        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)
    
    def forward(self, t):
        return self.embedding(t)

class UnetBlock(nn.Module):
    def __init__(self, in_c, out_c, residual=False):
        super().__init__()
        
        # --- 修复 GroupNorm 问题 ---
        # 默认尝试使用 8 个组
        num_groups = 8
        # 如果通道数不能被 8 整除，就不断减半，直到能被整除（最小为1，即 LayerNorm）
        while num_groups > 1 and in_c % num_groups != 0:
            num_groups //= 2
            
        self.norm = nn.GroupNorm(num_groups, in_c)
        # -------------------------

        self.conv1 = nn.Conv1d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(in_c, out_c, 1)

    def forward(self, x):
        out = self.norm(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out

class UNet(nn.Module):
    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 residual=False,
                 in_channels=2,   # 新增：默认2，但支持传入6
                 out_channels=2) -> None: # 新增
        super().__init__()
        
        # 移除硬编码
        # C, H = get_signal_shape() 
        C = in_channels
        
        # 简单估算下采样次数，用于构建 UNet 层级
        layers = len(channels)
        
        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        
        prev_channel = C
        
        # --- Encoder 部分 ---
        for channel in channels[0:-1]:
            # 时间嵌入投影
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            
            # Encoder Block
            # 注意：移除了 shape 参数，因为长度是动态的
            self.encoders.append(
                nn.Sequential(
                    UnetBlock(prev_channel, channel, residual=residual),
                    UnetBlock(channel, channel, residual=residual)))
            
            # 下采样
            self.downs.append(nn.Conv1d(channel, channel, 2, 2))
            prev_channel = channel

        # --- Middle 部分 ---
        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock(prev_channel, channel, residual=residual),
            UnetBlock(channel, channel, residual=residual),
        )
        prev_channel = channel
        
        # --- Decoder 部分 ---
        for channel in channels[-2::-1]:
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose1d(prev_channel, channel, 2, 2))
            
            # Decoder Block (输入是 concat 后的，所以是 channel * 2)
            self.decoders.append(
                nn.Sequential(
                    UnetBlock(channel * 2, channel, residual=residual),
                    UnetBlock(channel, channel, residual=residual)))

            prev_channel = channel

        # 输出层：映射回 out_channels (通常是2，即I/Q噪声)
        self.conv_out = nn.Conv1d(prev_channel, out_channels, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        
        # Encoder
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders, self.downs):
            pe = pe_linear(t).reshape(n, -1, 1)
            # 时间嵌入加到特征上
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
            
        # Mid
        pe = self.pe_mid(t).reshape(n, -1, 1)
        x = self.mid(x + pe)
        
        # Decoder
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1)
            x = up(x)

            # 处理 Padding (如果下采样后长度不匹配)
            pad_x = encoder_out.shape[2] - x.shape[2]
            if pad_x != 0:
                x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2))
                
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
            
        x = self.conv_out(x)
        return x

def build_network(config: dict, n_steps):
    network_type = config.pop('type')
    # 提取并移除可能存在的 in/out channels 参数，防止重复
    in_c = config.pop('in_channels', 2)
    out_c = config.pop('out_channels', 2)
    
    if network_type == 'UNet':
        network = UNet(n_steps, in_channels=in_c, out_channels=out_c, **config)
    # ConvNet 类似修改...
    else:
        raise NotImplementedError(f"Unknown network type {network_type}")
        
    return network