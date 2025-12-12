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

class AdaGNResBlock1D(nn.Module):
    """ 带 AdaGN 的空洞残差块 (来自 Pro 版) """
    def __init__(self, channels, time_emb_dim, dilation=1, dropout=0.1):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.norm1 = nn.GroupNorm(8, channels)
        self.norm2 = nn.GroupNorm(8, channels)
        self.act = nn.GELU()
        self.time_proj = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, channels * 2))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t_emb):
        residual = x
        h = self.norm1(x)
        t_proj = self.time_proj(t_emb).unsqueeze(-1)
        scale, shift = t_proj.chunk(2, dim=1)
        h = h * (1 + scale) + shift
        h = self.act(h)
        h = self.conv1(h)
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        return h + residual

# ==========================================
# 2. 新增：一维多头自注意力模块
# ==========================================
class SelfAttention1D(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        # 归一化 (Pre-Norm)
        self.norm = nn.GroupNorm(8, channels)
        
        # Q, K, V 映射 (使用 1x1 卷积实现线性映射)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size=1, bias=False)
        
        # 输出映射
        self.proj_out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x):
        """
        x: [Batch, Channels, Length]
        """
        b, c, l = x.shape
        residual = x
        
        # 1. Norm
        x = self.norm(x)
        
        # 2. QKV Projection
        # qkv: [b, 3*c, l]
        qkv = self.qkv(x)
        # 分离 q, k, v -> [b, c, l]
        q, k, v = qkv.chunk(3, dim=1)
        
        # 3. Reshape for Multi-head Attention
        # [b, heads, head_dim, l]
        q = q.view(b, self.num_heads, self.head_dim, l)
        k = k.view(b, self.num_heads, self.head_dim, l)
        v = v.view(b, self.num_heads, self.head_dim, l)
        
        # 4. Attention Score: Attention(Q, K, V) = softmax(QK^T / sqrt(d)) * V
        # q: [..., l], k^T: [..., l, head_dim] -> attn: [..., l, l]
        # 这里的 @ 是矩阵乘法 (torch.matmul)
        scale = 1.0 / math.sqrt(self.head_dim)
        # [b, heads, l, l]
        attn_scores = torch.einsum('b h d i, b h d j -> b h i j', q, k) * scale
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        
        # 5. Aggregate V
        # [b, heads, l, l] @ [b, heads, head_dim, l] (viewed as l, d) -> 
        # Easier with einsum: weighted sum of v based on attn_probs
        # Result: [b, heads, head_dim, l]
        out = torch.einsum('b h i j, b h d j -> b h d i', attn_probs, v)
        
        # 6. Reshape back
        out = out.reshape(b, c, l)
        
        # 7. Output Projection
        out = self.proj_out(out)
        
        # 8. Residual Connection
        return out + residual

# ==========================================
# 3. 集成网络: AttentionTimeResNet1D
# ==========================================
class AttentionTimeResNet1D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, hidden_dim=128, num_blocks=8, time_emb_dim=128):
        super().__init__()
        
        # 时间嵌入
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.GELU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # 入口卷积
        self.entry = nn.Conv1d(in_channels, hidden_dim, kernel_size=3, padding=1)
        
        # 构建主干：卷积块与注意力块交替
        self.layers = nn.ModuleList()
        
        # 策略：每 3 个 ResBlock 插 1 个 Attention Block
        # 或者是只在中间插一个。这里采用间隔插入策略。
        for i in range(num_blocks):
            # 1. 正常的 ResBlock (带空洞)
            dilation = 2 ** (i % 4)
            self.layers.append(
                AdaGNResBlock1D(hidden_dim, time_emb_dim, dilation=dilation)
            )
            
            # 2. 在网络的中后段插入 Attention (例如第 4 层和第 8 层之后)
            # 这样既节省计算量，又能利用前面卷积提取的高级特征
            if (i + 1) % 4 == 0: 
                self.layers.append(
                    SelfAttention1D(hidden_dim, num_heads=4)
                )
        
        # 出口
        self.final_norm = nn.GroupNorm(8, hidden_dim)
        self.final_act = nn.GELU()
        self.exit = nn.Conv1d(hidden_dim, out_channels, kernel_size=1)
        
        # Zero Init
        self.exit.weight.data.zero_()
        self.exit.bias.data.zero_()

    def forward(self, x, t):
        # 1. Time Embedding
        t_emb = self.time_mlp(t.float())
        
        # 2. Entry
        h = self.entry(x.float())
        
        # 3. Backbone
        for layer in self.layers:
            if isinstance(layer, AdaGNResBlock1D):
                # 卷积块需要时间信息
                h = layer(h, t_emb)
            else:
                # Attention 块不需要时间信息(或者是隐式包含在特征里了)
                h = layer(h)
                
        # 4. Exit
        h = self.final_norm(h)
        h = self.final_act(h)
        out = self.exit(h)
        
        return out