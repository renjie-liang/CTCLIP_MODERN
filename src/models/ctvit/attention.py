"""
CTViT 注意力模块 (Attention Modules)

包含：
- Attention (多头自注意力)
- AlibiPositionalBias (ALiBi位置偏置)
- ContinuousPositionBias (连续位置偏置)
- Transformer (完整Transformer块)
"""

import math
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from beartype import beartype
from typing import Tuple, Optional

from .layers import (
    exists, default, leaky_relu, l2norm,
    LayerNorm, RMSNorm, GEGLU, SwiGLU, FeedForward, PEG
)
from flash_attn.flash_attn_interface import (
    flash_attn_varlen_qkvpacked_func
)

from flash_attn.flash_attn_interface import flash_attn_qkvpacked_func

class FlashAttentionQKV(nn.Module):
    """
    Clean FlashAttention v2 module for CT-ViT.
    """

    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        num_null_kv=0,
        dropout=0.
    ):
        super().__init__()
        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = dim_head * heads
        self.causal = causal
        self.num_null_kv = num_null_kv
        self.dropout = dropout

        dim_context = dim if dim_context is None else dim_context

        self.norm = nn.LayerNorm(dim)
        self.context_norm = nn.LayerNorm(dim_context)

        self.to_q = nn.Linear(dim, self.inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, self.inner_dim * 2, bias=False)

        # null kv
        self.null_kv = nn.Parameter(torch.randn(heads, num_null_kv, 2, dim_head))

        self.to_out = nn.Linear(self.inner_dim, dim, bias=False)

    def forward(self, x, mask=None, context=None):
        b, n, device = x.shape[0], x.shape[1], x.device

        if context is not None:
            context = self.context_norm(context)

        kv_input = context if context is not None else x
        x = self.norm(x)

        # project
        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # reshape: (b, n, h*d) -> (b, n, h, d)
        q = q.view(b, -1, self.heads, self.dim_head)
        k = k.view(b, -1, self.heads, self.dim_head)
        v = v.view(b, -1, self.heads, self.dim_head)

        # add null kv if exists
        if self.num_null_kv > 0:
            nk = self.null_kv[:, :, 0, :]  # (H, N, D)
            nv = self.null_kv[:, :, 1, :]

            # Reshape to (B, num_null, H, D) for concatenation
            nk = nk.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)  # (B, N, H, D)
            nv = nv.permute(1, 0, 2).unsqueeze(0).expand(b, -1, -1, -1)

            k = torch.cat((nk, k), dim=1)  # concat on sequence dim
            v = torch.cat((nv, v), dim=1)

        # For flash_attn_qkvpacked_func, q/k/v must have same sequence length
        # When using null_kv, we need to use the unpacked version instead
        if self.num_null_kv > 0:
            # Use unpacked version for different q/kv lengths
            from flash_attn import flash_attn_func
            out = flash_attn_func(
                q.half(),
                k.half(),
                v.half(),
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal
            )
        else:
            # pack to qkv for flash-attn: (b, seqlen, 3, h, d)
            qkv = torch.stack([q, k, v], dim=2)

            out = flash_attn_qkvpacked_func(
                qkv.half(),
                dropout_p=self.dropout if self.training else 0.0,
                causal=self.causal
            )

        # reshape back
        out = out.view(b, -1, self.inner_dim)
        return self.to_out(out.to(x.dtype))


# ============================================================================
# Attention (多头自注意力)
# ============================================================================

class Attention(nn.Module):
    """
    Multi-Head Self-Attention (多头自注意力机制)

    特点：
    1. QK Normalization: Q和K向量进行L2归一化，提升训练稳定性
    2. Learnable Scale: 为Q和K添加可学习的缩放参数
    3. Null Key-Value: 额外的可学习KV对，增强表达能力
    4. 支持Cross-Attention: 可接受外部context
    5. 支持Causal Attention: 用于自回归生成

    计算流程:
        1. RMSNorm(x) -> Q, K, V
        2. L2 Normalize Q, K
        3. Attention = softmax(Q @ K^T * scale) @ V
        4. Linear projection

    Args:
        dim: 输入特征维度
        dim_context: Context维度 (用于cross-attention)
        dim_head: 每个注意力头的维度 (默认64)
        heads: 注意力头数 (默认8)
        causal: 是否使用因果注意力 (默认False)
        num_null_kv: Null key-value对的数量 (默认0)
        norm_context: 是否对context进行归一化 (默认True)
        dropout: Dropout比率 (默认0)
        scale: 注意力缩放因子 (默认8)

    🔧 [现代化改造点] 可以升级为：
    1. Flash Attention 2.0:
       - 使用融合CUDA kernel，大幅减少内存访问
       - 加速2-4倍，支持更长序列
       - 实现: 替换 einsum + softmax 为 flash_attn_func()

    2. Grouped-Query Attention (GQA):
       - 多个Query head共享一组KV head
       - 减少KV cache，加速推理
       - 例如: 8个Q head, 2个KV head (4:1比例)

    3. Multi-Query Attention (MQA):
       - 所有Query head共享1组KV
       - 最大化推理速度

    4. Sliding Window Attention:
       - 只关注局部窗口，减少计算复杂度
       - 适合超长序列
    """

    def __init__(
        self,
        dim,
        dim_context=None,
        dim_head=64,
        heads=8,
        causal=False,
        num_null_kv=0,
        norm_context=True,
        dropout=0.,
        scale=8,
        use_rms_norm=False
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        # 如果是因果注意力，使用ALiBi位置偏置
        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.attn_dropout = nn.Dropout(dropout)

        # Normalization layers (configurable: LayerNorm for baseline, RMSNorm for optimized)
        norm_class = RMSNorm if use_rms_norm else LayerNorm
        self.norm = norm_class(dim)
        self.context_norm = norm_class(dim_context) if norm_context else nn.Identity()

        # Null Key-Value pairs (额外的可学习KV，增强表达能力)
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        # Q, K, V projection
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)

        # QK Normalization的可学习缩放参数
        # 提升训练稳定性，防止softmax饱和
        self.q_scale = nn.Parameter(torch.ones(dim_head))
        self.k_scale = nn.Parameter(torch.ones(dim_head))

        # Output projection
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(
        self,
        x,
        mask=None,
        context=None,
        attn_bias=None
    ):
        """
        Args:
            x: 输入特征 (B, N, D)
            mask: 注意力mask (B, N) - True表示保留，False表示mask掉
            context: 外部context用于cross-attention (B, M, D_ctx)
            attn_bias: 额外的注意力偏置 (H, N, N) 如位置编码

        Returns:
            输出特征 (B, N, D)
        """
        batch, device, dtype = x.shape[0], x.device, x.dtype

        # Normalize context (如果有)
        if exists(context):
            context = self.context_norm(context)

        # 选择KV来源: context (cross-attn) 或 x (self-attn)
        kv_input = default(context, x)

        # Normalize input
        x = self.norm(x)

        # 计算 Q, K, V
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        # Reshape为多头: (B, N, H*D) -> (B, H, N, D)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # 添加Null Key-Value pairs

        if self.num_null_kv > 0:
            null = self.null_kv
            null = null.view(self.heads, self.num_null_kv, 2, self.dim_head)
            null = null.unsqueeze(0).expand(batch, -1, -1, -1, -1)

            nk = null[..., 0, :]
            nv = null[..., 1, :]

            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

        # QK Normalization (提升训练稳定性)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale  # 可学习缩放
        k = k * self.k_scale

        # 计算注意力分数: Q @ K^T
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]

        # 添加位置编码偏置 (如果有)
        # if exists(attn_bias):
        #     # 为null_kv部分padding 0
        #     attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)
        #     sim = sim + attn_bias

        # 应用attention mask (如果有)
        if exists(mask):
            # 为null_kv部分padding True (不mask)
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # mask掉的位置填充为-inf，softmax后变成0
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # 因果注意力mask (如果需要)
        if self.causal:
            # 添加ALiBi位置偏置
            sim = sim + self.rel_pos_bias(sim)
            # 创建上三角mask (只能看到过去和当前)
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # Softmax计算注意力权重
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # 应用注意力权重到V: Attention @ V
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # 合并多头: (B, H, N, D) -> (B, N, H*D)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # 输出投影
        return self.to_out(out)


# ============================================================================
# ALiBi Positional Bias (ALiBi位置偏置)
# ============================================================================

class AlibiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) 位置偏置

    论文: Train Short, Test Long: Attention with Linear Biases Enables
          Input Length Extrapolation

    原理:
        - 不使用位置编码，而是在attention score上添加线性偏置
        - 偏置随距离线性增长，距离越远惩罚越大
        - 每个注意力头使用不同的斜率 (slope)

    优点:
        1. 外推能力强：训练短序列，推理时可以处理更长序列
        2. 简单高效：不需要复杂的位置编码
        3. 无需额外参数

    公式:
        bias[i, j] = -slope * |i - j|
        其中slope对每个头不同，按2的幂次递减

    Args:
        heads: 注意力头数

    🔧 [现代化改造点] 相关替代方案：
    1. RoPE (Rotary Position Embedding):
       - 通过旋转变换编码位置信息
       - 外推能力也很好
       - 被LLaMA等模型采用

    2. xPos (Extrapolatable Position Embedding):
       - ALiBi的改进版
       - 更好的外推性能
    """

    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        # 计算每个头的slope
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        # 注册为buffer (不参与训练，但会随模型保存/加载)
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        """
        生成位置偏置矩阵

        Args:
            i: query序列长度
            j: key序列长度
            device: 设备

        Returns:
            bias: (1, i, j) - 位置偏置矩阵
        """
        # 生成position indices
        i_arange = torch.arange(j - i, j, device=device)  # query positions
        j_arange = torch.arange(j, device=device)          # key positions

        # 计算距离矩阵: |i - j|
        bias = -torch.abs(
            rearrange(j_arange, 'j -> 1 1 j') -
            rearrange(i_arange, 'i -> 1 i 1')
        )
        return bias

    @staticmethod
    def _get_slopes(heads):
        """
        计算每个注意力头的slope

        策略: 按2的幂次递减
            - 如果heads=8: slopes = [2^-1, 2^-2, ..., 2^-8]
        """
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        # 如果heads是2的幂
        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        # 如果不是，取最接近的2的幂，然后插值
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (get_slopes_power_of_2(closest_power_of_2) +
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2])

    def forward(self, sim):
        """
        Args:
            sim: 注意力分数 (B, H, i, j)

        Returns:
            ALiBi偏置 (H, i, j)
        """
        h, i, j, device = *sim.shape[-3:], sim.device

        # 如果已缓存且尺寸足够大，直接使用
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        # 生成bias
        bias = self.get_bias(i, j, device)
        # 乘以每个头的slope
        bias = bias * self.slopes

        # 如果heads数量大于已计算的bias头数，padding 0
        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))

        # 缓存起来
        self.register_buffer('bias', bias, persistent=False)

        return self.bias


# ============================================================================
# Continuous Position Bias (连续位置偏置)
# ============================================================================

class ContinuousPositionBias(nn.Module):
    """
    Continuous Position Bias (连续位置偏置)

    论文: "Conditional Positional Encodings for Vision Transformers"

    原理:
        - 使用小型MLP将相对位置坐标映射为注意力偏置
        - 支持2D (图像) 和 3D (视频) 位置编码
        - 使用对数距离编码，增强远距离建模

    结构:
        Relative Position Coords
        → MLP (Linear + LeakyReLU) × layers
        → Linear(heads)
        → Attention Bias

    Args:
        dim: MLP隐藏维度
        heads: 注意力头数
        num_dims: 位置维度数 (2=图像, 3=视频)
        layers: MLP层数
        log_dist: 是否使用对数距离 (默认True)
        cache_rel_pos: 是否缓存相对位置 (默认False)

    🔧 [现代化改造点] 相关替代方案：
    1. 2D RoPE: 将RoPE扩展到2D，为H和W维度分别应用旋转
    2. 可学习的2D Sinusoidal: 将固定sin/cos位置编码改为可学习
    3. 简化MLP: 减少层数或使用更轻量的网络
    """

    def __init__(
        self,
        *,
        dim,
        heads,
        num_dims=2,  # 2 for images, 3 for video
        layers=2,
        log_dist=True,
        cache_rel_pos=False
    ):
        super().__init__()
        self.num_dims = num_dims
        self.log_dist = log_dist

        # 构建MLP
        self.net = nn.ModuleList([])
        # 输入层: 从位置坐标(num_dims)映射到隐藏维度
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        # 中间层
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        # 输出层: 映射到每个注意力头
        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent=False)

    def forward(self, *dimensions, device=torch.device('cpu')):
        """
        Args:
            *dimensions: 各维度大小，例如 (H, W) 或 (T, H, W)
            device: 设备

        Returns:
            位置偏置 (H, i, j) 其中 i=j=H*W 或 T*H*W
        """
        # 如果未缓存或不使用缓存，重新计算
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            # 生成各维度的position indices
            # 例如: H=3, W=4 -> positions = [range(3), range(4)]
            positions = [torch.arange(d, device=device) for d in dimensions]

            # 生成网格坐标
            # grid.shape = (num_dims, *dimensions)
            # 例如: (2, 3, 4) -> [[0,0,0,0,1,1,1,1,2,2,2,2], [0,1,2,3,0,1,2,3,0,1,2,3]]
            grid = torch.stack(torch.meshgrid(*positions, indexing='ij'))
            grid = rearrange(grid, 'c ... -> (...) c')  # (HW, num_dims)

            # 计算相对位置: pos[i] - pos[j]
            # rel_pos.shape = (i, j, num_dims)
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            # 对数距离编码: sign(x) * log(|x| + 1)
            # 让远距离的区分度降低，更关注近距离
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            # 缓存
            self.register_buffer('rel_pos', rel_pos, persistent=False)

        # 转为float32 (MLP计算)
        rel_pos = self.rel_pos.to(torch.float32)

        # 通过MLP: (i, j, num_dims) -> (i, j, heads)
        for layer in self.net:
            rel_pos = layer(rel_pos.float())

        # 转换维度顺序: (i, j, heads) -> (heads, i, j)
        return rearrange(rel_pos, 'i j h -> h i j')


# ============================================================================
# Transformer (完整Transformer块)
# ============================================================================

class Transformer(nn.Module):
    """
    Transformer模块 (多层堆叠)

    结构 (每层):
        Input
        → PEG (位置编码, 可选)
        → Self-Attention + Residual
        → Cross-Attention + Residual (可选)
        → FeedForward + Residual
        → Output

    Args:
        dim: 特征维度
        depth: Transformer层数
        dim_context: Context维度 (用于cross-attention)
        causal: 是否使用因果注意力
        dim_head: 每个注意力头的维度
        heads: 注意力头数
        ff_mult: FeedForward扩展倍数
        peg: 是否使用PEG位置编码
        peg_causal: PEG是否使用因果padding
        attn_num_null_kv: Null key-value对数量
        has_cross_attn: 是否包含cross-attention
        attn_dropout: 注意力dropout
        ff_dropout: FeedForward dropout

    🔧 [现代化改造点] 整体架构优化：
    1. Pre-LN vs Post-LN:
       - 当前: Post-LN (LN在Attention内部)
       - 改为Pre-LN: LN(x) + Attn(...) 更稳定
       - 参考: GPT-3, LLaMA

    2. Parallel Attention + FFN:
       - 将Attention和FFN并行计算后相加
       - 加速10-15%，性能相当
       - 参考: PaLM

    3. MOE (Mixture of Experts):
       - 将FFN改为多个专家的混合
       - 增加参数量但保持计算量
       - 参考: Switch Transformer
    """

    def __init__(
        self,
        dim,
        *,
        depth,
        dim_context=None,
        causal=False,
        dim_head=64,
        heads=8,
        ff_mult=4,
        peg=False,
        peg_causal=False,
        attn_num_null_kv=2,
        has_cross_attn=False,
        attn_dropout=0.,
        ff_dropout=0.,
        # NEW: Optimization flags
        use_flash_attention=False,
        use_rms_norm=False,
        use_swiglu=False
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # Choose attention class based on optimization flag
        attn_class = FlashAttentionQKV if use_flash_attention else Attention

        # Choose normalization class based on optimization flag
        norm_class = RMSNorm if use_rms_norm else LayerNorm

        # 堆叠depth层
        for _ in range(depth):
            # Build attention layers with appropriate parameters
            if use_flash_attention:
                # FlashAttentionQKV doesn't need use_rms_norm (it has hardcoded LayerNorm)
                self_attn = FlashAttentionQKV(
                    dim=dim, dim_head=dim_head, heads=heads,
                    causal=causal, dropout=attn_dropout
                )
                cross_attn = FlashAttentionQKV(
                    dim=dim, dim_head=dim_head, dim_context=dim_context,
                    heads=heads, causal=False, num_null_kv=attn_num_null_kv,
                    dropout=attn_dropout
                ) if has_cross_attn else None
            else:
                # Attention class needs use_rms_norm parameter
                self_attn = Attention(
                    dim=dim, dim_head=dim_head, heads=heads,
                    causal=causal, dropout=attn_dropout,
                    use_rms_norm=use_rms_norm
                )
                cross_attn = Attention(
                    dim=dim, dim_head=dim_head, dim_context=dim_context,
                    heads=heads, causal=False, num_null_kv=attn_num_null_kv,
                    dropout=attn_dropout,
                    use_rms_norm=use_rms_norm
                ) if has_cross_attn else None

            self.layers.append(nn.ModuleList([
                # 1. PEG (位置编码生成器, 可选)
                PEG(dim=dim, causal=peg_causal) if peg else None,

                # 2. Self-Attention
                self_attn,

                # 3. Cross-Attention (可选)
                cross_attn,

                # 4. FeedForward (with configurable activation)
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout, use_swiglu=use_swiglu)
            ]))

        # Output normalization (configurable)
        self.norm_out = norm_class(dim)


    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None,
        attn_bias=None
    ):


        """
        Args:
            x: 输入特征 (B, N, D)
            video_shape: 用于PEG的形状 (B, T, H, W)
            attn_bias: 注意力偏置
            context: Cross-attention的context
            self_attn_mask: Self-attention的mask
            cross_attn_context_mask: Cross-attention的mask

        Returns:
            输出特征 (B, N, D)
        """
        # 遍历每一层
        for peg, self_attn, cross_attn, ff in self.layers:
            # 1. 位置编码 (如果有)
            if exists(peg):
                x = peg(x, shape=video_shape) + x

            # 2. Self-Attention + Residual
            x = self_attn(x, mask=self_attn_mask) + x


            # 3. Cross-Attention + Residual (如果有)
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x


            # 4. FeedForward + Residual
            x = ff(x) + x

        # 输出归一化
        return self.norm_out(x)
