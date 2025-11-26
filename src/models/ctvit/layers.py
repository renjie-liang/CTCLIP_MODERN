"""
CTViT 基础层组件 (Basic Layer Components)

包含：
- Helper functions (辅助函数)
- LayerNorm (层归一化 - baseline)
- RMSNorm (层归一化 - optimized)
- GEGLU activation (门控激活函数 - baseline)
- SwiGLU activation (门控激活函数 - optimized)
- FeedForward (前馈网络 - configurable)
- PEG (位置编码生成器)
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from beartype import beartype
from typing import Tuple, Optional



# ============================================================================
# Helper Functions (辅助函数)
# ============================================================================

def exists(val):
    """检查值是否存在 (不为None)"""
    return val is not None


def default(val, d):
    """如果值不存在，返回默认值"""
    return val if exists(val) else d


def pair(val):
    """
    将单个值转换为pair
    例如: 480 -> (480, 480)
    """
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def leaky_relu(p=0.1):
    """创建LeakyReLU激活函数"""
    return nn.LeakyReLU(p)


def l2norm(t):
    """
    L2归一化 (沿最后一个维度)
    用于QK归一化，提升训练稳定性
    """
    return F.normalize(t, dim=-1)


# ============================================================================
# Normalization Layers
# ============================================================================

class LayerNorm(nn.Module):
    """
    Bias-less LayerNorm (Layer Normalization without bias)

    Features:
    - Does not use bias parameter
    - Follows design of modern models like T5, PaLM
    - More stable training (baseline implementation)
    """

    def __init__(self, dim):
        super().__init__()
        # gamma: Learnable scaling parameter
        self.gamma = nn.Parameter(torch.ones(dim))
        # beta: Fixed bias at 0 (not learnable)
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


class RMSNorm(nn.Module):
    """
    RMSNorm (Root Mean Square Normalization)

    Faster than LayerNorm - removes mean centering step.
    Used in modern models like LLaMA, T5.

    Performance: 5-10% speedup vs LayerNorm
    """

    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        rms = x.pow(2).mean(-1, keepdim=True).sqrt()
        return x * (self.weight / (rms + self.eps))


# ============================================================================
# Activation Functions
# ============================================================================

class SwiGLU(nn.Module):
    """
    SwiGLU (Swish-Gated Linear Unit)

    Used in modern models like LLaMA, PaLM.
    Generally better performance than GEGLU.

    Formula: SwiGLU(x) = Swish(x_gate) * x_value
    where Swish(x) = x * sigmoid(x) = x * σ(x)
    """

    def forward(self, x):
        a, b = x.chunk(2, dim=-1)
        return F.silu(a) * b

# ============================================================================
# GEGLU Activation (门控激活函数)
# ============================================================================

class GEGLU(nn.Module):
    """
    GEGLU (Gated GLU with GELU activation)

    公式: GEGLU(x) = GELU(x_gate) * x_value
    其中 x_gate, x_value = x.chunk(2)

    特点：
    - 比标准ReLU/GELU性能更好
    - 用于FeedForward网络

    🔧 [现代化改造点] 可以升级为：
    - SwiGLU: 使用Swish激活函数代替GELU
      公式: Swish(x_gate) * x_value
      参考: LLaMA, PaLM模型
      性能: 通常比GEGLU略好
    """

    def forward(self, x):
        # 将输入分成两半：gate和value
        x, gate = x.chunk(2, dim=-1)
        # 用GELU激活gate，然后与value相乘
        return F.gelu(gate) * x


# ============================================================================
# FeedForward Network (前馈网络)
# ============================================================================

def FeedForward(dim, mult=4, dropout=0., use_swiglu=False):
    """
    FeedForward Network

    Architecture:
        LayerNorm → Linear(expand) → Activation → Dropout → Linear(compress)

    Args:
        dim: Input/output dimension
        mult: Hidden layer expansion multiplier (default 4x)
        dropout: Dropout ratio
        use_swiglu: Use SwiGLU instead of GEGLU (default False for baseline)

    Inner dimension calculation:
        inner_dim = dim * mult * (2/3)
        - When mult=4, inner_dim ≈ 2.67 * dim
        - Multiply by 2 because gated activations split into two halves
    """
    inner_dim = int(mult * (2 / 3) * dim)

    # Choose activation function
    activation = SwiGLU() if use_swiglu else GEGLU()

    return nn.Sequential(
        nn.LayerNorm(dim),                          # Normalization
        nn.Linear(dim, inner_dim * 2, bias=False),  # Expand (×2 for gating)
        activation,                                  # Gated activation
        nn.Dropout(dropout),                        # Dropout
        nn.Linear(inner_dim, dim, bias=False)       # Compress back
    )


# ============================================================================
# PEG (Position Encoding Generator)
# ============================================================================

class PEG(nn.Module):
    """
    PEG (Position Encoding Generator) - 位置编码生成器

    使用3D深度可分离卷积生成位置编码

    特点：
    - 动态生成位置编码（不是固定的）
    - 使用groups=dim的卷积（每个通道独立）
    - 支持因果padding（用于时间维度）

    工作原理：
    1. 通过3x3x3的深度卷积捕获局部位置信息
    2. 与原始特征相加，为每个位置注入位置信息

    Args:
        dim: 特征维度
        causal: 是否使用因果padding（时间维度只看过去）

    🔧 [现代化改造点] 可以优化为：
    1. 使用可分离卷积 (Depthwise-Separable Conv):
       - Conv3D(3x3x3) → Conv3D(3x1x1) + Conv3D(1x3x1) + Conv3D(1x1x3)
       - 参数量和计算量大幅减少

    2. 可选择禁用PEG:
       - 如果使用RoPE等其他位置编码，可能不需要PEG
       - 某些任务下PEG提升有限

    3. 使用更轻量的MLP:
       - 用小型MLP代替卷积生成位置编码
    """

    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        # 3D深度可分离卷积 (每个通道独立，groups=dim)
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        """
        Args:
            x: 输入特征 (B, N, D) 或 (B, T, H, W, D)
            shape: 如果输入是(B, N, D)，需要提供原始形状(B, T, H, W)

        Returns:
            位置编码后的特征
        """
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        # 如果是flatten的，先reshape回来
        if needs_shape:
            x = x.reshape(*shape, -1)

        # 转换维度顺序: (B, T, H, W, D) -> (B, D, T, H, W)
        x = rearrange(x, 'b ... d -> b d ...')

        # Padding策略
        # 空间维度(H, W): 两边各padding 1 -> (1, 1, 1, 1)
        # 时间维度(T): 根据causal选择
        #   - causal=True: 只padding前面 -> (2, 0) 只看过去
        #   - causal=False: 两边各padding 1 -> (1, 1)
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.)

        # 应用3D卷积
        x = self.dsconv(x)

        # 转回原来的维度顺序
        x = rearrange(x, 'b d ... -> b ... d')

        # 如果原来是flatten的，flatten回去
        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)
