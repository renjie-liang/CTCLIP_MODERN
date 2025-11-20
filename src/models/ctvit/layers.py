"""
CTViT Basic Layer Components

Contains:
- Helper functions
- LayerNorm (Layer Normalization)
- GEGLU activation (Gated activation function)
- FeedForward (Feedforward network)
- PEG (Position Encoding Generator)
"""

import math
import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange
from beartype import beartype
from typing import Tuple, Optional


# ============================================================================
# Helper Functions
# ============================================================================

def exists(val):
    """Check if value exists (not None)"""
    return val is not None


def default(val, d):
    """If value doesn't exist, return default value"""
    return val if exists(val) else d


def pair(val):
    """
    Convert single value to pair
    Example: 480 -> (480, 480)
    """
    ret = (val, val) if not isinstance(val, tuple) else val
    assert len(ret) == 2
    return ret


def leaky_relu(p=0.1):
    """Create LeakyReLU activation function"""
    return nn.LeakyReLU(p)


def l2norm(t):
    """
    L2 normalization (along last dimension)
    Used for QK normalization, improves training stability
    """
    return F.normalize(t, dim=-1)


# ============================================================================
# LayerNorm (Bias-less version)
# ============================================================================

class LayerNorm(nn.Module):
    """
    Bias-less LayerNorm (Layer Normalization without bias)

    Features:
    - Does not use bias parameter
    - Follows design of modern models like T5, PaLM
    - More stable training

    Modernization Opportunities:
    - RMSNorm: Faster, removes mean centering, only does RMS normalization
      Implementation: x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * gamma
      Performance gain: 5-10% speedup
    """

    def __init__(self, dim):
        super().__init__()
        # gamma: Learnable scaling parameter
        self.gamma = nn.Parameter(torch.ones(dim))
        # beta: Fixed bias at 0 (not learnable)
        self.register_buffer("beta", torch.zeros(dim))

    def forward(self, x):
        return F.layer_norm(x, x.shape[-1:], self.gamma, self.beta)


# ============================================================================
# GEGLU Activation (Gated activation function)
# ============================================================================

class GEGLU(nn.Module):
    """
    GEGLU (Gated GLU with GELU activation)

    Formula: GEGLU(x) = GELU(x_gate) * x_value
    where x_gate, x_value = x.chunk(2)

    Features:
    - Better performance than standard ReLU/GELU
    - Used in FeedForward network

    Modernization Opportunities:
    - SwiGLU: Use Swish activation instead of GELU
      Formula: Swish(x_gate) * x_value
      Reference: LLaMA, PaLM models
      Performance: Usually slightly better than GEGLU
    """

    def forward(self, x):
        # Split input into two halves: gate and value
        x, gate = x.chunk(2, dim=-1)
        # Activate gate with GELU, then multiply with value
        return F.gelu(gate) * x


# ============================================================================
# FeedForward Network
# ============================================================================

def FeedForward(dim, mult=4, dropout=0.):
    """
    FeedForward Network

    Architecture:
        LayerNorm → Linear(expand) → GEGLU → Dropout → Linear(compress)

    Args:
        dim: Input/output dimension
        mult: Hidden layer expansion multiplier (default 4x)
        dropout: Dropout ratio

    Inner dimension calculation:
        inner_dim = dim * mult * (2/3)
        - When mult=4, inner_dim ≈ 2.67 * dim
        - Multiply by 2 because GEGLU splits into two halves

    Modernization Opportunities:
    1. Use SwiGLU instead of GEGLU (see GEGLU class comments)
    2. Remove LayerNorm (some architectures like Pre-LN do it outside)
    3. Use different expansion multipliers (LLaMA uses 8/3≈2.67, GPT-3 uses 4)
    """
    inner_dim = int(mult * (2 / 3) * dim)

    return nn.Sequential(
        nn.LayerNorm(dim),                      # Normalization
        nn.Linear(dim, inner_dim * 2, bias=False),  # Expand (×2 because GEGLU splits)
        GEGLU(),                                # Gated activation
        nn.Dropout(dropout),                    # Dropout
        nn.Linear(inner_dim, dim, bias=False)   # Compress back to original dimension
    )


# ============================================================================
# PEG (Position Encoding Generator)
# ============================================================================

class PEG(nn.Module):
    """
    PEG (Position Encoding Generator)

    Generates position encoding using 3D depthwise separable convolution

    Features:
    - Dynamically generates position encoding (not fixed)
    - Uses groups=dim convolution (each channel independent)
    - Supports causal padding (for temporal dimension)

    Working principle:
    1. Captures local positional information through 3x3x3 depthwise convolution
    2. Adds to original features, injecting positional information for each position

    Args:
        dim: Feature dimension
        causal: Whether to use causal padding (temporal dimension only looks at past)

    Modernization Opportunities:
    1. Use separable convolution (Depthwise-Separable Conv):
       - Conv3D(3x3x3) → Conv3D(3x1x1) + Conv3D(1x3x1) + Conv3D(1x1x3)
       - Significantly reduces parameters and computation

    2. Option to disable PEG:
       - If using other position encodings like RoPE, PEG may not be needed
       - PEG improvement is limited for certain tasks

    3. Use lighter MLP:
       - Replace convolution with small MLP to generate position encoding
    """

    def __init__(self, dim, causal=False):
        super().__init__()
        self.causal = causal
        # 3D depthwise separable convolution (each channel independent, groups=dim)
        self.dsconv = nn.Conv3d(dim, dim, 3, groups=dim)

    @beartype
    def forward(self, x, shape: Tuple[int, int, int, int] = None):
        """
        Args:
            x: Input features (B, N, D) or (B, T, H, W, D)
            shape: If input is (B, N, D), need to provide original shape (B, T, H, W)

        Returns:
            Position-encoded features
        """
        needs_shape = x.ndim == 3
        assert not (needs_shape and not exists(shape))

        orig_shape = x.shape

        # If flattened, reshape back
        if needs_shape:
            x = x.reshape(*shape, -1)

        # Rearrange dimensions: (B, T, H, W, D) -> (B, D, T, H, W)
        x = rearrange(x, 'b ... d -> b d ...')

        # Padding strategy
        # Spatial dimensions (H, W): pad 1 on each side -> (1, 1, 1, 1)
        # Temporal dimension (T): depends on causal
        #   - causal=True: only pad front -> (2, 0) only look at past
        #   - causal=False: pad 1 on each side -> (1, 1)
        frame_padding = (2, 0) if self.causal else (1, 1)
        x = F.pad(x, (1, 1, 1, 1, *frame_padding), value=0.)

        # Apply 3D convolution
        x = self.dsconv(x)

        # Rearrange back to original dimension order
        x = rearrange(x, 'b d ... -> b ... d')

        # If originally flattened, flatten back
        if needs_shape:
            x = rearrange(x, 'b ... d -> b (...) d')

        return x.reshape(orig_shape)
