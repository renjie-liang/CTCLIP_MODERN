"""
CTViT Attention Modules

Contains:
- Attention (Multi-Head Self-Attention)
- AlibiPositionalBias (ALiBi Positional Bias)
- ContinuousPositionBias (Continuous Position Bias)
- Transformer (Complete Transformer Block)
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
    LayerNorm, FeedForward, PEG
)


# ============================================================================
# Attention (Multi-Head Self-Attention)
# ============================================================================

class Attention(nn.Module):
    """
    Multi-Head Self-Attention Mechanism

    Features:
    1. QK Normalization: L2 normalization for Q and K vectors, improving training stability
    2. Learnable Scale: Learnable scaling parameters for Q and K
    3. Null Key-Value: Additional learnable KV pairs to enhance expressiveness
    4. Cross-Attention Support: Can accept external context
    5. Causal Attention Support: For autoregressive generation

    Computation Flow:
        1. LayerNorm(x) -> Q, K, V
        2. L2 Normalize Q, K
        3. Attention = softmax(Q @ K^T * scale) @ V
        4. Linear projection

    Args:
        dim: Input feature dimension
        dim_context: Context dimension (for cross-attention)
        dim_head: Dimension per attention head (default 64)
        heads: Number of attention heads (default 8)
        causal: Whether to use causal attention (default False)
        num_null_kv: Number of null key-value pairs (default 0)
        norm_context: Whether to normalize context (default True)
        dropout: Dropout ratio (default 0)
        scale: Attention scale factor (default 8)

    Modernization Opportunities:
    1. Flash Attention 2.0:
       - Uses fused CUDA kernels, significantly reducing memory access
       - 2-4x speedup, supports longer sequences
       - Implementation: Replace einsum + softmax with flash_attn_func()

    2. Grouped-Query Attention (GQA):
       - Multiple Query heads share one set of KV heads
       - Reduces KV cache, accelerates inference
       - Example: 8 Q heads, 2 KV heads (4:1 ratio)

    3. Multi-Query Attention (MQA):
       - All Query heads share 1 set of KV
       - Maximizes inference speed

    4. Sliding Window Attention:
       - Only attends to local windows, reducing computational complexity
       - Suitable for very long sequences
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
        scale=8
    ):
        super().__init__()
        self.heads = heads
        self.causal = causal
        self.scale = scale
        inner_dim = dim_head * heads
        dim_context = default(dim_context, dim)

        # If causal attention, use ALiBi positional bias
        if causal:
            self.rel_pos_bias = AlibiPositionalBias(heads=heads)

        self.attn_dropout = nn.Dropout(dropout)

        # Normalization layers
        self.norm = LayerNorm(dim)
        self.context_norm = LayerNorm(dim_context) if norm_context else nn.Identity()

        # Null Key-Value pairs (additional learnable KV to enhance expressiveness)
        self.num_null_kv = num_null_kv
        self.null_kv = nn.Parameter(torch.randn(heads, 2 * num_null_kv, dim_head))

        # Q, K, V projection
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_context, inner_dim * 2, bias=False)

        # Learnable scaling parameters for QK Normalization
        # Improves training stability, prevents softmax saturation
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
            x: Input features (B, N, D)
            mask: Attention mask (B, N) - True means keep, False means mask out
            context: External context for cross-attention (B, M, D_ctx)
            attn_bias: Additional attention bias (H, N, N) such as positional encoding

        Returns:
            Output features (B, N, D)
        """
        batch, device, dtype = x.shape[0], x.device, x.dtype

        # Normalize context (if exists)
        if exists(context):
            context = self.context_norm(context)

        # Choose KV source: context (cross-attn) or x (self-attn)
        kv_input = default(context, x)

        # Normalize input
        x = self.norm(x)

        # Compute Q, K, V
        q, k, v = self.to_q(x), *self.to_kv(kv_input).chunk(2, dim=-1)

        # Reshape to multi-head: (B, N, H*D) -> (B, H, N, D)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), (q, k, v))

        # Add Null Key-Value pairs
        # Split (H, 2*num_null_kv, D) into two (H, num_null_kv, D) parts
        nk, nv = repeat(self.null_kv, 'h (n r) d -> b h n r d', b=batch, r=2).unbind(dim=-2)

        # Concatenate to the front of K, V
        k = torch.cat((nk, k), dim=-2)  # (B, H, num_null_kv+N, D)
        v = torch.cat((nv, v), dim=-2)

        # QK Normalization (improves training stability)
        q, k = map(l2norm, (q, k))
        q = q * self.q_scale  # Learnable scaling
        k = k * self.k_scale

        # Compute attention scores: Q @ K^T
        sim = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        i, j = sim.shape[-2:]

        # Add positional encoding bias (if exists)
        if exists(attn_bias):
            # Pad null_kv part with 0
            attn_bias = F.pad(attn_bias, (self.num_null_kv, 0), value=0.)
            sim = sim + attn_bias

        # Apply attention mask (if exists)
        if exists(mask):
            # Pad null_kv part with True (don't mask)
            mask = F.pad(mask, (self.num_null_kv, 0), value=True)
            mask = rearrange(mask, 'b j -> b 1 1 j')
            # Fill masked positions with -inf, becomes 0 after softmax
            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        # Causal attention mask (if needed)
        if self.causal:
            # Add ALiBi positional bias
            sim = sim + self.rel_pos_bias(sim)
            # Create upper triangular mask (can only see past and current)
            causal_mask = torch.ones((i, j), device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, -torch.finfo(sim.dtype).max)

        # Softmax to compute attention weights
        attn = sim.softmax(dim=-1)
        attn = self.attn_dropout(attn)

        # Apply attention weights to V: Attention @ V
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # Merge multi-head: (B, H, N, D) -> (B, N, H*D)
        out = rearrange(out, 'b h n d -> b n (h d)')

        # Output projection
        return self.to_out(out)


# ============================================================================
# ALiBi Positional Bias
# ============================================================================

class AlibiPositionalBias(nn.Module):
    """
    ALiBi (Attention with Linear Biases) Positional Bias

    Paper: Train Short, Test Long: Attention with Linear Biases Enables
           Input Length Extrapolation

    Principle:
        - Instead of using position encoding, adds linear bias to attention scores
        - Bias increases linearly with distance, penalizing distant positions more
        - Each attention head uses a different slope

    Advantages:
        1. Strong extrapolation: Train on short sequences, can handle longer sequences at inference
        2. Simple and efficient: No complex positional encoding needed
        3. No additional parameters required

    Formula:
        bias[i, j] = -slope * |i - j|
        where slope is different for each head, decreasing by powers of 2

    Args:
        heads: Number of attention heads

    Modernization Opportunities - Alternative Approaches:
    1. RoPE (Rotary Position Embedding):
       - Encodes positional information through rotary transformations
       - Also has good extrapolation capability
       - Adopted by models like LLaMA

    2. xPos (Extrapolatable Position Embedding):
       - Improved version of ALiBi
       - Better extrapolation performance
    """

    def __init__(self, heads):
        super().__init__()
        self.heads = heads
        # Compute slopes for each head
        slopes = torch.Tensor(self._get_slopes(heads))
        slopes = rearrange(slopes, 'h -> h 1 1')
        # Register as buffer (not trainable, but saved/loaded with model)
        self.register_buffer('slopes', slopes, persistent=False)
        self.register_buffer('bias', None, persistent=False)

    def get_bias(self, i, j, device):
        """
        Generate positional bias matrix

        Args:
            i: Query sequence length
            j: Key sequence length
            device: Device

        Returns:
            bias: (1, i, j) - Positional bias matrix
        """
        # Generate position indices
        i_arange = torch.arange(j - i, j, device=device)  # query positions
        j_arange = torch.arange(j, device=device)          # key positions

        # Compute distance matrix: |i - j|
        bias = -torch.abs(
            rearrange(j_arange, 'j -> 1 1 j') -
            rearrange(i_arange, 'i -> 1 i 1')
        )
        return bias

    @staticmethod
    def _get_slopes(heads):
        """
        Compute slopes for each attention head

        Strategy: Decrease by powers of 2
            - If heads=8: slopes = [2^-1, 2^-2, ..., 2^-8]
        """
        def get_slopes_power_of_2(n):
            start = (2**(-2**-(math.log2(n)-3)))
            ratio = start
            return [start*ratio**i for i in range(n)]

        # If heads is a power of 2
        if math.log2(heads).is_integer():
            return get_slopes_power_of_2(heads)

        # If not, take the closest power of 2 and interpolate
        closest_power_of_2 = 2 ** math.floor(math.log2(heads))
        return (get_slopes_power_of_2(closest_power_of_2) +
                get_slopes_power_of_2(2 * closest_power_of_2)[0::2][:heads-closest_power_of_2])

    def forward(self, sim):
        """
        Args:
            sim: Attention scores (B, H, i, j)

        Returns:
            ALiBi bias (H, i, j)
        """
        h, i, j, device = *sim.shape[-3:], sim.device

        # If already cached and size is sufficient, use directly
        if exists(self.bias) and self.bias.shape[-1] >= j:
            return self.bias[..., :i, :j]

        # Generate bias
        bias = self.get_bias(i, j, device)
        # Multiply by each head's slope
        bias = bias * self.slopes

        # If number of heads is greater than computed bias heads, pad with 0
        num_heads_unalibied = h - bias.shape[0]
        bias = F.pad(bias, (0, 0, 0, 0, 0, num_heads_unalibied))

        # Cache it
        self.register_buffer('bias', bias, persistent=False)

        return self.bias


# ============================================================================
# Continuous Position Bias
# ============================================================================

class ContinuousPositionBias(nn.Module):
    """
    Continuous Position Bias

    Paper: "Conditional Positional Encodings for Vision Transformers"

    Principle:
        - Uses a small MLP to map relative position coordinates to attention bias
        - Supports 2D (image) and 3D (video) position encoding
        - Uses logarithmic distance encoding to enhance long-range modeling

    Architecture:
        Relative Position Coords
        → MLP (Linear + LeakyReLU) × layers
        → Linear(heads)
        → Attention Bias

    Args:
        dim: MLP hidden dimension
        heads: Number of attention heads
        num_dims: Number of position dimensions (2=image, 3=video)
        layers: Number of MLP layers
        log_dist: Whether to use logarithmic distance (default True)
        cache_rel_pos: Whether to cache relative positions (default False)

    Modernization Opportunities - Alternative Approaches:
    1. 2D RoPE: Extend RoPE to 2D, applying rotation separately for H and W dimensions
    2. Learnable 2D Sinusoidal: Make fixed sin/cos positional encoding learnable
    3. Simplified MLP: Reduce number of layers or use lighter network
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

        # Build MLP
        self.net = nn.ModuleList([])
        # Input layer: map from position coordinates (num_dims) to hidden dimension
        self.net.append(nn.Sequential(nn.Linear(self.num_dims, dim), leaky_relu()))

        # Hidden layers
        for _ in range(layers - 1):
            self.net.append(nn.Sequential(nn.Linear(dim, dim), leaky_relu()))

        # Output layer: map to each attention head
        self.net.append(nn.Linear(dim, heads))

        self.cache_rel_pos = cache_rel_pos
        self.register_buffer('rel_pos', None, persistent=False)

    def forward(self, *dimensions, device=torch.device('cpu')):
        """
        Args:
            *dimensions: Sizes of each dimension, e.g., (H, W) or (T, H, W)
            device: Device

        Returns:
            Positional bias (H, i, j) where i=j=H*W or T*H*W
        """
        # If not cached or not using cache, recompute
        if not exists(self.rel_pos) or not self.cache_rel_pos:
            # Generate position indices for each dimension
            # Example: H=3, W=4 -> positions = [range(3), range(4)]
            positions = [torch.arange(d, device=device) for d in dimensions]

            # Generate grid coordinates
            # grid.shape = (num_dims, *dimensions)
            # Example: (2, 3, 4) -> [[0,0,0,0,1,1,1,1,2,2,2,2], [0,1,2,3,0,1,2,3,0,1,2,3]]
            grid = torch.stack(torch.meshgrid(*positions, indexing='ij'))
            grid = rearrange(grid, 'c ... -> (...) c')  # (HW, num_dims)

            # Compute relative positions: pos[i] - pos[j]
            # rel_pos.shape = (i, j, num_dims)
            rel_pos = rearrange(grid, 'i c -> i 1 c') - rearrange(grid, 'j c -> 1 j c')

            # Logarithmic distance encoding: sign(x) * log(|x| + 1)
            # Reduces distinction for distant positions, focuses more on nearby positions
            if self.log_dist:
                rel_pos = torch.sign(rel_pos) * torch.log(rel_pos.abs() + 1)

            # Cache
            self.register_buffer('rel_pos', rel_pos, persistent=False)

        # Convert to float32 (for MLP computation)
        rel_pos = self.rel_pos.to(torch.float32)

        # Through MLP: (i, j, num_dims) -> (i, j, heads)
        for layer in self.net:
            rel_pos = layer(rel_pos.float())

        # Rearrange dimensions: (i, j, heads) -> (heads, i, j)
        return rearrange(rel_pos, 'i j h -> h i j')


# ============================================================================
# Transformer (Complete Transformer Block)
# ============================================================================

class Transformer(nn.Module):
    """
    Transformer Module (Multi-layer Stack)

    Architecture (per layer):
        Input
        → PEG (Position Encoding, optional)
        → Self-Attention + Residual
        → Cross-Attention + Residual (optional)
        → FeedForward + Residual
        → Output

    Args:
        dim: Feature dimension
        depth: Number of Transformer layers
        dim_context: Context dimension (for cross-attention)
        causal: Whether to use causal attention
        dim_head: Dimension per attention head
        heads: Number of attention heads
        ff_mult: FeedForward expansion multiplier
        peg: Whether to use PEG position encoding
        peg_causal: Whether PEG uses causal padding
        attn_num_null_kv: Number of null key-value pairs
        has_cross_attn: Whether to include cross-attention
        attn_dropout: Attention dropout
        ff_dropout: FeedForward dropout

    Modernization Opportunities - Architecture Optimization:
    1. Pre-LN vs Post-LN:
       - Current: Post-LN (LN inside Attention)
       - Switch to Pre-LN: LN(x) + Attn(...) more stable
       - Reference: GPT-3, LLaMA

    2. Parallel Attention + FFN:
       - Compute Attention and FFN in parallel then add
       - 10-15% speedup, comparable performance
       - Reference: PaLM

    3. MOE (Mixture of Experts):
       - Replace FFN with mixture of experts
       - Increase parameters while maintaining computation
       - Reference: Switch Transformer
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
        ff_dropout=0.
    ):
        super().__init__()
        self.layers = nn.ModuleList([])

        # Stack depth layers
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                # 1. PEG (Position Encoding Generator, optional)
                PEG(dim=dim, causal=peg_causal) if peg else None,

                # 2. Self-Attention
                Attention(
                    dim=dim, dim_head=dim_head, heads=heads,
                    causal=causal, dropout=attn_dropout
                ),

                # 3. Cross-Attention (optional)
                Attention(
                    dim=dim, dim_head=dim_head, dim_context=dim_context,
                    heads=heads, causal=False, num_null_kv=attn_num_null_kv,
                    dropout=attn_dropout
                ) if has_cross_attn else None,

                # 4. FeedForward
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        # Output normalization
        self.norm_out = LayerNorm(dim)

    @beartype
    def forward(
        self,
        x,
        video_shape: Tuple[int, int, int, int] = None,
        attn_bias=None,
        context=None,
        self_attn_mask=None,
        cross_attn_context_mask=None
    ):
        """
        Args:
            x: Input features (B, N, D)
            video_shape: Shape for PEG (B, T, H, W)
            attn_bias: Attention bias
            context: Context for cross-attention
            self_attn_mask: Mask for self-attention
            cross_attn_context_mask: Mask for cross-attention

        Returns:
            Output features (B, N, D)
        """
        # Iterate through each layer
        for peg, self_attn, cross_attn, ff in self.layers:
            # 1. Position encoding (if exists)
            if exists(peg):
                x = peg(x, shape=video_shape) + x

            # 2. Self-Attention + Residual
            x = self_attn(x, attn_bias=attn_bias, mask=self_attn_mask) + x

            # 3. Cross-Attention + Residual (if exists)
            if exists(cross_attn) and exists(context):
                x = cross_attn(x, context=context, mask=cross_attn_context_mask) + x

            # 4. FeedForward + Residual
            x = ff(x) + x

        # Output normalization
        return self.norm_out(x)
