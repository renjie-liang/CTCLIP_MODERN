"""
CTViT: 3D Vision Transformer for CT Volume Processing
VQ-VAE Autoencoder Based on Factorized Spatial-Temporal Attention

Architecture Flow:
    Input (B, C, T, H, W) - CT Volume Data
    ↓
    Patch Embedding - Split 3D volume into patches
    ↓
    Spatial Encoder - Spatial attention for each time frame
    ↓
    Temporal Encoder - Temporal attention for each spatial location
    ↓
    Vector Quantization (VQ) - Discrete encoding
    ↓
    Temporal Decoder - Temporal dimension decoding
    ↓
    Spatial Decoder - Spatial dimension decoding
    ↓
    Pixel Reconstruction - Reconstruct to original size
    ↓
    Output (B, C, T, H, W) - Reconstructed volume

Key Features:
    1. Factorized Spatial-Temporal Attention
    2. Vector Quantization for discrete representation
    3. Continuous Position Bias
    4. PEG (Position Encoding Generator)

Modernization Opportunities - Overall Architecture:
1. Use Flash Attention to accelerate attention computation
2. Introduce Grouped-Query Attention to reduce KV cache
3. Replace with more efficient positional encoding (e.g., RoPE)
4. Consider using Mixture of Experts (MOE) to increase model capacity
"""

import copy
import time
from pathlib import Path
from typing import Union, Tuple, Optional

import torch
import torch.nn.functional as F
from torch import nn
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from vector_quantize_pytorch import VectorQuantize

from .layers import exists, pair
from .attention import Transformer, ContinuousPositionBias


# ============================================================================
# CTViT Main Model
# ============================================================================

class CTViT(nn.Module):
    """
    CTViT: 3D Vision Transformer for CT Volumes

    VQ-VAE model based on factorized spatial-temporal attention for CT volume encoding and reconstruction

    Args:
        dim: Transformer hidden dimension (e.g., 512)
        codebook_size: VQ codebook size (e.g., 8192)
        image_size: Image size H, W (e.g., 480)
        patch_size: Patch size (e.g., 20, each patch is 20x20)
        temporal_patch_size: Temporal dimension patch size (e.g., 10)
        spatial_depth: Number of spatial Transformer layers (e.g., 4)
        temporal_depth: Number of temporal Transformer layers (e.g., 4)
        dim_head: Dimension per attention head (default: 64)
        heads: Number of attention heads (default: 8)
        channels: Number of input channels (typically 1 for CT)
        attn_dropout: Attention layer dropout (default: 0.)
        ff_dropout: FeedForward layer dropout (default: 0.)

    Input Shape:
        (B, C, T, H, W) - Batch, Channels, Time, Height, Width

    Output Modes:
        1. Default: (recon_loss, commit_loss, recon_video)
        2. return_recons_only=True: recon_video
        3. return_only_codebook_ids=True: indices
        4. return_encoded_tokens=True: tokens
    """

    def __init__(
        self,
        *,
        dim: int,
        codebook_size: int,
        image_size: int,
        patch_size: int,
        temporal_patch_size: int,
        spatial_depth: int,
        temporal_depth: int,
        dim_head: int = 64,
        heads: int = 8,
        channels: int = 1,
        attn_dropout: float = 0.,
        ff_dropout: float = 0.,
        # NEW: Optimization flags for ablation studies
        use_flash_attention: bool = False,
        use_rms_norm: bool = False,
        use_swiglu: bool = False,
        profile_timing: bool = False
    ):
        """
        Initialize CTViT model

        Einstein Notation:
            b - batch
            c - channels
            t - time (temporal dimension)
            h, w - height, width (spatial dimensions)
            d - feature dimension
            p1, p2 - patch height, patch width
            pt - temporal patch size
        """
        super().__init__()

        # ===== Basic Configuration =====
        self.image_size = pair(image_size)  # (H, W)
        self.patch_size = pair(patch_size)  # (pH, pW)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        # ===== Performance Profiling =====
        self.profile_timing = profile_timing
        if self.profile_timing:
            print("⚠️  CTViT: Performance profiling enabled")
        self.timing_buffer = {}

        # Check if size is divisible by patch size
        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"

        # ===== Position Encoding =====
        # Continuous position bias for spatial dimensions (used by Spatial Transformer)
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        # ===== Patch Embedding =====
        # Split 3D volume into patches and map to embedding space
        # Input:  (B, C, T, H, W)
        # Output: (B, T', H', W', D) where T'=T/pt, H'=H/pH, W'=W/pW
        self.to_patch_emb = nn.Sequential(
            # Rearrange: Split patches
            # (B, C, T, H, W) -> (B, T/pt, H/pH, W/pW, C*pt*pH*pW)
            Rearrange(
                'b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)',
                p1=patch_height, p2=patch_width, pt=temporal_patch_size
            ),
            # Normalization
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            # Linear projection to hidden dimension
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            # Normalization again
            nn.LayerNorm(dim)
        )

        # ===== Transformer Configuration =====
        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,        # Use PEG position encoding
            peg_causal=True, # Use causal padding for temporal dimension
            # Optimization flags
            use_flash_attention=use_flash_attention,
            use_rms_norm=use_rms_norm,
            use_swiglu=use_swiglu,
        )

        # ===== Encoder =====
        # 1. Spatial Encoder: Attention over spatial patches for each time frame
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)

        # 2. Temporal Encoder: Attention over time sequence for each spatial location
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)

        # ===== Vector Quantization =====
        # Quantize continuous features to discrete codebook indices
        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            use_cosine_sim=True  # Use cosine similarity for quantization
        )

        # ===== Decoder =====
        # Note: Original code was missing decoder definition, completed here
        # Decoder structure is symmetric to encoder but in reverse order: Temporal -> Spatial

        # 1. Temporal Decoder
        self.dec_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)

        # 2. Spatial Decoder
        self.dec_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)

        # ===== Pixel Reconstruction Layer =====
        # Map patches back to pixel space
        # Input:  (B, T', H', W', D)
        # Output: (B, C, T, H, W)
        self.to_pixels = nn.Sequential(
            # Linear projection: D -> C*pt*pH*pW
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            # Rearrange: Reconstruct to original shape
            # (B, T', H', W', C*pt*pH*pW) -> (B, C, T, H, W)
            Rearrange(
                'b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)',
                p1=patch_height, p2=patch_width, pt=temporal_patch_size
            ),
        )

    @property
    def patch_height_width(self):
        """Return patch grid dimensions (H', W')"""
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    @property
    def image_num_tokens(self):
        """Return number of tokens per time frame"""
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Encoding process: Spatial attention -> Temporal attention

        Args:
            tokens: (B, T', H', W', D) - Patch embeddings

        Returns:
            tokens: (B, T', H', W', D) - Encoded tokens
        """
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])  # (B, T', H', W')

        # ===== Spatial Encoding =====
        # Independent spatial attention for each time frame
        # (B, T', H', W', D) -> (B*T', H'*W', D)
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['rearrange_spatial_in'] = time.time() - t_start

        # Compute spatial positional bias
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['spatial_pos_bias'] = time.time() - t_start

        # Spatial Transformer
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['spatial_transformer'] = time.time() - t_start

        # Reshape back to 4D: (B*T', H'*W', D) -> (B, T', H', W', D)
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['rearrange_spatial_out'] = time.time() - t_start

        # ===== Temporal Encoding =====
        # Attention over time sequence for each spatial location
        # (B, T', H', W', D) -> (B*H'*W', T', D)
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['rearrange_temporal_in'] = time.time() - t_start

        # Temporal Transformer
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['temporal_transformer'] = time.time() - t_start

        # Reshape back to 4D: (B*H'*W', T', D) -> (B, T', H', W', D)
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['rearrange_temporal_out'] = time.time() - t_start

        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Decoding process: Temporal attention -> Spatial attention -> Pixel reconstruction

        Note: Decoding order is reverse of encoding
            Encoding: Spatial -> Temporal
            Decoding: Temporal -> Spatial

        Args:
            tokens: (B, T', H', W', D) or (B, N, D) - Quantized tokens

        Returns:
            recon_video: (B, C, T, H, W) - Reconstructed video
        """
        b = tokens.shape[0]
        h, w = self.patch_height_width

        # If input is flattened (B, N, D), reshape to 4D first
        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        video_shape = tuple(tokens.shape[:-1])  # (B, T', H', W')

        # ===== Temporal Decoding =====
        # (B, T', H', W', D) -> (B*H'*W', T', D)
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        # Temporal Transformer
        tokens = self.dec_temporal_transformer(tokens, video_shape=video_shape)

        # Reshape: (B*H'*W', T', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        # ===== Spatial Decoding =====
        # (B, T', H', W', D) -> (B*T', H'*W', D)
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        # Compute spatial positional bias
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        # Spatial Transformer
        tokens = self.dec_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        # Reshape: (B*T', H'*W', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # ===== Pixel Reconstruction =====
        # (B, T', H', W', D) -> (B, C, T, H, W)
        recon_video = self.to_pixels(tokens)

        return recon_video

    def forward(
        self,
        video: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_recons_only: bool = False,
        return_only_codebook_ids: bool = False,
        return_encoded_tokens: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, ...]]:
        """
        Forward pass

        Args:
            video: (B, C, T, H, W) - Input CT volume
            mask: (B, T) - Optional temporal mask (True=keep, False=mask)
            return_recons_only: Whether to return only reconstruction result
            return_only_codebook_ids: Whether to return only VQ indices
            return_encoded_tokens: Whether to return only encoded tokens

        Returns:
            Default: (recon_loss, commit_loss, recon_video)
            return_recons_only=True: recon_video
            return_only_codebook_ids=True: indices
            return_encoded_tokens=True: tokens

        Note:
            - Input must be 5D tensor (video format)
            - Original is_image branch removed as only video input is needed in practice
        """
        # ===== Input Validation =====
        assert video.ndim == 5, f"Input must be 5D (B, C, T, H, W), got shape {video.shape}"

        b, c, f, *image_dims, device = *video.shape, video.device

        # Check image dimensions
        assert tuple(image_dims) == self.image_size, \
            f"Input image size {image_dims} doesn't match model image size {self.image_size}"

        # Check mask dimensions
        assert not exists(mask) or mask.shape[-1] == f, \
            f"Mask temporal dimension {mask.shape[-1]} doesn't match video frames {f}"

        # Clear timing buffer
        if self.profile_timing:
            self.timing_buffer.clear()
            torch.cuda.synchronize()
            forward_start = time.time()

        # ===== 1. Patch Embedding =====
        # (B, C, T, H, W) -> (B, T', H', W', D)
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = self.to_patch_emb(video)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['patch_embedding'] = time.time() - t_start

        # Save shape information
        *_, h, w, _ = tokens.shape

        # ===== 2. Encode (Spatial -> Temporal) =====
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens = self.encode(tokens)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['encode_total'] = time.time() - t_start

        # ===== 3. Vector Quantization =====
        # Flatten tokens: (B, T', H', W', D) -> (B, T'*H'*W', D)
        if self.profile_timing:
            torch.cuda.synchronize()
            t_start = time.time()

        tokens, packed_fhw_shape = pack([tokens], 'b * d')

        # Compute VQ mask (if temporal mask is provided)
        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)

        # VQ quantization
        # tokens: Quantized continuous features
        # indices: Codebook indices
        # commit_loss: VQ commitment loss
        tokens, indices, commit_loss = self.vq(tokens, mask=vq_mask)

        if self.profile_timing:
            torch.cuda.synchronize()
            self.timing_buffer['vector_quantization'] = time.time() - t_start

        # If only need to return codebook indices
        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        # Reshape back to 4D: (B, T'*H'*W', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        # If only need to return encoded tokens
        if return_encoded_tokens:
            if self.profile_timing:
                torch.cuda.synchronize()
                total_time = time.time() - forward_start
                self.timing_buffer['total_forward'] = total_time
                self._print_timing_stats()
            return tokens

        # ===== 4. Decode (Temporal -> Spatial -> Pixels) =====
        recon_video = self.decode(tokens)

        # If only need to return reconstruction result
        if return_recons_only:
            return recon_video

        # ===== 5. Compute Loss =====
        # Reconstruction loss (MSE)
        if exists(mask):
            # If mask exists, only compute loss for non-masked positions
            recon_loss = F.mse_loss(video, recon_video, reduction='none')
            # Apply mask: (B, T) -> (B, C, T, 1, 1)
            mask_expanded = repeat(mask, 'b t -> b c t 1 1', c=c)
            recon_loss = recon_loss[mask_expanded]
            recon_loss = recon_loss.mean()
        else:
            # Compute loss for all positions
            recon_loss = F.mse_loss(video, recon_video)

        # ===== 6. Return Results =====
        # Return: (reconstruction loss, VQ commitment loss, reconstructed video)
        return recon_loss, commit_loss, recon_video

    def _print_timing_stats(self):
        """Print detailed timing breakdown of CTViT forward pass"""
        if not self.timing_buffer:
            return

        total_time = self.timing_buffer.get('total_forward', 0)

        print("\n" + "="*80)
        print("🔍 CTViT Timing Breakdown")
        print("="*80)

        # Main stages
        print("\nMain Stages:")
        for key in ['patch_embedding', 'encode_total', 'vector_quantization']:
            if key in self.timing_buffer:
                t = self.timing_buffer[key] * 1000  # Convert to ms
                pct = (self.timing_buffer[key] / total_time * 100) if total_time > 0 else 0
                print(f"  {key:25s}: {t:8.2f}ms ({pct:5.1f}%)")

        # Detailed encode breakdown
        print("\nDetailed Encode Breakdown:")
        for key in ['spatial_transformer', 'temporal_transformer', 'spatial_pos_bias',
                    'rearrange_spatial_in', 'rearrange_spatial_out',
                    'rearrange_temporal_in', 'rearrange_temporal_out']:
            if key in self.timing_buffer:
                t = self.timing_buffer[key] * 1000
                pct = (self.timing_buffer[key] / total_time * 100) if total_time > 0 else 0
                print(f"  {key:25s}: {t:8.2f}ms ({pct:5.1f}%)")

        print(f"\n{'─'*80}")
        print(f"  {'Total CTViT Forward':25s}: {total_time*1000:8.2f}ms (100.0%)")
        print("="*80 + "\n")

    def calculate_video_token_mask(self, videos: torch.Tensor, video_frame_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute token-level mask (for VQ)

        Convert frame-level mask to token-level mask

        Args:
            videos: (B, C, T, H, W)
            video_frame_mask: (B, T) - Frame-level mask

        Returns:
            token_mask: (B, N) - Token-level mask, N = T' * H' * W'
        """
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        # Group frame mask by temporal_patch_size
        # If any frame in a group is True, that patch is True
        rest_vq_mask = rearrange(video_frame_mask, 'b (f p) -> b f p', p=self.temporal_patch_size)
        video_mask = rest_vq_mask.any(dim=-1)  # (B, T')

        # Expand to all spatial locations
        # (B, T') -> (B, T' * H' * W')
        return repeat(video_mask, 'b f -> b (f hw)', hw=(h // ph) * (w // pw))

    def copy_for_eval(self):
        """
        Create evaluation copy of the model

        Used for saving/deployment to remove training-related components

        Returns:
            vae_copy: Model copy in evaluation mode
        """
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())
        vae_copy.eval()
        return vae_copy.to(device)

    def load(self, path: Union[str, Path]):
        """
        Load model weights from checkpoint

        Args:
            path: Path to checkpoint file
        """
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        Decode directly from codebook indices

        Used to reconstruct video from discrete indices

        Args:
            indices: (B, N) - Codebook indices

        Returns:
            recon_video: (B, C, T, H, W) - Reconstructed video
        """
        # Get corresponding feature vectors from codebook
        codes = self.vq.codebook[indices]
        # Decode
        return self.decode(codes)

    def num_tokens_per_frames(self, num_frames: int, include_first_frame: bool = True) -> int:
        """
        Compute number of tokens for given number of frames

        Args:
            num_frames: Number of frames
            include_first_frame: Whether to include first frame (for backward compatibility, no longer distinguished)

        Returns:
            total_tokens: Total number of tokens
        """
        image_num_tokens = self.image_num_tokens

        # Check if num_frames is divisible by temporal_patch_size
        assert (num_frames % self.temporal_patch_size) == 0, \
            f"num_frames {num_frames} must be divisible by temporal_patch_size {self.temporal_patch_size}"

        # Compute: (T / temporal_patch_size) * (H' * W')
        return int(num_frames / self.temporal_patch_size) * image_num_tokens
