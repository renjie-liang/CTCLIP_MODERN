"""
CTViT: 3D Vision Transformer for CT Volume Processing
基于时空分离注意力机制的VQ-VAE自编码器

Architecture Flow:
    Input (B, C, T, H, W) - CT Volume数据
    ↓
    Patch Embedding - 将3D volume切分为patches
    ↓
    Spatial Encoder - 对每个时间帧做空间注意力
    ↓
    Temporal Encoder - 对同一空间位置做时间注意力
    ↓
    Vector Quantization (VQ) - 离散化编码
    ↓
    Temporal Decoder - 时间维度解码
    ↓
    Spatial Decoder - 空间维度解码
    ↓
    Pixel Reconstruction - 重建为原始尺寸
    ↓
    Output (B, C, T, H, W) - 重建的volume

Key Features:
    1. Factorized Spatial-Temporal Attention (时空分离注意力)
    2. Vector Quantization for discrete representation (VQ离散化表示)
    3. Continuous Position Bias (连续位置偏置)
    4. PEG Position Encoding (位置编码生成器)

🔧 [整体架构现代化改造方向]:
1. 使用Flash Attention加速注意力计算
2. 引入Grouped-Query Attention减少KV cache
3. 替换为更高效的位置编码 (如RoPE)
4. 考虑使用混合专家(MOE)增加模型容量
"""

import copy
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

    基于时空分离注意力机制的VQ-VAE模型，用于CT体积数据的编码和重建

    Args:
        dim: Transformer隐藏维度 (例如: 512)
        codebook_size: VQ码本大小 (例如: 8192)
        image_size: 图像尺寸 H, W (例如: 480)
        patch_size: Patch大小 (例如: 20, 则每个patch为20x20)
        temporal_patch_size: 时间维度patch大小 (例如: 10)
        spatial_depth: 空间Transformer层数 (例如: 4)
        temporal_depth: 时间Transformer层数 (例如: 4)
        dim_head: 每个注意力头的维度 (默认: 64)
        heads: 注意力头数 (默认: 8)
        channels: 输入通道数 (CT通常为1)
        attn_dropout: 注意力层dropout (默认: 0.)
        ff_dropout: FeedForward层dropout (默认: 0.)

    Input Shape:
        (B, C, T, H, W) - Batch, Channels, Time, Height, Width

    Output Modes:
        1. 默认: (recon_loss, commit_loss, recon_video)
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
        ff_dropout: float = 0.
    ):
        """
        初始化CTViT模型

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

        # ===== 基本配置 =====
        self.image_size = pair(image_size)  # (H, W)
        self.patch_size = pair(patch_size)  # (pH, pW)
        patch_height, patch_width = self.patch_size

        self.temporal_patch_size = temporal_patch_size

        # 检查尺寸是否能被patch size整除
        image_height, image_width = self.image_size
        assert (image_height % patch_height) == 0 and (image_width % patch_width) == 0, \
            f"Image size {self.image_size} must be divisible by patch size {self.patch_size}"

        # ===== 位置编码 =====
        # 空间维度的连续位置偏置 (用于Spatial Transformer)
        self.spatial_rel_pos_bias = ContinuousPositionBias(dim=dim, heads=heads)

        # ===== Patch Embedding =====
        # 将3D volume切分为patches并映射到embedding空间
        # Input:  (B, C, T, H, W)
        # Output: (B, T', H', W', D) 其中 T'=T/pt, H'=H/pH, W'=W/pW
        self.to_patch_emb = nn.Sequential(
            # Rearrange: 切分patches
            # (B, C, T, H, W) -> (B, T/pt, H/pH, W/pW, C*pt*pH*pW)
            Rearrange(
                'b c (t pt) (h p1) (w p2) -> b t h w (c pt p1 p2)',
                p1=patch_height, p2=patch_width, pt=temporal_patch_size
            ),
            # 归一化
            nn.LayerNorm(channels * patch_width * patch_height * temporal_patch_size),
            # 线性投影到隐藏维度
            nn.Linear(channels * patch_width * patch_height * temporal_patch_size, dim),
            # 再次归一化
            nn.LayerNorm(dim)
        )

        # ===== Transformer配置 =====
        transformer_kwargs = dict(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            peg=True,        # 使用PEG位置编码
            peg_causal=True, # 时间维度使用因果padding
        )

        # ===== 编码器 (Encoder) =====
        # 1. 空间编码器: 对每个时间帧的空间patches做注意力
        self.enc_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)

        # 2. 时间编码器: 对同一空间位置的时间序列做注意力
        self.enc_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)

        # ===== Vector Quantization =====
        # 将连续特征量化为离散的codebook索引
        self.vq = VectorQuantize(
            dim=dim,
            codebook_size=codebook_size,
            use_cosine_sim=True  # 使用余弦相似度进行量化
        )

        # ===== 解码器 (Decoder) =====
        # 注意: 原始代码缺少解码器定义，这里补充完整
        # 解码器结构与编码器对称，但顺序相反: 时间 -> 空间

        # 1. 时间解码器
        self.dec_temporal_transformer = Transformer(depth=temporal_depth, **transformer_kwargs)

        # 2. 空间解码器
        self.dec_spatial_transformer = Transformer(depth=spatial_depth, **transformer_kwargs)

        # ===== 像素重建层 =====
        # 将patches映射回像素空间
        # Input:  (B, T', H', W', D)
        # Output: (B, C, T, H, W)
        self.to_pixels = nn.Sequential(
            # 线性投影: D -> C*pt*pH*pW
            nn.Linear(dim, channels * patch_width * patch_height * temporal_patch_size),
            # Rearrange: 重组为原始形状
            # (B, T', H', W', C*pt*pH*pW) -> (B, C, T, H, W)
            Rearrange(
                'b t h w (c pt p1 p2) -> b c (t pt) (h p1) (w p2)',
                p1=patch_height, p2=patch_width, pt=temporal_patch_size
            ),
        )

    @property
    def patch_height_width(self):
        """返回patch grid的尺寸 (H', W')"""
        return self.image_size[0] // self.patch_size[0], self.image_size[1] // self.patch_size[1]

    @property
    def image_num_tokens(self):
        """返回每个时间帧的token数量"""
        return int(self.image_size[0] / self.patch_size[0]) * int(self.image_size[1] / self.patch_size[1])

    def encode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        编码过程: 空间注意力 -> 时间注意力

        Args:
            tokens: (B, T', H', W', D) - Patch embeddings

        Returns:
            tokens: (B, T', H', W', D) - Encoded tokens
        """
        b = tokens.shape[0]
        h, w = self.patch_height_width

        video_shape = tuple(tokens.shape[:-1])  # (B, T', H', W')

        # ===== 空间编码 (Spatial Encoding) =====
        # 对每个时间帧独立做空间注意力
        # (B, T', H', W', D) -> (B*T', H'*W', D)
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        # 计算空间位置偏置
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        # 空间Transformer
        tokens = self.enc_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        # Reshape回4D: (B*T', H'*W', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # ===== 时间编码 (Temporal Encoding) =====
        # 对同一空间位置的时间序列做注意力
        # (B, T', H', W', D) -> (B*H'*W', T', D)
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        # 时间Transformer
        tokens = self.enc_temporal_transformer(tokens, video_shape=video_shape)

        # Reshape回4D: (B*H'*W', T', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        return tokens

    def decode(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        解码过程: 时间注意力 -> 空间注意力 -> 像素重建

        注意: 解码顺序与编码相反
            编码: 空间 -> 时间
            解码: 时间 -> 空间

        Args:
            tokens: (B, T', H', W', D) 或 (B, N, D) - Quantized tokens

        Returns:
            recon_video: (B, C, T, H, W) - 重建的video
        """
        b = tokens.shape[0]
        h, w = self.patch_height_width

        # 如果输入是flatten的 (B, N, D)，先reshape为4D
        if tokens.ndim == 3:
            tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        video_shape = tuple(tokens.shape[:-1])  # (B, T', H', W')

        # ===== 时间解码 (Temporal Decoding) =====
        # (B, T', H', W', D) -> (B*H'*W', T', D)
        tokens = rearrange(tokens, 'b t h w d -> (b h w) t d')

        # 时间Transformer
        tokens = self.dec_temporal_transformer(tokens, video_shape=video_shape)

        # Reshape: (B*H'*W', T', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, '(b h w) t d -> b t h w d', b=b, h=h, w=w)

        # ===== 空间解码 (Spatial Decoding) =====
        # (B, T', H', W', D) -> (B*T', H'*W', D)
        tokens = rearrange(tokens, 'b t h w d -> (b t) (h w) d')

        # 计算空间位置偏置
        attn_bias = self.spatial_rel_pos_bias(h, w, device=tokens.device)

        # 空间Transformer
        tokens = self.dec_spatial_transformer(tokens, attn_bias=attn_bias, video_shape=video_shape)

        # Reshape: (B*T', H'*W', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

        # ===== 像素重建 =====
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
        前向传播

        Args:
            video: (B, C, T, H, W) - 输入CT volume
            mask: (B, T) - 可选的时间mask (True=保留, False=mask)
            return_recons_only: 是否只返回重建结果
            return_only_codebook_ids: 是否只返回VQ索引
            return_encoded_tokens: 是否只返回编码后的tokens

        Returns:
            默认: (recon_loss, commit_loss, recon_video)
            return_recons_only=True: recon_video
            return_only_codebook_ids=True: indices
            return_encoded_tokens=True: tokens

        Note:
            - 输入必须是5D tensor (video格式)
            - 原始代码的is_image分支已删除，因为实际使用中只需要video输入
        """
        # ===== 输入检查 =====
        assert video.ndim == 5, f"Input must be 5D (B, C, T, H, W), got shape {video.shape}"

        b, c, f, *image_dims, device = *video.shape, video.device

        # 检查图像尺寸
        assert tuple(image_dims) == self.image_size, \
            f"Input image size {image_dims} doesn't match model image size {self.image_size}"

        # 检查mask尺寸
        assert not exists(mask) or mask.shape[-1] == f, \
            f"Mask temporal dimension {mask.shape[-1]} doesn't match video frames {f}"

        # ===== 1. Patch Embedding =====
        # (B, C, T, H, W) -> (B, T', H', W', D)
        tokens = self.to_patch_emb(video)

        # 保存shape信息
        *_, h, w, _ = tokens.shape

        # ===== 2. 编码 (Spatial -> Temporal) =====
        tokens = self.encode(tokens)

        # ===== 3. Vector Quantization =====
        # Flatten tokens: (B, T', H', W', D) -> (B, T'*H'*W', D)
        tokens, packed_fhw_shape = pack([tokens], 'b * d')

        # 计算VQ mask (如果提供了时间mask)
        vq_mask = None
        if exists(mask):
            vq_mask = self.calculate_video_token_mask(video, mask)

        # VQ量化
        # tokens: 量化后的连续特征
        # indices: codebook索引
        # commit_loss: VQ承诺损失
        tokens, indices, commit_loss = self.vq(tokens, mask=vq_mask)

        # 如果只需要返回codebook索引
        if return_only_codebook_ids:
            indices, = unpack(indices, packed_fhw_shape, 'b *')
            return indices

        # Reshape回4D: (B, T'*H'*W', D) -> (B, T', H', W', D)
        tokens = rearrange(tokens, 'b (t h w) d -> b t h w d', h=h, w=w)

        # 如果只需要返回编码后的tokens
        if return_encoded_tokens:
            return tokens

        # ===== 4. 解码 (Temporal -> Spatial -> Pixels) =====
        recon_video = self.decode(tokens)

        # 如果只需要返回重建结果
        if return_recons_only:
            return recon_video

        # ===== 5. 计算损失 =====
        # 重建损失 (MSE)
        if exists(mask):
            # 如果有mask，只计算非mask位置的损失
            recon_loss = F.mse_loss(video, recon_video, reduction='none')
            # 应用mask: (B, T) -> (B, C, T, 1, 1)
            mask_expanded = repeat(mask, 'b t -> b c t 1 1', c=c)
            recon_loss = recon_loss[mask_expanded]
            recon_loss = recon_loss.mean()
        else:
            # 全部位置都计算损失
            recon_loss = F.mse_loss(video, recon_video)

        # ===== 6. 返回结果 =====
        # 返回: (重建损失, VQ承诺损失, 重建video)
        return recon_loss, commit_loss, recon_video

    def calculate_video_token_mask(self, videos: torch.Tensor, video_frame_mask: torch.Tensor) -> torch.Tensor:
        """
        计算token级别的mask (用于VQ)

        将帧级别的mask转换为token级别的mask

        Args:
            videos: (B, C, T, H, W)
            video_frame_mask: (B, T) - 帧级别mask

        Returns:
            token_mask: (B, N) - Token级别mask, N = T' * H' * W'
        """
        *_, h, w = videos.shape
        ph, pw = self.patch_size

        # 将帧mask按temporal_patch_size分组
        # 如果一组内有任何帧为True，则该patch为True
        rest_vq_mask = rearrange(video_frame_mask, 'b (f p) -> b f p', p=self.temporal_patch_size)
        video_mask = rest_vq_mask.any(dim=-1)  # (B, T')

        # 扩展到所有空间位置
        # (B, T') -> (B, T' * H' * W')
        return repeat(video_mask, 'b f -> b (f hw)', hw=(h // ph) * (w // pw))

    def copy_for_eval(self):
        """
        创建模型的评估副本

        用于保存/部署时去除训练相关组件

        Returns:
            vae_copy: 评估模式的模型副本
        """
        device = next(self.parameters()).device
        vae_copy = copy.deepcopy(self.cpu())
        vae_copy.eval()
        return vae_copy.to(device)

    def load(self, path: Union[str, Path]):
        """
        从checkpoint加载模型权重

        Args:
            path: checkpoint文件路径
        """
        path = Path(path)
        assert path.exists(), f"Checkpoint not found: {path}"
        pt = torch.load(str(path))
        self.load_state_dict(pt)

    def decode_from_codebook_indices(self, indices: torch.Tensor) -> torch.Tensor:
        """
        从codebook索引直接解码

        用于从离散索引重建video

        Args:
            indices: (B, N) - Codebook索引

        Returns:
            recon_video: (B, C, T, H, W) - 重建的video
        """
        # 从codebook获取对应的特征向量
        codes = self.vq.codebook[indices]
        # 解码
        return self.decode(codes)

    def num_tokens_per_frames(self, num_frames: int, include_first_frame: bool = True) -> int:
        """
        计算给定帧数对应的token数量

        Args:
            num_frames: 帧数
            include_first_frame: 是否包含第一帧 (兼容旧代码，实际上已不区分)

        Returns:
            total_tokens: Token总数
        """
        image_num_tokens = self.image_num_tokens

        # 检查帧数能否被temporal_patch_size整除
        assert (num_frames % self.temporal_patch_size) == 0, \
            f"num_frames {num_frames} must be divisible by temporal_patch_size {self.temporal_patch_size}"

        # 计算: (T / temporal_patch_size) * (H' * W')
        return int(num_frames / self.temporal_patch_size) * image_num_tokens
