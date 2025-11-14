#!/usr/bin/env python3
"""
Profile CTViT forward pass to identify bottlenecks.

This script adds timing hooks to CTViT to measure:
- Patch embedding time
- Spatial encoder time
- Temporal encoder time
- VQ quantization time
- Rearrange operations time
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import torch
import numpy as np
from src.utils.config import load_config
from src.models import build_model


def profile_ctvit_forward(model, device, num_iterations=20):
    """Profile CTViT forward pass with detailed timing"""

    # Create dummy input (batch_size=8, like real training)
    dummy_input = torch.randn(8, 1, 240, 480, 480, device=device)

    print("="*80)
    print("Profiling CTViT Forward Pass")
    print("="*80)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Device: {device}")
    print(f"Iterations: {num_iterations}")
    print("="*80)

    # Warmup
    print("\nWarming up...")
    for _ in range(5):
        with torch.no_grad():
            _ = model.visual_transformer(dummy_input, return_encoded_tokens=True)
    torch.cuda.synchronize()
    print("Warmup complete.")

    # Profile
    print("\nProfiling...")
    timings = {
        'total_forward': [],
        'patch_embed': [],
        'encode': [],
        'vq': [],
    }

    for i in range(num_iterations):
        with torch.no_grad():
            # Total forward time
            torch.cuda.synchronize()
            t_start = time.time()

            # Manually step through CTViT to measure each component
            vit = model.visual_transformer

            # 1. Patch Embedding
            torch.cuda.synchronize()
            t0 = time.time()
            tokens = vit.to_patch_emb(dummy_input)
            torch.cuda.synchronize()
            t_patch = time.time() - t0

            # 2. Encoding (spatial + temporal)
            torch.cuda.synchronize()
            t0 = time.time()
            tokens = vit.encode(tokens)
            torch.cuda.synchronize()
            t_encode = time.time() - t0

            # 3. VQ
            torch.cuda.synchronize()
            t0 = time.time()
            from einops import pack
            tokens_flat, packed_shape = pack([tokens], 'b * d')
            tokens_vq, indices, commit_loss = vit.vq(tokens_flat)
            torch.cuda.synchronize()
            t_vq = time.time() - t0

            torch.cuda.synchronize()
            t_total = time.time() - t_start

            timings['total_forward'].append(t_total)
            timings['patch_embed'].append(t_patch)
            timings['encode'].append(t_encode)
            timings['vq'].append(t_vq)

        if (i + 1) % 5 == 0:
            print(f"Iteration {i+1}/{num_iterations} completed")

    # Print results
    print("\n" + "="*80)
    print("Results (averaged over {} iterations)".format(num_iterations))
    print("="*80)

    for key in ['patch_embed', 'encode', 'vq', 'total_forward']:
        times = np.array(timings[key]) * 1000  # Convert to ms
        mean_time = np.mean(times)
        std_time = np.std(times)

        if key == 'total_forward':
            pct = 100.0
        else:
            pct = (mean_time / np.mean(np.array(timings['total_forward']) * 1000)) * 100

        print(f"{key:20s}: {mean_time:8.2f}ms ± {std_time:6.2f}ms ({pct:5.1f}%)")

    print("="*80)

    # Now profile encode() separately to see spatial vs temporal
    print("\n" + "="*80)
    print("Detailed Profiling of Encode Stage")
    print("="*80)

    enc_timings = {
        'spatial_transformer': [],
        'temporal_transformer': [],
    }

    for i in range(num_iterations):
        with torch.no_grad():
            tokens = vit.to_patch_emb(dummy_input)

            b = tokens.shape[0]
            h, w = vit.patch_height_width
            video_shape = tuple(tokens.shape[:-1])

            # Spatial encoding
            from einops import rearrange
            tokens_spatial = rearrange(tokens, 'b t h w d -> (b t) (h w) d')
            attn_bias = vit.spatial_rel_pos_bias(h, w, device=tokens_spatial.device)

            torch.cuda.synchronize()
            t0 = time.time()
            tokens_spatial = vit.enc_spatial_transformer(tokens_spatial, attn_bias=attn_bias, video_shape=video_shape)
            torch.cuda.synchronize()
            t_spatial = time.time() - t0

            tokens_spatial = rearrange(tokens_spatial, '(b t) (h w) d -> b t h w d', b=b, h=h, w=w)

            # Temporal encoding
            tokens_temporal = rearrange(tokens_spatial, 'b t h w d -> (b h w) t d')

            torch.cuda.synchronize()
            t0 = time.time()
            tokens_temporal = vit.enc_temporal_transformer(tokens_temporal, video_shape=video_shape)
            torch.cuda.synchronize()
            t_temporal = time.time() - t0

            enc_timings['spatial_transformer'].append(t_spatial)
            enc_timings['temporal_transformer'].append(t_temporal)

    total_encode_time = np.mean(timings['encode']) * 1000

    for key in ['spatial_transformer', 'temporal_transformer']:
        times = np.array(enc_timings[key]) * 1000
        mean_time = np.mean(times)
        std_time = np.std(times)
        pct = (mean_time / total_encode_time) * 100

        print(f"{key:25s}: {mean_time:8.2f}ms ± {std_time:6.2f}ms ({pct:5.1f}% of encode)")

    print("="*80)

    # Memory usage
    print("\n" + "="*80)
    print("Memory Usage")
    print("="*80)
    free_mem, total_mem = torch.cuda.mem_get_info()
    used_mem = total_mem - free_mem
    print(f"GPU Memory: {used_mem / 1024**3:.2f} / {total_mem / 1024**3:.2f} GB ({used_mem/total_mem*100:.1f}%)")
    print("="*80)

    return timings


def main():
    # Load config
    config_path = project_root / "configs" / "base_config.yaml"
    config = load_config(str(config_path))

    # Build model
    print("Building model...")
    model = build_model(config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()

    print(f"Model loaded on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Profile
    timings = profile_ctvit_forward(model, device, num_iterations=20)

    # Summary
    print("\n" + "="*80)
    print("Summary")
    print("="*80)
    avg_forward = np.mean(timings['total_forward']) * 1000
    print(f"Average CTViT forward pass: {avg_forward:.2f}ms")
    print(f"For batch_size=8, this is {avg_forward/8:.2f}ms per sample")

    # Compare with actual training time
    print("\n" + "="*80)
    print("Comparison with Training")
    print("="*80)
    print(f"CTViT forward time:        {avg_forward:.2f}ms")
    print(f"Training total forward:    26318.65ms (from your logs)")
    print(f"Text encoding overhead:    {26318.65 - avg_forward:.2f}ms")
    print(f"  (includes: tokenization, BERT, CLIP similarity)")
    print("="*80)


if __name__ == "__main__":
    main()
