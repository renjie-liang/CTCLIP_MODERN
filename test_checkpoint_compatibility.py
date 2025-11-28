#!/usr/bin/env python3
"""
Test script to verify checkpoint compatibility between old baseline and new baseline
"""
import torch
from src.models.ctvit import CTViT

def test_baseline_uses_layernorm():
    """Test that baseline config uses LayerNorm (for old checkpoint compatibility)"""
    print("="*80)
    print("Testing Baseline Configuration (should use LayerNorm)")
    print("="*80)

    model = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=2,  # Reduced for faster testing
        temporal_depth=2,
        dim_head=32,
        heads=8,
        # Baseline configuration
        use_flash_attention=False,
        use_rms_norm=False,
        use_swiglu=False
    )

    state_dict = model.state_dict()

    # Check for LayerNorm parameters (gamma/beta)
    layernorm_keys = [k for k in state_dict.keys() if '.gamma' in k or '.beta' in k]
    rmsnorm_keys = [k for k in state_dict.keys() if 'norm' in k and '.weight' in k and 'gamma' not in k]

    print(f"\n✓ Total parameters: {len(state_dict.keys())}")
    print(f"\n✓ LayerNorm parameters (.gamma/.beta): {len(layernorm_keys)}")
    if layernorm_keys[:5]:
        print("  Examples:")
        for key in layernorm_keys[:5]:
            print(f"    - {key}")

    print(f"\n✓ RMSNorm parameters (.weight): {len(rmsnorm_keys)}")
    if rmsnorm_keys[:5]:
        print("  Examples:")
        for key in rmsnorm_keys[:5]:
            print(f"    - {key}")

    # Check specific Attention layers
    print("\n" + "="*80)
    print("Checking Attention Layer Normalization")
    print("="*80)
    attn_norm_keys = [k for k in state_dict.keys() if 'transformer' in k and 'layers' in k and 'norm' in k]
    print(f"\nFound {len(attn_norm_keys)} normalization parameters in Transformer layers:")
    for key in sorted(attn_norm_keys)[:10]:
        print(f"  - {key}")

    # Verify we have LayerNorm (gamma) not RMSNorm (weight)
    has_gamma = any('.gamma' in k for k in attn_norm_keys)
    has_weight_only = any('.weight' in k and '.gamma' not in k for k in attn_norm_keys)

    print(f"\n✓ Has LayerNorm (.gamma): {has_gamma}")
    print(f"✓ Has RMSNorm (.weight only): {has_weight_only}")

    if has_gamma and not has_weight_only:
        print("\n✅ SUCCESS: Baseline config correctly uses LayerNorm!")
        return True
    else:
        print("\n❌ FAILED: Baseline config is not using LayerNorm!")
        return False


def test_optimized_uses_rmsnorm():
    """Test that optimized config uses RMSNorm"""
    print("\n" + "="*80)
    print("Testing Optimized Configuration (should use RMSNorm)")
    print("="*80)

    model = CTViT(
        dim=512,
        codebook_size=8192,
        image_size=480,
        patch_size=20,
        temporal_patch_size=10,
        spatial_depth=2,
        temporal_depth=2,
        dim_head=32,
        heads=8,
        # Optimized configuration
        use_flash_attention=True,
        use_rms_norm=True,
        use_swiglu=True
    )

    state_dict = model.state_dict()

    # Check Transformer normalization
    transformer_norm_keys = [k for k in state_dict.keys() if 'transformer' in k and 'norm_out' in k]
    print(f"\nFound {len(transformer_norm_keys)} Transformer output norm parameters:")
    for key in sorted(transformer_norm_keys):
        print(f"  - {key}")

    # RMSNorm should have .weight parameter
    has_rmsnorm = any('.weight' in k for k in transformer_norm_keys)
    has_layernorm = any('.gamma' in k for k in transformer_norm_keys)

    print(f"\n✓ Has RMSNorm (.weight): {has_rmsnorm}")
    print(f"✓ Has LayerNorm (.gamma): {has_layernorm}")

    if has_rmsnorm and not has_layernorm:
        print("\n✅ SUCCESS: Optimized config correctly uses RMSNorm!")
        return True
    else:
        print("\n❌ FAILED: Optimized config is not using RMSNorm!")
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("Checkpoint Compatibility Test")
    print("="*80)

    baseline_ok = test_baseline_uses_layernorm()
    optimized_ok = test_optimized_uses_rmsnorm()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Baseline (LayerNorm):  {'✅ PASS' if baseline_ok else '❌ FAIL'}")
    print(f"Optimized (RMSNorm):   {'✅ PASS' if optimized_ok else '❌ FAIL'}")

    if baseline_ok and optimized_ok:
        print("\n🎉 All tests passed! Old checkpoints should now be compatible.")
        exit(0)
    else:
        print("\n❌ Some tests failed!")
        exit(1)
