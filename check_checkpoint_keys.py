import torch

ckpt = torch.load('saves/important/baseline/best.pt', map_location='cpu', weights_only=False)

state_dict = ckpt['model_state_dict']

print("=" * 80)
print(f"Total parameters: {len(state_dict)}")
print("=" * 80)

# Find all keys related to enc_spatial_transformer.layers.0
spatial_keys = [k for k in state_dict.keys() if 'enc_spatial_transformer.layers.0' in k]

print("\nKeys for enc_spatial_transformer.layers.0:")
for key in sorted(spatial_keys):
    print(f"  {key}")

# Check norm-related keys
print("\n" + "=" * 80)
print("All norm-related keys:")
norm_keys = [k for k in state_dict.keys() if 'norm' in k.lower()]
for key in sorted(norm_keys)[:20]:  # First 20
    print(f"  {key}")

# Check what type of norm is used
print("\n" + "=" * 80)
print("Checking norm type in checkpoint:")

# Check if it has 'gamma' (LayerNorm) or 'weight' (RMSNorm or nn.LayerNorm)
has_gamma = any('gamma' in k for k in state_dict.keys())
has_norm_weight = any('norm.weight' in k for k in state_dict.keys())

print(f"  Has 'gamma' keys (custom LayerNorm): {has_gamma}")
print(f"  Has 'norm.weight' keys (nn.LayerNorm/RMSNorm): {has_norm_weight}")

# Check a specific layer structure
print("\n" + "=" * 80)
print("Layer structure check:")
layer_0_1_keys = [k for k in state_dict.keys() if 'enc_spatial_transformer.layers.0.1' in k]
print(f"Keys for layers.0.1 (should be Attention):")
for key in sorted(layer_0_1_keys):
    print(f"  {key}")
