#!/usr/bin/env python3
"""
Profile data loading pipeline to identify bottlenecks.

This script measures the time spent in each step of _process_volume()
to determine where optimization efforts should focus.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import time
import numpy as np
import torch
from src.data.webdataset_loader import CTReportWebDataset
from src.utils.config import load_config


def profile_single_sample(dataset, num_samples=10):
    """
    Profile the time spent in each step of data loading.

    This creates a modified WebDataset that times each operation.
    """
    import webdataset as wds

    print("="*80)
    print("Profiling Data Loading Pipeline")
    print("="*80)
    print(f"Will profile {num_samples} samples\n")

    # Create WebDataset pipeline
    shard_shuffle = 100 if dataset.shuffle else False
    wds_dataset = (
        wds.WebDataset(dataset.shard_pattern, shardshuffle=shard_shuffle, empty_check=False)
        .shuffle(dataset.buffer_size if dataset.shuffle else 0)
    )

    # Timing accumulators
    times = {
        'total': [],
        'tar_read': [],
        'metadata_decode': [],
        'volume_decode': [],
        'report_decode': [],
        'labels_decode': [],
        'dtype_convert': [],
        'rescale': [],
        'clip': [],
        'transpose': [],
        'resize': [],
        'normalize': [],
        'crop_pad': [],
        'permute': [],
    }

    print("Loading samples...")
    for i, sample in enumerate(wds_dataset):
        if i >= num_samples:
            break

        sample_start = time.time()

        try:
            # Decode metadata
            t0 = time.time()
            import json
            metadata = json.loads(sample['json'].decode('utf-8'))
            study_id = metadata['study_id']
            times['metadata_decode'].append(time.time() - t0)

            # Decode volume
            t0 = time.time()
            volume_shape = tuple(metadata['volume_shape'])
            volume_dtype = np.dtype(metadata['volume_dtype'])
            volume_data = np.frombuffer(sample['bin'], dtype=volume_dtype).reshape(volume_shape)
            times['volume_decode'].append(time.time() - t0)

            # Decode report
            t0 = time.time()
            report_text = sample['txt'].decode('utf-8')
            times['report_decode'].append(time.time() - t0)

            # Decode labels
            t0 = time.time()
            labels = np.frombuffer(sample['labels'], dtype=np.float32)
            times['labels_decode'].append(time.time() - t0)

            # Now profile _process_volume step by step
            # Step 1: Convert float16 to float32
            t0 = time.time()
            img_data = volume_data.astype(np.float32)
            times['dtype_convert'].append(time.time() - t0)

            # Step 2: Get metadata
            slope = float(metadata["RescaleSlope"])
            intercept = float(metadata["RescaleIntercept"])
            xy_spacing_str = str(metadata["XYSpacing"])
            xy_spacing = float(xy_spacing_str.strip("[]").split(",")[0])
            z_spacing = float(metadata["ZSpacing"])

            # Step 3: Apply rescale slope and intercept
            t0 = time.time()
            img_data = slope * img_data + intercept
            times['rescale'].append(time.time() - t0)

            # Step 4: Clip to HU range
            t0 = time.time()
            hu_min, hu_max = -1000, 1000
            img_data = np.clip(img_data, hu_min, hu_max)
            times['clip'].append(time.time() - t0)

            # Step 5: Transpose to (D, H, W)
            t0 = time.time()
            img_data = img_data.transpose(2, 0, 1)
            times['transpose'].append(time.time() - t0)

            # Step 6: Convert to tensor and resize
            t0 = time.time()
            tensor = torch.tensor(img_data, dtype=torch.float32)
            tensor = tensor.unsqueeze(0).unsqueeze(0)  # (1, 1, D, H, W)

            # Resize (THIS IS THE KEY OPERATION)
            from src.data.webdataset_loader import resize_array
            target_x_spacing = 0.75
            target_y_spacing = 0.75
            target_z_spacing = 1.5
            current_spacing = (z_spacing, xy_spacing, xy_spacing)
            target_spacing = (target_z_spacing, target_x_spacing, target_y_spacing)

            img_data = resize_array(tensor, current_spacing, target_spacing)
            img_data = img_data[0][0]  # Remove batch and channel dims
            img_data = np.transpose(img_data, (1, 2, 0))  # (H, W, D)
            times['resize'].append(time.time() - t0)

            # Step 7: Normalize
            t0 = time.time()
            img_data = (img_data / 1000).astype(np.float32)
            times['normalize'].append(time.time() - t0)

            # Step 8: Convert back to tensor
            tensor = torch.tensor(img_data)

            # Step 9: Crop/pad to target shape
            t0 = time.time()
            target_shape = (480, 480, 240)
            h, w, d = tensor.shape
            dh, dw, dd = target_shape

            # Calculate crop/pad indices
            h_start = max((h - dh) // 2, 0)
            h_end = min(h_start + dh, h)
            w_start = max((w - dw) // 2, 0)
            w_end = min(w_start + dw, w)
            d_start = max((d - dd) // 2, 0)
            d_end = min(d_start + dd, d)

            # Crop
            tensor = tensor[h_start:h_end, w_start:w_end, d_start:d_end]

            # Pad if necessary
            pad_h_before = (dh - tensor.size(0)) // 2
            pad_h_after = dh - tensor.size(0) - pad_h_before
            pad_w_before = (dw - tensor.size(1)) // 2
            pad_w_after = dw - tensor.size(1) - pad_w_before
            pad_d_before = (dd - tensor.size(2)) // 2
            pad_d_after = dd - tensor.size(2) - pad_d_before

            tensor = torch.nn.functional.pad(
                tensor,
                (pad_d_before, pad_d_after, pad_w_before, pad_w_after, pad_h_before, pad_h_after),
                value=-1
            )
            times['crop_pad'].append(time.time() - t0)

            # Step 10: Permute and add channel dim
            t0 = time.time()
            tensor = tensor.permute(2, 0, 1)  # (D, H, W)
            tensor = tensor.unsqueeze(0)      # (1, D, H, W)
            times['permute'].append(time.time() - t0)

            times['total'].append(time.time() - sample_start)

            # Print progress
            if (i + 1) % 5 == 0:
                print(f"  Processed {i+1}/{num_samples} samples")

        except Exception as e:
            print(f"\n✗ Error processing sample {i+1}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Calculate statistics
    print("\n" + "="*80)
    print("Profiling Results (averaged over {} samples)".format(len(times['total'])))
    print("="*80)

    def print_stat(name, values, indent=0):
        if not values:
            return
        mean = np.mean(values) * 1000  # Convert to ms
        std = np.std(values) * 1000
        min_val = np.min(values) * 1000
        max_val = np.max(values) * 1000
        percent = (np.mean(values) / np.mean(times['total'])) * 100 if times['total'] else 0

        prefix = "  " * indent
        print(f"{prefix}{name:20s}: {mean:7.2f}ms ± {std:5.2f}ms  [{min_val:6.2f} - {max_val:6.2f}]  ({percent:5.1f}%)")

    print("\nOverall:")
    print_stat("Total per sample", times['total'])

    print("\nDetailed Breakdown:")
    print_stat("1. Metadata decode", times['metadata_decode'], 1)
    print_stat("2. Volume decode", times['volume_decode'], 1)
    print_stat("3. Report decode", times['report_decode'], 1)
    print_stat("4. Labels decode", times['labels_decode'], 1)
    print_stat("5. dtype convert", times['dtype_convert'], 1)
    print_stat("6. Rescale", times['rescale'], 1)
    print_stat("7. Clip", times['clip'], 1)
    print_stat("8. Transpose", times['transpose'], 1)
    print_stat("9. Resize", times['resize'], 1)  # KEY OPERATION
    print_stat("10. Normalize", times['normalize'], 1)
    print_stat("11. Crop/Pad", times['crop_pad'], 1)
    print_stat("12. Permute", times['permute'], 1)

    # Identify bottlenecks
    print("\n" + "="*80)
    print("Bottleneck Analysis")
    print("="*80)

    # Calculate which operations take > 10% of time
    bottlenecks = []
    for name, values in times.items():
        if name == 'total' or not values:
            continue
        percent = (np.mean(values) / np.mean(times['total'])) * 100
        if percent > 10:
            bottlenecks.append((name, percent, np.mean(values)))

    bottlenecks.sort(key=lambda x: x[1], reverse=True)

    if bottlenecks:
        print("\nOperations taking > 10% of time:")
        for name, percent, mean_time in bottlenecks:
            print(f"  ⚠ {name:20s}: {percent:5.1f}% ({mean_time*1000:.2f}ms)")

        print("\nRecommendations:")
        for name, percent, mean_time in bottlenecks:
            if 'resize' in name.lower():
                print(f"  • Resize is the bottleneck ({percent:.1f}%)")
                print(f"    → Consider GPU-based resize (Kornia, DALI)")
                print(f"    → Expected speedup: 10-20x (from {mean_time*1000:.0f}ms to {mean_time*1000/15:.0f}ms)")
            elif 'crop' in name.lower() or 'pad' in name.lower():
                print(f"  • Crop/Pad takes {percent:.1f}%")
                print(f"    → Can be done on GPU")
                print(f"    → Expected speedup: 5-10x")
            elif 'decode' in name.lower():
                print(f"  • {name} takes {percent:.1f}%")
                print(f"    → Already optimized (binary format)")
    else:
        print("\n✓ No major bottlenecks found (all operations < 10% of time)")
        print("  → Consider increasing batch_size or num_workers")

    print("="*80)

    return times


def main():
    # Load config
    config_path = project_root / "configs" / "base_config.yaml"
    config = load_config(str(config_path))

    # Create dataset
    print("\nCreating dataset...")
    dataset = CTReportWebDataset(
        shard_pattern=config['data']['webdataset_shards_train'],
        shuffle=False,
        buffer_size=0,
        mode="train"
    )

    # Profile
    times = profile_single_sample(dataset, num_samples=10)

    return 0


if __name__ == "__main__":
    exit(main())
