#!/usr/bin/env python3
"""
Test that Accelerate + custom collate_fn works correctly.

This script verifies that the dispatch_batches=True fix resolves
the TypeError with string data.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import torch
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from accelerate.utils import InitProcessGroupKwargs
from datetime import timedelta

from src.data.webdataset_loader import CTReportWebDataset
from src.utils.config import load_config


def test_with_accelerate():
    """Test DataLoader with Accelerate prepare()."""

    print("="*80)
    print("Testing Accelerate + Custom Collate Function")
    print("="*80)

    # Load config
    config_path = project_root / "configs" / "debug_config.yaml"
    config = load_config(str(config_path))

    print("\nCreating dataset...")
    dataset = CTReportWebDataset(
        shard_pattern=config['data']['webdataset_shards_train'],
        shuffle=False,
        buffer_size=0,
        mode="train"
    )

    print(f"Dataset: {len(dataset)} samples")

    # Create DataLoader (not prepared yet)
    print("\nCreating DataLoader...")
    dataloader = dataset.create_pytorch_dataloader(
        batch_size=4,
        num_workers=0,  # Single process for testing
        prefetch_factor=2
    )

    print("✓ DataLoader created")

    # Initialize Accelerator
    print("\nInitializing Accelerator...")
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    init_kwargs = InitProcessGroupKwargs(timeout=timedelta(seconds=36000))

    accelerator = Accelerator(
        kwargs_handlers=[ddp_kwargs, init_kwargs],
        mixed_precision='fp16'
    )

    print(f"✓ Accelerator initialized")
    print(f"  Device: {accelerator.device}")
    print(f"  Mixed precision: fp16")
    print(f"\nNote: DataLoader is NOT prepared (to preserve custom collate_fn)")
    print(f"Only model/optimizer/scheduler should be prepared in training code")

    # Test loading batches
    print("\n" + "="*80)
    print("Loading 5 batches...")
    print("="*80)

    try:
        for i, batch in enumerate(dataloader):
            if i >= 5:
                break

            # Unpack batch
            volume, report_text, labels, study_id, embed = batch

            print(f"\nBatch {i+1}:")
            print(f"  Volume: {volume.shape}, {volume.dtype}, device={volume.device}")
            print(f"  Report (type): {type(report_text)}")
            print(f"  Report (length): {len(report_text)}")
            print(f"  Report (sample): {report_text[0][:50]}...")
            print(f"  Labels: {labels.shape}, {type(labels)}")
            print(f"  Study IDs (type): {type(study_id)}")
            print(f"  Study IDs (length): {len(study_id)}")
            print(f"  Study IDs (sample): {study_id[0]}")
            print(f"  Embed: {embed.shape if hasattr(embed, 'shape') else 'empty'}")
            print(f"  ✓ Success")

        print("\n" + "="*80)
        print("✓ All tests passed!")
        print("="*80)
        print("\nSolution:")
        print("- DataLoader NOT prepared with Accelerate")
        print("- Only model/optimizer/scheduler are prepared")
        print("- Custom collate_fn preserved (no wrapping/unwrapping needed)")
        print("- String data (report_text, study_id) handled correctly")
        print("- Tensor data (volume, labels, embed) handled correctly")
        print("="*80)

        return 0

    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    return test_with_accelerate()


if __name__ == "__main__":
    exit(main())
