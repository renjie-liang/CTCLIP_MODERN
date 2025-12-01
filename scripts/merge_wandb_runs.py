"""
Merge two WandB runs into a single continuous run

This script downloads data from two separate WandB runs and merges them
into a new run with continuous metrics.

Usage:
    python scripts/merge_wandb_runs.py \
        --entity your-entity \
        --project ct-clip \
        --run1 abc123 \
        --run2 def456 \
        --new-name "merged-training-run"
"""

import argparse
import pandas as pd
from tqdm import tqdm
import wandb


def parse_args():
    parser = argparse.ArgumentParser(description="Merge two WandB runs")

    parser.add_argument(
        '--entity',
        type=str,
        required=True,
        help='WandB entity (username or team name)'
    )

    parser.add_argument(
        '--project',
        type=str,
        required=True,
        help='WandB project name'
    )

    parser.add_argument(
        '--run1',
        type=str,
        required=True,
        help='First run ID (earlier training)'
    )

    parser.add_argument(
        '--run2',
        type=str,
        required=True,
        help='Second run ID (resumed training)'
    )

    parser.add_argument(
        '--new-name',
        type=str,
        default='merged-run',
        help='Name for the new merged run'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview merge without creating new run'
    )

    return parser.parse_args()


def download_run_history(api, entity, project, run_id):
    """Download all metrics from a run"""
    print(f"\n📥 Downloading data from run: {run_id}")

    run = api.run(f"{entity}/{project}/{run_id}")
    history = run.history()

    print(f"  • Found {len(history)} data points")
    print(f"  • Step range: {history['_step'].min()} → {history['_step'].max()}")

    # Get run metadata
    metadata = {
        'name': run.name,
        'config': dict(run.config),
        'tags': run.tags,
        'notes': run.notes,
    }

    return history, metadata


def merge_histories(hist1, hist2):
    """Merge two run histories"""
    print(f"\n🔄 Merging data...")

    # Combine dataframes
    merged = pd.concat([hist1, hist2], ignore_index=True)

    # Sort by step
    merged = merged.sort_values('_step').reset_index(drop=True)

    # Remove duplicate steps (keep last)
    merged = merged.drop_duplicates(subset=['_step'], keep='last')

    print(f"  • Total data points: {len(merged)}")
    print(f"  • Step range: {merged['_step'].min()} → {merged['_step'].max()}")

    # Check for gaps
    steps = merged['_step'].values
    gaps = []
    for i in range(1, len(steps)):
        gap = steps[i] - steps[i-1]
        if gap > 100:  # Arbitrary threshold
            gaps.append((steps[i-1], steps[i], gap))

    if gaps:
        print(f"  ⚠ Warning: Found {len(gaps)} large gaps in steps:")
        for start, end, gap in gaps[:3]:  # Show first 3 gaps
            print(f"    • Gap of {gap} steps between {start} and {end}")
    else:
        print(f"  ✓ No large gaps detected")

    return merged


def create_merged_run(merged_data, metadata1, metadata2, project, new_name, dry_run=False):
    """Create new run with merged data"""

    if dry_run:
        print(f"\n🔍 DRY RUN - Would create new run with:")
        print(f"  • Name: {new_name}")
        print(f"  • Data points: {len(merged_data)}")
        print(f"  • Step range: {merged_data['_step'].min()} → {merged_data['_step'].max()}")
        return None

    print(f"\n📤 Creating new merged run: {new_name}")

    # Merge configs (prefer run2's config as it's more recent)
    merged_config = {**metadata1['config'], **metadata2['config']}

    # Merge tags
    merged_tags = list(set(metadata1['tags'] + metadata2['tags'] + ['merged']))

    # Create new run
    new_run = wandb.init(
        project=project,
        name=new_name,
        tags=merged_tags,
        config=merged_config,
        notes=f"Merged from runs: {metadata1['name']} + {metadata2['name']}"
    )

    print(f"  • New run created: {new_run.url}")
    print(f"  • Uploading {len(merged_data)} data points...")

    # Upload data
    for idx, row in tqdm(merged_data.iterrows(), total=len(merged_data), desc="Uploading"):
        # Convert row to dict
        metrics = row.to_dict()

        # Extract step
        step = int(metrics.pop('_step'))

        # Remove internal wandb columns
        metrics = {k: v for k, v in metrics.items()
                  if not k.startswith('_') and pd.notna(v)}

        # Log to wandb
        if metrics:  # Only log if there are metrics
            wandb.log(metrics, step=step)

    print(f"\n✅ Upload complete!")
    print(f"  • New run URL: {new_run.url}")

    new_run.finish()

    return new_run


def main():
    args = parse_args()

    print("="*80)
    print("WandB Run Merger")
    print("="*80)
    print(f"Entity: {args.entity}")
    print(f"Project: {args.project}")
    print(f"Run 1: {args.run1}")
    print(f"Run 2: {args.run2}")
    print(f"New name: {args.new_name}")

    if args.dry_run:
        print("\n⚠ DRY RUN MODE - No new run will be created")

    # Initialize API
    api = wandb.Api()

    # Download data from both runs
    hist1, meta1 = download_run_history(api, args.entity, args.project, args.run1)
    hist2, meta2 = download_run_history(api, args.entity, args.project, args.run2)

    # Merge histories
    merged = merge_histories(hist1, hist2)

    # Create new run
    new_run = create_merged_run(merged, meta1, meta2, args.project, args.new_name, args.dry_run)

    if not args.dry_run:
        print("\n" + "="*80)
        print("✅ Merge complete!")
        print("="*80)
        print(f"\nNext steps:")
        print(f"1. Check the new run: {new_run.url}")
        print(f"2. Verify the metrics look correct")
        print(f"3. (Optional) Delete old runs: {args.run1} and {args.run2}")
        print(f"\nTo delete old runs, go to:")
        print(f"  https://wandb.ai/{args.entity}/{args.project}/runs/{args.run1}")
        print(f"  https://wandb.ai/{args.entity}/{args.project}/runs/{args.run2}")

    print("\n" + "="*80)


if __name__ == '__main__':
    main()
