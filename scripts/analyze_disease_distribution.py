#!/usr/bin/env python3
"""
Analyze disease distribution statistics for CT-RATE dataset.
Shows positive/negative distribution for all 18 diseases on:
1. Entire dataset (train + validation from CSV files)
2. Subset (train_subset_50 + val_subset_10 from webdataset tar files)
"""

import pandas as pd
import numpy as np
import json
import tarfile
from pathlib import Path
from typing import Dict, List
from collections import defaultdict


# 18 disease names in order
DISEASE_NAMES = [
    "Medical material",
    "Arterial wall calcification",
    "Cardiomegaly",
    "Pericardial effusion",
    "Coronary artery wall calcification",
    "Hiatal hernia",
    "Lymphadenopathy",
    "Emphysema",
    "Atelectasis",
    "Lung nodule",
    "Lung opacity",
    "Pulmonary fibrotic sequela",
    "Pleural effusion",
    "Mosaic attenuation pattern",
    "Peribronchial thickening",
    "Consolidation",
    "Bronchiectasis",
    "Interlobular septal thickening"
]


def load_labels_from_csv(csv_path: str) -> pd.DataFrame:
    """Load labels from CSV file."""
    return pd.read_csv(csv_path)


def load_labels_from_webdataset(shard_pattern: str) -> pd.DataFrame:
    """Load labels from webdataset tar files."""
    # Expand shard pattern
    base_path = Path(shard_pattern.split('{')[0])
    parent_dir = base_path.parent

    # Parse the shard range
    range_part = shard_pattern.split('{')[1].split('}')[0]
    start, end = range_part.split('..')

    all_labels = []
    all_study_ids = []

    print(f"  Loading from {parent_dir}...")
    for i in range(int(start), int(end) + 1):
        shard_path = parent_dir / f"shard-{i:06d}.tar"

        if not shard_path.exists():
            print(f"    Warning: {shard_path} not found, skipping...")
            continue

        with tarfile.open(shard_path, 'r') as tar:
            members = tar.getmembers()

            # Group files by sample
            samples = defaultdict(dict)
            for member in members:
                if member.name.endswith('.labels'):
                    key = member.name.replace('.labels', '')
                    samples[key]['labels'] = member
                elif member.name.endswith('.json'):
                    key = member.name.replace('.json', '')
                    samples[key]['json'] = member

            # Extract labels for each sample
            for key, files in samples.items():
                if 'labels' in files and 'json' in files:
                    # Read JSON for study_id
                    json_file = tar.extractfile(files['json'])
                    metadata = json.load(json_file)
                    study_id = metadata['study_id']

                    # Read labels (binary format, 18 float32 values)
                    labels_file = tar.extractfile(files['labels'])
                    labels = np.frombuffer(labels_file.read(), dtype=np.float32)

                    all_study_ids.append(study_id)
                    all_labels.append(labels)

    print(f"    Loaded {len(all_labels)} samples")

    # Create DataFrame
    labels_array = np.stack(all_labels)
    df = pd.DataFrame(labels_array, columns=DISEASE_NAMES)
    df.insert(0, 'VolumeName', all_study_ids)

    return df


def calculate_disease_stats(df: pd.DataFrame, disease_columns: List[str]) -> Dict:
    """Calculate positive/negative statistics for each disease."""
    stats = {}
    total_samples = len(df)

    for disease in disease_columns:
        positive = df[disease].sum()
        negative = total_samples - positive
        pos_ratio = positive / total_samples * 100
        neg_ratio = negative / total_samples * 100

        stats[disease] = {
            'positive': int(positive),
            'negative': int(negative),
            'pos_ratio': pos_ratio,
            'neg_ratio': neg_ratio,
            'total': total_samples
        }

    return stats


def print_statistics(stats: Dict, title: str):
    """Print statistics in a formatted table."""
    print("\n" + "=" * 100)
    print(f"{title}")
    print("=" * 100)
    print(f"{'Disease':<40} {'Positive':<12} {'Negative':<12} {'Pos %':<10} {'Neg %':<10}")
    print("-" * 100)

    for disease, data in stats.items():
        print(f"{disease:<40} {data['positive']:<12} {data['negative']:<12} "
              f"{data['pos_ratio']:<10.2f} {data['neg_ratio']:<10.2f}")

    # Print overall statistics
    total_samples = list(stats.values())[0]['total']
    total_positive = sum(data['positive'] for data in stats.values())
    total_negative = sum(data['negative'] for data in stats.values())
    avg_pos_per_sample = total_positive / total_samples
    avg_neg_per_sample = total_negative / total_samples

    print("-" * 100)
    print(f"{'TOTAL':<40} {total_positive:<12} {total_negative:<12}")
    print(f"Total samples: {total_samples}")
    print(f"Average positive labels per sample: {avg_pos_per_sample:.2f}")
    print(f"Average negative labels per sample: {avg_neg_per_sample:.2f}")
    print("=" * 100)


def compare_distributions(stats1: Dict, stats2: Dict, name1: str, name2: str):
    """Compare distributions between two datasets."""
    print("\n" + "=" * 120)
    print(f"Distribution Comparison: {name1} vs {name2}")
    print("=" * 120)
    print(f"{'Disease':<40} {name1 + ' Pos%':<15} {name2 + ' Pos%':<15} {'Difference':<15}")
    print("-" * 120)

    for disease in stats1.keys():
        pos_ratio1 = stats1[disease]['pos_ratio']
        pos_ratio2 = stats2[disease]['pos_ratio']
        diff = pos_ratio2 - pos_ratio1

        print(f"{disease:<40} {pos_ratio1:<15.2f} {pos_ratio2:<15.2f} {diff:+15.2f}")

    print("=" * 120)


def main():
    print("\n" + "=" * 100)
    print("CT-RATE Dataset - Disease Distribution Analysis")
    print("=" * 100)

    # ========== Load Entire Dataset ==========
    print("\n[1/2] Loading ENTIRE DATASET from CSV files...")
    train_csv_path = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/train_predicted_labels.csv"
    valid_csv_path = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/multi_abnormality_labels/valid_predicted_labels.csv"

    train_entire_df = load_labels_from_csv(train_csv_path)
    valid_entire_df = load_labels_from_csv(valid_csv_path)
    entire_df = pd.concat([train_entire_df, valid_entire_df], ignore_index=True)

    print(f"  Train: {len(train_entire_df)} samples")
    print(f"  Valid: {len(valid_entire_df)} samples")
    print(f"  Total: {len(entire_df)} samples")

    # ========== Load Subset ==========
    print("\n[2/2] Loading SUBSET from WebDataset tar files...")
    train_subset_pattern = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/train_subset_50/shard-{000000..000049}.tar"
    valid_subset_pattern = "/orange/xujie/liang.renjie/DATA/dataset/CT-RATE/dataset/val_subset_10/shard-{000000..000009}.tar"

    print("  Train subset (50 shards):")
    train_subset_df = load_labels_from_webdataset(train_subset_pattern)
    print("  Valid subset (10 shards):")
    valid_subset_df = load_labels_from_webdataset(valid_subset_pattern)
    subset_df = pd.concat([train_subset_df, valid_subset_df], ignore_index=True)

    print(f"  Train: {len(train_subset_df)} samples")
    print(f"  Valid: {len(valid_subset_df)} samples")
    print(f"  Total: {len(subset_df)} samples")

    # Get disease columns
    disease_columns = DISEASE_NAMES

    print(f"\n✓ Found {len(disease_columns)} diseases")

    # ========== Calculate Statistics ==========
    print("\n" + "=" * 100)
    print("Calculating statistics...")
    print("=" * 100)

    # Entire dataset statistics
    entire_stats = calculate_disease_stats(entire_df, disease_columns)
    print_statistics(entire_stats, "ENTIRE DATASET (Train + Validation)")

    # Subset statistics
    subset_stats = calculate_disease_stats(subset_df, disease_columns)
    print_statistics(subset_stats, "SUBSET (train_subset_50 + val_subset_10)")

    # Compare entire vs subset
    compare_distributions(entire_stats, subset_stats, "Entire", "Subset")

    # ========== Save Results ==========
    output_dir = Path("./analysis_results")
    output_dir.mkdir(exist_ok=True)

    # Create summary dataframe
    summary_data = []
    for disease in disease_columns:
        summary_data.append({
            'Disease': disease,
            'Entire_Positive': entire_stats[disease]['positive'],
            'Entire_Negative': entire_stats[disease]['negative'],
            'Entire_Pos%': entire_stats[disease]['pos_ratio'],
            'Entire_Total': entire_stats[disease]['total'],
            'Subset_Positive': subset_stats[disease]['positive'],
            'Subset_Negative': subset_stats[disease]['negative'],
            'Subset_Pos%': subset_stats[disease]['pos_ratio'],
            'Subset_Total': subset_stats[disease]['total'],
            'Pos%_Difference': subset_stats[disease]['pos_ratio'] - entire_stats[disease]['pos_ratio'],
        })

    summary_df = pd.DataFrame(summary_data)
    output_file = output_dir / "disease_distribution_comparison.csv"
    summary_df.to_csv(output_file, index=False)
    print(f"\n✓ Summary saved to: {output_file}")

    # Create detailed report
    report_file = output_dir / "disease_distribution_report.txt"
    with open(report_file, 'w') as f:
        f.write("CT-RATE Dataset - Disease Distribution Analysis\n")
        f.write("=" * 100 + "\n\n")

        f.write(f"Dataset Summary:\n")
        f.write(f"  Entire Dataset:\n")
        f.write(f"    Train samples: {len(train_entire_df)}\n")
        f.write(f"    Valid samples: {len(valid_entire_df)}\n")
        f.write(f"    Total samples: {len(entire_df)}\n")
        f.write(f"  Subset:\n")
        f.write(f"    Train samples: {len(train_subset_df)}\n")
        f.write(f"    Valid samples: {len(valid_subset_df)}\n")
        f.write(f"    Total samples: {len(subset_df)}\n")
        f.write(f"  Number of diseases: {len(disease_columns)}\n\n")

        for disease in disease_columns:
            f.write(f"\n{disease}\n")
            f.write("-" * 80 + "\n")
            f.write(f"  Entire Dataset:\n")
            f.write(f"    Positive: {entire_stats[disease]['positive']:>6} ({entire_stats[disease]['pos_ratio']:>6.2f}%)\n")
            f.write(f"    Negative: {entire_stats[disease]['negative']:>6} ({entire_stats[disease]['neg_ratio']:>6.2f}%)\n")
            f.write(f"  Subset:\n")
            f.write(f"    Positive: {subset_stats[disease]['positive']:>6} ({subset_stats[disease]['pos_ratio']:>6.2f}%)\n")
            f.write(f"    Negative: {subset_stats[disease]['negative']:>6} ({subset_stats[disease]['neg_ratio']:>6.2f}%)\n")
            diff = subset_stats[disease]['pos_ratio'] - entire_stats[disease]['pos_ratio']
            f.write(f"  Difference (Subset - Entire): {diff:+.2f}%\n")

    print(f"✓ Detailed report saved to: {report_file}")
    print(f"\n{'=' * 100}")
    print("Analysis complete!")
    print("=" * 100)


if __name__ == "__main__":
    main()
