"""
CT-CLIP Inference Script

Full evaluation with bootstrap CI
"""

import argparse
import json
from pathlib import Path

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel

from src.models.ctvit import CTViT
from src.models.ct_clip import CTCLIP
from src.data.webdataset_loader import CTReportWebDataset
from src.validation import DiseaseEvaluator
from src.utils.config import load_config
from src.utils.seed import set_seed


def apply_softmax(array):
    """Apply softmax function"""
    softmax = torch.nn.Softmax(dim=0)
    return softmax(array)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CT-CLIP Inference")

    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to model checkpoint'
    )

    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='results/inference_results.json',
        help='Path to save results'
    )

    parser.add_argument(
        '--bootstrap',
        action='store_true',
        help='Use bootstrap to compute confidence intervals'
    )

    parser.add_argument(
        '--n_bootstrap',
        type=int,
        default=1000,
        help='Number of bootstrap samples'
    )

    return parser.parse_args()


def build_model(config: dict, device: torch.device):
    """Build CT-CLIP model"""
    model_config = config['model']

    # Text Encoder
    text_config = model_config['text_encoder']
    tokenizer = AutoTokenizer.from_pretrained(
        text_config['path'],
        do_lower_case=text_config['do_lower_case']
    )
    text_encoder = AutoModel.from_pretrained(text_config['path'], trust_remote_code=True)
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Image Encoder
    image_config = model_config['image_encoder'].copy()
    image_config.pop('type', None)
    image_encoder = CTViT(**image_config)

    # CLIP
    clip_config = model_config['clip']
    clip = CTCLIP(
        image_encoder=image_encoder,
        text_encoder=text_encoder,
        **clip_config
    )

    clip = clip.to(device)

    return clip


def load_checkpoint(model, checkpoint_path: str, device: torch.device):
    """Load model weights from checkpoint"""
    checkpoint = torch.load(checkpoint_path, map_location=device)

    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    print(f"Loaded checkpoint from {checkpoint_path}")
    if 'epoch' in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if 'global_step' in checkpoint:
        print(f"  Step: {checkpoint['global_step']}")
    if 'metrics' in checkpoint:
        metrics = checkpoint['metrics']
        if 'macro_auroc' in metrics:
            print(f"  AUROC: {metrics['macro_auroc']:.4f}")

    return model


def run_inference(
    model,
    dataloader,
    pathologies,
    tokenizer,
    device
):
    """
    Run inference on dataset

    Returns:
        predictions: (N, num_classes)
        labels: (N, num_classes)
    """
    model.eval()

    all_predictions = []
    all_labels = []

    print(f"\nRunning inference on {len(dataloader)} samples...")

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            volume_tensor, report_text, disease_labels, study_id, embed_tensor = batch
            volume_tensor = volume_tensor.to(device)

            # Predict for each pathology
            predicted_labels = []
            for pathology in pathologies:
                texts = [f"There is {pathology}.", f"There is no {pathology}."]
                text_tokens = tokenizer(
                    texts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=512
                ).to(device)

                output = model(text_tokens, volume_tensor, device=device)
                output = apply_softmax(output)

                predicted_labels.append(output[0].detach().cpu().numpy())

            all_predictions.append(predicted_labels)
            all_labels.append(disease_labels.detach().cpu().numpy()[0])

            if (batch_idx + 1) % 100 == 0:
                print(f"  Processed {batch_idx + 1}/{len(dataloader)} samples")

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    return all_predictions, all_labels


def main():
    """Main inference flow"""
    args = parse_args()

    # Load config
    print("="*80)
    print("CT-CLIP Inference")
    print("="*80)

    config = load_config(args.config)
    print(f"Config: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")

    # Set seed
    seed = config['experiment'].get('seed', 2025)
    set_seed(seed)

    # Set device
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['cuda_device']}")
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    # Build model
    print("\nBuilding model...")
    model = build_model(config, device)

    # Load checkpoint
    model = load_checkpoint(model, args.checkpoint, device)

    # Load pathology classes
    data_cfg = config['data']
    df = pd.read_csv(data_cfg['labels_valid'])
    exclude_cols = {'study_id', 'VolumeName'}
    pathologies = [col for col in df.columns if col not in exclude_cols]

    print(f"\nPathology classes ({len(pathologies)}): {pathologies}")

    # Tokenizer
    text_cfg = config['model']['text_encoder']
    tokenizer = AutoTokenizer.from_pretrained(
        text_cfg['path'],
        do_lower_case=text_cfg['do_lower_case']
    )

    # Validation dataset
    val_dataset = CTReportWebDataset(
        data_folder=data_cfg['valid_dir'],
        reports_file=data_cfg['reports_valid'],
        meta_file=data_cfg['valid_meta'],
        labels=data_cfg['labels_valid'],
        mode="val"
    )

    val_dataloader = DataLoader(
        val_dataset,
        num_workers=data_cfg['num_workers'],
        batch_size=1,
        shuffle=False,
        pin_memory=True,

    )

    print(f"Validation samples: {len(val_dataset)}")

    # Run inference
    predictions, labels = run_inference(
        model,
        val_dataloader,
        pathologies,
        tokenizer,
        device
    )

    # Evaluate
    print("\n" + "="*80)
    print("Evaluating...")
    print("="*80)

    evaluator = DiseaseEvaluator(
        pathology_classes=pathologies,
        metrics=["auroc", "auprc", "f1", "precision", "recall"],
        use_bootstrap=args.bootstrap,
        n_bootstrap=args.n_bootstrap,
        threshold=0.5
    )

    results = evaluator.evaluate(
        predictions,
        labels,
        return_per_class=True,
        verbose=True
    )

    # Print results
    print("\n" + "="*80)
    print("Results")
    print("="*80)
    print(evaluator.format_results(results))
    print("="*80)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        else:
            return obj

    results_serializable = convert_to_serializable(results)

    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
