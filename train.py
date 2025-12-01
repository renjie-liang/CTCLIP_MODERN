"""
CT-CLIP Training Script

Step-based training with multi-GPU support
"""

import argparse
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModel

from src.models.ctvit import CTViT
from src.models.ct_clip import CTCLIP
from src.training.trainer import CTClipTrainer
from src.utils.config import load_config
from src.utils.seed import set_seed


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="CT-CLIP Training")

    parser.add_argument(
        '--config',
        type=str,
        default='configs/base_config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    parser.add_argument(
        '--no-load-optimizer',
        action='store_true',
        help='Do not load optimizer state from checkpoint (use when resuming from different model architecture)'
    )

    parser.add_argument(
        '--debug',
        action='store_true',
        help='Enable debug mode: disable wandb, enable profiling, reduce validation samples, verbose logging'
    )

    return parser.parse_args()


def build_model(config: dict, device: torch.device):
    """
    Build CT-CLIP model

    Args:
        config: Config dict
        device: Device

    Returns:
        CT-CLIP model
    """
    model_config = config['model']

    # Text Encoder
    text_config = model_config['text_encoder']
    tokenizer = AutoTokenizer.from_pretrained(
        text_config['path'],
        do_lower_case=text_config['do_lower_case'],
        trust_remote_code=True
    )
    text_encoder = AutoModel.from_pretrained(text_config['path'], trust_remote_code=True)
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Image Encoder (CTViT)
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


def main():
    """Main training flow"""
    args = parse_args()

    # Load config
    print("="*80)
    print("Loading configuration...")
    print("="*80)

    config = load_config(args.config)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Config: {args.config}")

    # Apply debug mode overrides
    if args.debug:
        print("\n" + "="*80)
        print("🐛 DEBUG MODE ENABLED")
        print("="*80)
        print("Debug settings:")
        print("  • Wandb: DISABLED (logs saved locally only)")
        print("  • Performance profiling: ENABLED (detailed timing every 100 steps)")
        print("  • Validation samples: 100 (faster validation)")
        print("  • Logging frequency: Every 1 step (verbose output)")
        print("  • Data loading: num_workers=8, prefetch_factor=2 (faster I/O)")
        print("="*80 + "\n")

        # Disable wandb
        config['logging']['use_wandb'] = False

        # Enable performance profiling
        config['logging']['profile_timing'] = True

        # Reduce validation samples for faster debugging
        config['validation']['eval_samples'] = 100

        # More frequent logging
        config['logging']['log_every_n_steps'] = 1

        # Increase data loading parallelism
        config['data']['num_workers'] = 8
        config['data']['prefetch_factor'] = 2

    # Set seed
    seed = config['experiment'].get('seed', 2025)
    set_seed(seed)
    print(f"Random seed: {seed}")

    # Set device
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['cuda_device']}")
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    # Build model
    print("\n" + "="*80)
    print("Building model...")
    print("="*80)

    model = build_model(config, device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create Trainer
    print("\n" + "="*80)
    print("Initializing Trainer...")
    print("="*80)

    trainer = CTClipTrainer(model, config)

    # Resume from checkpoint if needed
    if args.resume:
        print("\n" + "="*80)
        print(f"Resuming from checkpoint: {args.resume}")
        if args.no_load_optimizer:
            print("⚠ WARNING: Optimizer state will NOT be loaded (--no-load-optimizer)")
        print("="*80)
        trainer.load_checkpoint(args.resume, load_optimizer=not args.no_load_optimizer)

    # Start training
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Max steps: {config['training']['max_steps']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print(f"Warmup steps: {config['training']['warmup_steps']}")
    print("="*80 + "\n")

    trainer.train()

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best AUROC: {trainer.best_auroc:.4f}")
    print(f"Results saved to: {trainer.results_folder}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
