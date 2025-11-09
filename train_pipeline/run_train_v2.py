"""
CT-CLIP训练脚本 V2 - 使用简洁的 CTClipTrainerV2

用法:
    # 使用默认配置
    python train_pipeline/run_train_v2.py

    # 使用自定义配置
    python train_pipeline/run_train_v2.py --config train_pipeline/configs/experiments/debug.yaml

    # 从checkpoint恢复
    python train_pipeline/run_train_v2.py --resume saves/best_model.pt

    # 指定GPU
    CUDA_VISIBLE_DEVICES=0 python train_pipeline/run_train_v2.py
"""

import sys
import argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from transformers import BertTokenizer, BertModel

from src.models.ctvit import CTViT
from ct_clip import CTCLIP
from CTCLIPTrainerV2 import CTClipTrainerV2
from configs import load_config


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="CT-CLIP Training V2")

    parser.add_argument(
        '--config',
        type=str,
        default='train_pipeline/configs/base_config.yaml',
        help='Path to config file'
    )

    parser.add_argument(
        '--resume',
        type=str,
        default=None,
        help='Path to checkpoint to resume from'
    )

    return parser.parse_args()


def build_model(config: dict, device: torch.device):
    """
    构建 CT-CLIP 模型

    Args:
        config: 配置字典
        device: 设备

    Returns:
        CT-CLIP 模型
    """
    model_config = config['model']

    # Text Encoder
    text_config = model_config['text_encoder']
    tokenizer = BertTokenizer.from_pretrained(
        text_config['path'],
        do_lower_case=text_config['do_lower_case']
    )
    text_encoder = BertModel.from_pretrained(text_config['path'])
    text_encoder.resize_token_embeddings(len(tokenizer))

    # Image Encoder (CTViT)
    image_config = model_config['image_encoder'].copy()
    image_config.pop('type', None)  # Remove 'type' field
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
    """主训练流程"""
    args = parse_args()

    # ========== 1. 加载配置 ==========
    print("="*80)
    print("Loading configuration...")
    print("="*80)

    config = load_config(args.config)

    print(f"Experiment: {config['experiment']['name']}")
    print(f"Config: {args.config}")

    # ========== 2. 设置设备 ==========
    device_config = config['device']
    if device_config['use_cuda'] and torch.cuda.is_available():
        device = torch.device(f"cuda:{device_config['cuda_device']}")
    else:
        device = torch.device('cpu')

    print(f"Device: {device}")

    # ========== 3. 构建模型 ==========
    print("\n" + "="*80)
    print("Building model...")
    print("="*80)

    model = build_model(config, device)

    # 统计参数
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # ========== 4. 创建 Trainer V2 ==========
    print("\n" + "="*80)
    print("Initializing CTClipTrainerV2...")
    print("="*80)

    trainer = CTClipTrainerV2(model, config)

    # ========== 5. 从checkpoint恢复（如果需要） ==========
    if args.resume:
        print("\n" + "="*80)
        print(f"Resuming from checkpoint: {args.resume}")
        print("="*80)
        trainer.load_checkpoint(args.resume)

    # ========== 6. 开始训练 ==========
    print("\n" + "="*80)
    print("Starting Training")
    print("="*80)
    print(f"Experiment: {config['experiment']['name']}")
    print(f"Epochs: {config['training']['num_epochs']}")
    print(f"Batch size: {config['data']['batch_size']}")
    print(f"Learning rate: {config['training']['learning_rate']}")
    print("="*80 + "\n")

    trainer.train()

    print("\n" + "="*80)
    print("Training Complete!")
    print(f"Best AUROC: {trainer.best_auroc:.4f}")
    print(f"Results saved to: {trainer.results_folder}")
    print("="*80 + "\n")


if __name__ == '__main__':
    main()
