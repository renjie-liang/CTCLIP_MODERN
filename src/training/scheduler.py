import math
from torch.optim.lr_scheduler import LambdaLR


def get_warmup_cosine_schedule(
    optimizer,
    warmup_steps: int,
    total_steps: int,
    min_lr_ratio: float = 0.01,
    last_epoch: int = -1
):
    """
    Learning rate scheduler with linear warmup and cosine decay

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        min_lr_ratio: Minimum lr as ratio of initial lr (default: 0.01)
        last_epoch: Last epoch for resuming
    """

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))

        # Cosine decay
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)


def get_linear_warmup_schedule(
    optimizer,
    warmup_steps: int,
    last_epoch: int = -1
):
    """
    Learning rate scheduler with only linear warmup, then constant

    Args:
        optimizer: PyTorch optimizer
        warmup_steps: Number of warmup steps
        last_epoch: Last epoch for resuming
    """

    def lr_lambda(current_step: int):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return 1.0

    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)
