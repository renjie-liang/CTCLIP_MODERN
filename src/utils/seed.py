import random
import numpy as np
import torch


def set_seed(seed: int = 2025):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed (default: 2025)
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # For deterministic behavior (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
