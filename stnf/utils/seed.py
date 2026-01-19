"""
Seed fixing utility
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    Fix all random seeds for reproducible results
    
    Args:
        seed: Seed value to fix
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For perfect reproducibility (may degrade performance)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Seed set to {seed}")
