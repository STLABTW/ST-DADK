"""
Seed 고정 유틸리티
"""
import random
import numpy as np
import torch


def set_seed(seed: int):
    """
    재현 가능한 결과를 위해 모든 랜덤 시드 고정
    
    Args:
        seed: 고정할 시드 값
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # 완벽한 재현성을 위해 (성능 저하 가능)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    print(f"[INFO] Seed set to {seed}")
