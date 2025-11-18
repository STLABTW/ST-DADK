"""
평가 지표 계산 유틸리티
"""
import numpy as np
import torch
from typing import Dict, Union


def compute_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    per_horizon: bool = False
) -> Dict[str, float]:
    """
    회귀 평가 지표 계산 (RMSE, MAE, R²)
    
    Args:
        y_true: Ground truth, shape (B, H, S, 1) or (N,)
        y_pred: Predictions, shape (B, H, S, 1) or (N,)
        per_horizon: True면 허라이즌별 지표도 반환
        
    Returns:
        metrics: {'rmse': float, 'mae': float, 'r2': float, ...}
    """
    # Tensor to numpy
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Flatten
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    
    # Remove NaN if any
    valid_mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]
    
    # Compute metrics
    mse = np.mean((y_true_flat - y_pred_flat) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true_flat - y_pred_flat))
    
    # R²
    ss_res = np.sum((y_true_flat - y_pred_flat) ** 2)
    ss_tot = np.sum((y_true_flat - np.mean(y_true_flat)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))
    
    metrics = {
        'rmse': float(rmse),
        'mae': float(mae),
        'r2': float(r2),
        'mse': float(mse)
    }
    
    # Per-horizon metrics
    if per_horizon and len(y_true.shape) == 4:
        # y_true/y_pred shape: (B, H, S, 1)
        H = y_true.shape[1]
        rmse_per_h = []
        mae_per_h = []
        
        for h in range(H):
            yt_h = y_true[:, h, :, :].flatten()
            yp_h = y_pred[:, h, :, :].flatten()
            
            valid_mask_h = ~(np.isnan(yt_h) | np.isnan(yp_h))
            yt_h = yt_h[valid_mask_h]
            yp_h = yp_h[valid_mask_h]
            
            rmse_h = np.sqrt(np.mean((yt_h - yp_h) ** 2))
            mae_h = np.mean(np.abs(yt_h - yp_h))
            
            rmse_per_h.append(float(rmse_h))
            mae_per_h.append(float(mae_h))
        
        metrics['rmse_per_horizon'] = rmse_per_h
        metrics['mae_per_horizon'] = mae_per_h
    
    return metrics


def compute_spatial_metrics(
    y_true: Union[np.ndarray, torch.Tensor],
    y_pred: Union[np.ndarray, torch.Tensor],
    coords: np.ndarray,
    n_bins: int = 5
) -> Dict[str, list]:
    """
    공간 영역별 성능 분석 (원점으로부터 거리 기준)
    
    Args:
        y_true: Ground truth, shape (B, H, S, 1)
        y_pred: Predictions, shape (B, H, S, 1)
        coords: Site coordinates, shape (S, 2) with [x, y]
        n_bins: 거리 구간 개수
        
    Returns:
        spatial_metrics: {'distances': [...], 'rmse_by_dist': [...], ...}
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    
    # Compute distance from origin
    distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)  # (S,)
    
    # Create distance bins
    dist_bins = np.linspace(0, distances.max(), n_bins + 1)
    
    rmse_by_bin = []
    mae_by_bin = []
    bin_centers = []
    
    for i in range(n_bins):
        # Sites in this bin
        mask = (distances >= dist_bins[i]) & (distances < dist_bins[i+1])
        if not mask.any():
            continue
        
        # Extract predictions for these sites
        yt_bin = y_true[:, :, mask, :].flatten()
        yp_bin = y_pred[:, :, mask, :].flatten()
        
        valid_mask = ~(np.isnan(yt_bin) | np.isnan(yp_bin))
        yt_bin = yt_bin[valid_mask]
        yp_bin = yp_bin[valid_mask]
        
        if len(yt_bin) > 0:
            rmse_bin = np.sqrt(np.mean((yt_bin - yp_bin) ** 2))
            mae_bin = np.mean(np.abs(yt_bin - yp_bin))
        else:
            rmse_bin = np.nan
            mae_bin = np.nan
        
        rmse_by_bin.append(float(rmse_bin))
        mae_by_bin.append(float(mae_bin))
        bin_centers.append(float((dist_bins[i] + dist_bins[i+1]) / 2))
    
    return {
        'bin_centers': bin_centers,
        'rmse_by_distance': rmse_by_bin,
        'mae_by_distance': mae_by_bin
    }


def print_metrics(metrics: Dict[str, float], prefix: str = ""):
    """
    지표를 보기 좋게 출력
    
    Args:
        metrics: compute_metrics의 출력
        prefix: 출력 앞에 붙일 접두사 (예: "Train", "Val")
    """
    print(f"{prefix} Metrics:")
    print(f"  RMSE: {metrics['rmse']:.6f}")
    print(f"  MAE:  {metrics['mae']:.6f}")
    print(f"  R²:   {metrics['r2']:.6f}")
    
    if 'rmse_per_horizon' in metrics:
        print(f"  RMSE per horizon: {metrics['rmse_per_horizon']}")
