"""
Spatio-Temporal Interpolation Model Training Script

Simple MLP-based spatio-temporal prediction:
- Input: (X, s=(x,y), t)
- Embedding: φ(s) + ψ(t)
- Output: ŷ(s,t)

Usage:
    python scripts/train_st_interp.py --config configs/config_st_interp.yaml
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib
from datetime import datetime
import json
import time
from scipy.interpolate import griddata

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from stnf.models.st_interp import STInterpMLP, create_model
from stnf.dataio.kaust_loader import load_kaust_csv_single, load_kaust_csv_with_test_gt
from stnf.utils import set_seed, compute_metrics, ModelEMA
from stnf.utils.conformal import (
    compute_cqr_qhat,
    compute_conformal_coverage,
    compute_cluster_aware_cqr,
    compute_cluster_conformal_coverage,
    _assign_nearest_center,
)


def quantile_loss(y_pred, y_true, quantile):
    """
    Compute quantile (check) loss for quantile regression
    
    Args:
        y_pred: predicted values (B,)
        y_true: true values (B,)
        quantile: target quantile level (e.g., 0.1, 0.5, 0.9)
    
    Returns:
        Quantile loss
    """
    errors = y_true - y_pred
    return torch.mean(torch.max((quantile - 1) * errors, quantile * errors))


def non_crossing_penalty(y_pred_multi_q: torch.Tensor, reduction: str = "mean", power: int = 1):
    """
    Penalize quantile crossing for multi-quantile outputs (prediction-level penalty).

    Given predicted quantiles \hat{q}_1, ..., \hat{q}_Q (in increasing tau order),
    crossing happens when \hat{q}_k > \hat{q}_{k+1}. We penalize positive violations:

        P_nc = sum_{k=1}^{Q-1} ReLU(\hat{q}_k - \hat{q}_{k+1})^power

    Args:
        y_pred_multi_q: (B, Q) predicted quantiles in increasing tau order
        reduction: "mean" or "sum" over batch
        power: 1 (hinge) or 2 (squared hinge)

    Returns:
        Scalar penalty tensor.
    """
    if y_pred_multi_q.dim() != 2 or y_pred_multi_q.shape[1] < 2:
        return torch.tensor(0.0, device=y_pred_multi_q.device)

    diffs = y_pred_multi_q[:, :-1] - y_pred_multi_q[:, 1:]  # (B, Q-1)
    violations = torch.relu(diffs)
    if power == 2:
        violations = violations ** 2
    elif power != 1:
        raise ValueError(f"Unsupported power={power}; use 1 or 2.")

    per_sample = violations.sum(dim=1)  # (B,)
    if reduction == "mean":
        return per_sample.mean()
    if reduction == "sum":
        return per_sample.sum()
    raise ValueError(f"Unsupported reduction='{reduction}'; use 'mean' or 'sum'.")


def compute_p_nc_delta_penalty(delta_params: list, use_positive_penalty: bool = False) -> torch.Tensor:
    """
    Compute P_nc(δ) penalty on δ parameters as defined in Section 3.2 (Equation 3.10).
    
    Original formula (use_positive_penalty=False):
        For each quantile k = 2, ..., Q:
            J(δ_k) = δ_k,0 - max(δ_k,0, Σ_{j=1}^d max(0, -δ_k,j))
        Then: P_nc(δ) = Σ_{k=2}^Q J(δ_k)
    
    NOTE: Original formula always gives J(δ_k) ≤ 0, which when added to loss as
    loss + λ * P_nc(δ) makes loss more negative (encourages more negative J).
    This may cause optimization issues.
    
    Corrected formula (use_positive_penalty=True):
        For each quantile k = 2, ..., Q:
            J(δ_k) = max(0, Σ_{j=1}^d max(0, -δ_k,j) - δ_k,0)
        Then: P_nc(δ) = Σ_{k=2}^Q J(δ_k)
    
    This penalizes violations (positive values) rather than rewarding feasibility
    (negative values), which is more standard for penalty terms.
    
    Args:
        delta_params: List of δ_k Parameter tensors, each of shape (d+1,) where:
            - δ_k[0] is the intercept δ_k,0
            - δ_k[1:] are feature coefficients δ_k,1, ..., δ_k,d
            - d is the last hidden layer dimension
        use_positive_penalty: If True, use corrected positive penalty formula.
                              If False, use original formula (for backward compatibility).
    
    Returns:
        Scalar penalty tensor P_nc(δ)
    """
    if delta_params is None or len(delta_params) < 2:
        # Need at least 2 quantiles for non-crossing penalty
        if delta_params and len(delta_params) > 0:
            device = delta_params[0].device
        else:
            device = torch.device('cpu')
        return torch.tensor(0.0, device=device)
    
    Q = len(delta_params)
    penalty = torch.tensor(0.0, device=delta_params[0].device)
    
    # Compute J(δ_k) for k = 2, ..., Q (k=1 doesn't need penalty)
    for k in range(1, Q):  # k=1 means δ_2 (second quantile, index 1)
        delta_k = delta_params[k]  # (d+1,) Parameter tensor
        
        # Extract intercept and feature coefficients
        delta_k_0 = delta_k[0]  # δ_k,0 (intercept)
        delta_k_features = delta_k[1:]  # δ_k,1, ..., δ_k,d (feature coefficients)
        
        # Compute Σ_{j=1}^d max(0, -δ_k,j)
        negative_features = torch.clamp(-delta_k_features, min=0.0)  # max(0, -δ_k,j)
        sum_negative = negative_features.sum()  # Σ_{j=1}^d max(0, -δ_k,j)
        
        if use_positive_penalty:
            # Corrected formula: penalize violations (positive penalty)
            # J(δ_k) = max(0, Σ_{j=1}^d max(0, -δ_k,j) - δ_k,0)
            J_delta_k = torch.clamp(sum_negative - delta_k_0, min=0.0)
        else:
            # Original formula: J(δ_k) = δ_k,0 - max(δ_k,0, Σ_{j=1}^d max(0, -δ_k,j))
            max_term = torch.max(delta_k_0, sum_negative)
            J_delta_k = delta_k_0 - max_term
        
        penalty = penalty + J_delta_k
    
    return penalty


def check_loss_numpy(y_pred, y_true, quantile):
    """
    Compute quantile (check) loss using numpy (for CRPS computation).
    
    Args:
        y_pred: predicted values (N,)
        y_true: true values (N,)
        quantile: target quantile level (e.g., 0.1, 0.5, 0.9)
    
    Returns:
        Mean check loss (scalar)
    """
    errors = y_true - y_pred
    return np.mean(np.maximum((quantile - 1) * errors, quantile * errors))


def compute_crps(predictions_dict, y_true, weights=None):
    """
    Compute Continuous Ranked Probability Score (CRPS) from quantile predictions
    using Equation 4.6 from the thesis.
    
    CRPS(F, y) = 2 * Σ_k w_k ρ_{τ_k}(y - Q_{τ_k})
    
    where:
    - ρ_{τ_k} is the check loss (quantile loss)
    - w_k are quadrature weights (default: uniform weights, w_k = 1/K)
    - 2× scaling factor as per Equation 4.6
    
    Args:
        predictions_dict: dict of {quantile_level: predictions (N,)}
        y_true: true values (N,)
        weights: optional array of quadrature weights (default: uniform weights)
                 If None, uses uniform weights w_k = 1/K where K is number of quantiles.
    
    Returns:
        CRPS score (lower is better)
    """
    # Sort quantiles
    quantiles = sorted(predictions_dict.keys())
    K = len(quantiles)
    
    if K == 0:
        raise ValueError("predictions_dict cannot be empty")
    
    if K == 1:
        # Single quantile: use check loss with 2× scaling
        q = quantiles[0]
        check_loss = check_loss_numpy(predictions_dict[q], y_true, q)
        return 2.0 * check_loss
    
    # Use uniform weights if not provided (as per thesis Section 4.2.2)
    if weights is None:
        weights = np.ones(K) / K  # Uniform weights: w_k = 1/K
    else:
        weights = np.asarray(weights)
        if len(weights) != K:
            raise ValueError(f"weights length ({len(weights)}) must match number of quantiles ({K})")
        # Normalize weights to sum to 1
        weights = weights / weights.sum()
    
    # Compute CRPS using Equation 4.6: CRPS(F, y) = 2 * Σ_k w_k ρ_{τ_k}(y - Q_{τ_k})
    crps_sum = 0.0
    for i, q in enumerate(quantiles):
        pred = predictions_dict[q]
        check_loss_q = check_loss_numpy(pred, y_true, q)
        crps_sum += weights[i] * check_loss_q
    
    # Apply 2× scaling factor as per Equation 4.6
    crps = 2.0 * crps_sum
    
    return crps


def compute_crps_multi_quantile(preds, y_true, quantile_levels, weights=None):
    """
    Compute CRPS for multi-quantile model output using Equation 4.6.
    
    Args:
        preds: predictions array (N, Q) where Q is number of quantiles
        y_true: true values (N,) or (N, 1)
        quantile_levels: list of quantile levels [q1, q2, ..., qQ]
        weights: optional array of quadrature weights (default: uniform weights)
    
    Returns:
        CRPS score (lower is better)
    """
    # Ensure y_true is 1D
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    
    # Convert to dict format for compute_crps
    predictions_dict = {}
    for i, q in enumerate(quantile_levels):
        predictions_dict[q] = preds[:, i]
    
    return compute_crps(predictions_dict, y_true, weights=weights)


def compute_coverage(preds, y_true, quantile_levels, alpha=0.1):
    """
    Empirical coverage of a prediction interval formed by the quantile_levels
    *closest* to alpha/2 and 1 - alpha/2 (not necessarily exact 0.05/0.95).

    E.g. alpha=0.1: uses the predicted quantiles nearest to 0.05 and 0.95.
    If quantile_levels = [0.1, 0.5, 0.9], the interval is [q_0.1, q_0.9] (80% nominal),
    not 90%. For a true 90% PI, use quantile_levels that include 0.05 and 0.95
    (e.g. [0.05, 0.25, 0.5, 0.75, 0.95]).

    Args:
        preds: (N, Q) quantile predictions
        y_true: (N,) or (N, 1) true values
        quantile_levels: list of quantile levels (e.g. [0.05, 0.25, 0.5, 0.75, 0.95])
        alpha: nominal miscoverage (0.1 -> interval targets 90% nominal)

    Returns:
        coverage: fraction of y_true inside the interval; well-calibrated ~ (1 - alpha)
        when quantile_levels include alpha/2 and 1 - alpha/2.
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    # Find indices of quantile_levels closest to q_lo, q_hi
    quantile_levels = np.asarray(quantile_levels)
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    low = preds[:, idx_lo]
    high = preds[:, idx_hi]
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))


def _get_quantile_predictions(model, data_loader, device):
    """
    Run model on data_loader and return (preds, trues) for multi-quantile.
    preds: (N, Q), trues: (N,). Returns ([], []) if loader is empty.
    """
    model.eval()
    all_preds = []
    all_trues = []
    with torch.no_grad():
        for batch in data_loader:
            X = batch['X'].to(device)
            coords = batch['coords'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            y_pred = model(X, coords, t)
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
    if len(all_preds) == 0:
        return np.empty((0, 0)), np.empty(0)
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    if trues.ndim > 1:
        trues = trues.flatten()
    return preds, trues


def _get_quantile_predictions_with_coords(model, data_loader, device):
    """Like _get_quantile_predictions but also return coords (N, 2) for cluster assignment."""
    model.eval()
    all_preds, all_trues, all_coords = [], [], []
    with torch.no_grad():
        for batch in data_loader:
            X = batch['X'].to(device)
            coords = batch['coords'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            y_pred = model(X, coords, t)
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
            all_coords.append(coords.cpu().numpy())
    if len(all_preds) == 0:
        return np.empty((0, 0)), np.empty(0), np.empty((0, 2))
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    coords = np.concatenate(all_coords, axis=0)
    if trues.ndim > 1:
        trues = trues.flatten()
    if coords.ndim == 1:
        coords = coords.reshape(-1, 2)
    return preds, trues, coords


def create_spatial_obs_prob_fn(pattern='uniform', intensity=1.0):
    """
    Create spatial observation probability function
    
    Args:
        pattern: 'uniform', 'corner', or custom function
        intensity: intensity parameter controlling the degree of non-uniformity
                   For 'corner': higher values = stronger concentration near (0,0)
    
    Returns:
        obs_prob_fn: function(coord) -> probability
    """
    if pattern == 'uniform' or pattern is None:
        return None
    
    elif pattern == 'corner':
        # Paper formula: p(s) ∝ (1 + 10||s||)^{-2}
        # Use distance (not distance squared) and coefficient 10
        # Note: intensity parameter is ignored for paper formula (always use 10)
        def obs_prob_fn(coord):
            x, y = coord
            dist = np.sqrt(x**2 + y**2)  # Use distance, not distance squared
            # Paper formula: (1 + 10||s||)^{-2} - always use 10, ignore intensity
            prob = 1.0 / (1.0 + 10.0 * dist)**2
            return prob
        return obs_prob_fn
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def sample_observations(z_data, coords, obs_method='site-wise', obs_ratio=0.5, 
                       obs_prob_fn=None, seed=None, config=None):
    """
    Sample observations from full data
    
    Args:
        z_data: (T, S) - full spatio-temporal data
        coords: (S, 2) - spatial coordinates in [0,1]^2
        obs_method: 'site-wise' or 'random'
        obs_ratio: observation ratio (used as base rate or for site-wise)
        obs_prob_fn: function(coords) -> probs, maps [0,1]^2 to observation probability
                     If None, uniform probability is used
        seed: random seed
    
    Returns:
        obs_mask: (T, S) boolean array indicating observed locations
        obs_sites: list of observed site indices (for site-wise method)
    """
    if seed is not None:
        np.random.seed(seed)
    
    T, S = z_data.shape
    
    # Compute observation probabilities per site
    if obs_prob_fn is not None:
        # Apply function to each coordinate to get relative weights
        obs_weights = np.array([obs_prob_fn(coords[i]) for i in range(S)])
        # Normalize and scale by obs_ratio to get actual probabilities
        obs_weights_normalized = obs_weights / obs_weights.mean()
        obs_probs = obs_weights_normalized * obs_ratio
        # Clip to [0, 1] range
        obs_probs = np.clip(obs_probs, 0, 1)
    else:
        # Uniform probability
        obs_probs = np.ones(S) * obs_ratio
    
    if obs_method == 'site-wise':
        # Select obs_ratio fraction of sites, observe all times
        # Use obs_probs as sampling weights
        n_obs_sites = int(S * obs_ratio)
        obs_weights_normalized = obs_probs / obs_probs.sum()
        obs_sites = np.random.choice(S, size=n_obs_sites, replace=False, p=obs_weights_normalized)
        
        obs_mask = np.zeros((T, S), dtype=bool)
        obs_mask[:, obs_sites] = True
        
        return obs_mask, obs_sites
    
    elif obs_method == 'random':
        # Paper: Each time point samples exactly 10% of sites (fixed per time)
        # Not Bernoulli sampling (which would have variable counts per time)
        # Note: obs_ratio is passed as parameter, not from config
        n_obs_per_time = int(S * obs_ratio)
        
        obs_mask = np.zeros((T, S), dtype=bool)
        obs_sites = set()
        
        # Normalize obs_probs to sum to 1 for proper sampling
        obs_probs_norm = obs_probs / (obs_probs.sum() + 1e-10)
        
        for t in range(T):
            # Sample exactly n_obs_per_time sites at each time point
            # Use obs_probs as weights (for uniform, all equal; for clustered, weighted by distance)
            sampled_sites = np.random.choice(S, size=n_obs_per_time, replace=False, p=obs_probs_norm)
            obs_mask[t, sampled_sites] = True
            obs_sites.update(sampled_sites)
        
        obs_sites = np.array(list(obs_sites))
        
        return obs_mask, obs_sites
    
    else:
        raise ValueError(f"Unknown obs_method: {obs_method}")


def split_train_valid(obs_mask, obs_sites, split_method='site-wise', train_ratio=0.8, seed=None):
    """
    Split observed data into train and validation sets
    
    Args:
        obs_mask: (T, S) boolean array of observations
        obs_sites: list of observed site indices
        split_method: 'site-wise' or 'random'
        train_ratio: train/valid split ratio
        seed: random seed
    
    Returns:
        train_mask: (T, S) boolean array for training
        valid_mask: (T, S) boolean array for validation
    """
    if seed is not None:
        np.random.seed(seed)
    
    T, S = obs_mask.shape
    
    if split_method == 'site-wise':
        # Split sites into train and valid
        n_train_sites = int(len(obs_sites) * train_ratio)
        shuffled_sites = obs_sites.copy()
        np.random.shuffle(shuffled_sites)
        
        train_sites = shuffled_sites[:n_train_sites]
        valid_sites = shuffled_sites[n_train_sites:]
        
        train_mask = np.zeros((T, S), dtype=bool)
        valid_mask = np.zeros((T, S), dtype=bool)
        
        # Assign all observations of train_sites to train
        train_mask[:, train_sites] = obs_mask[:, train_sites]
        # Assign all observations of valid_sites to valid
        valid_mask[:, valid_sites] = obs_mask[:, valid_sites]
        
        return train_mask, valid_mask
    
    elif split_method == 'random':
        # Randomly split each observation
        # Get all observed (t, s) pairs
        obs_indices = np.argwhere(obs_mask)  # (N, 2) array of (t, s)
        n_obs = len(obs_indices)
        n_train = int(n_obs * train_ratio)
        
        # Shuffle and split
        shuffled_idx = np.random.permutation(n_obs)
        train_idx = shuffled_idx[:n_train]
        valid_idx = shuffled_idx[n_train:]
        
        train_mask = np.zeros((T, S), dtype=bool)
        valid_mask = np.zeros((T, S), dtype=bool)
        
        for idx in train_idx:
            t, s = obs_indices[idx]
            train_mask[t, s] = True
        
        for idx in valid_idx:
            t, s = obs_indices[idx]
            valid_mask[t, s] = True
        
        return train_mask, valid_mask
    
    else:
        raise ValueError(f"Unknown split_method: {split_method}")


def create_dataset_from_mask(z_data, coords, mask, p_covariates=0):
    """
    Create dataset from observation mask
    
    Args:
        z_data: (T, S) - full data
        coords: (S, 2) - coordinates in [0,1]^2
        mask: (T, S) - boolean mask
        p_covariates: number of covariates
    
    Returns:
        dataset: list of samples
    """
    T, S = z_data.shape
    
    dataset = []
    obs_indices = np.argwhere(mask)  # (N, 2) array of (t, s)
    
    for t_idx, site_idx in obs_indices:
        y_val = z_data[t_idx, site_idx]
        
        # Skip NaN values
        if np.isnan(y_val):
            continue
        
        # Time: normalize to [0,1] for model input
        t_normalized = t_idx / (T - 1) if T > 1 else 0.0
        X = np.zeros(p_covariates, dtype=np.float32)
        
        sample = {
            'X': torch.from_numpy(X).float(),
            'coords': torch.from_numpy(coords[site_idx]).float(),  # Already in [0,1]^2
            't': torch.tensor([t_normalized], dtype=torch.float32),  # Normalized to [0,1]
            'y': torch.tensor([y_val], dtype=torch.float32)
        }
        dataset.append(sample)
    
    return dataset


def collate_fn(batch):
    """Collate function for DataLoader"""
    X = torch.stack([item['X'] for item in batch])
    coords = torch.stack([item['coords'] for item in batch])
    t = torch.stack([item['t'] for item in batch])
    y = torch.stack([item['y'] for item in batch])
    
    return {'X': X, 'coords': coords, 't': t, 'y': y}


def train_model(model, train_loader, val_loader, config, device, output_dir):
    """Train the spatio-temporal interpolation model"""
    
    # Progressive unfreezing settings
    basis_unfreeze_epoch = config.get('basis_unfreeze_epoch', 0)  # 0 = train from start
    basis_lr_rampup_epochs = config.get('basis_lr_rampup_epochs', 0)  # epochs to gradually increase basis lr
    
    if config.get('spatial_learnable', False):
        # Learnable basis: use differential learning rates
        basis_params = list(model.spatial_basis.parameters())
        mlp_params = [p for p in model.parameters() if not any(p is bp for bp in basis_params)]
        
        lr = float(config.get('lr', 1e-3))
        basis_lr_ratio = config.get('basis_lr_ratio', 0.05)  # Default: 5% of MLP lr
        
        # If unfreezing later, start with lr=0 for basis
        initial_basis_lr = 0.0 if basis_unfreeze_epoch > 0 else lr * basis_lr_ratio
        
        optimizer = optim.AdamW([
            {'params': mlp_params, 'lr': lr, 'name': 'mlp'},
            {'params': basis_params, 'lr': initial_basis_lr, 'name': 'basis'}
        ], weight_decay=float(config.get('weight_decay', 1e-5)))
        
        # Store initial learning rates for warmup and unfreezing
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
            if param_group.get('name') == 'basis':
                param_group['target_lr'] = lr * basis_lr_ratio  # Target lr after unfreezing

        if basis_unfreeze_epoch > 0:
            print(f"Spatial basis: LEARNABLE (Progressive unfreezing)")
            print(f"  - Epoch 0-{basis_unfreeze_epoch-1}: Basis FROZEN (lr=0)")
            print(f"  - Epoch {basis_unfreeze_epoch}+: Basis unfrozen (target lr={lr*basis_lr_ratio:.2e})")
            if basis_lr_rampup_epochs > 0:
                print(f"  - Rampup over {basis_lr_rampup_epochs} epochs")
        else:
            print(f"Spatial basis: LEARNABLE (MLP lr={lr:.2e}, Basis lr={lr*basis_lr_ratio:.2e})")
    else:
        # Fixed basis: only optimize MLP parameters
        # model.spatial_basis has no parameters when learnable=False (uses buffers)
        mlp_params = [p for p in model.parameters() if p.requires_grad]
        
        lr = float(config.get('lr', 1e-3))
        optimizer = optim.AdamW(
            mlp_params,
            lr=lr,
            weight_decay=float(config.get('weight_decay', 1e-5))
        )
        
        # Store initial learning rates for warmup
        for param_group in optimizer.param_groups:
            param_group['initial_lr'] = param_group['lr']
        
        print(f"Spatial basis: FIXED (only MLP trained, lr={lr:.2e})")
    
    # Warmup settings
    warmup_epochs = config.get('warmup_epochs', 0)
    warmup_steps = warmup_epochs * len(train_loader) if warmup_epochs > 0 else 0
    
    # Scheduler
    scheduler = None
    if config.get('scheduler') == 'cosine':
        eta_min = lr * 0.5  # Minimum LR = 50% of initial LR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('epochs', 100),
            eta_min=eta_min
        )
        print(f"Cosine scheduler: lr={lr:.2e} → eta_min={eta_min:.2e} (50% of initial)")
    
    if warmup_steps > 0:
        print(f"Using warmup: {warmup_epochs} epochs ({warmup_steps} steps)")
    
    # Initialize EMA (can be disabled via config)
    use_ema = config.get('use_ema', True)  # Default to True for backward compatibility
    ema = None
    if use_ema:
        # Decay = 1 - 1/(10 * batches_per_epoch)
        batches_per_epoch = len(train_loader)
        ema_decay = 1.0 - 1.0 / (10.0 * batches_per_epoch)
        ema = ModelEMA(model, decay=ema_decay)
        print(f"EMA initialized: decay={ema_decay:.6f} (batches_per_epoch={batches_per_epoch})")
    else:
        print("EMA disabled (use_ema=False)")
    
    # Loss function based on regression type
    regression_type = config.get('regression_type', 'mean')
    current_quantile = config.get('current_quantile', None)  # Single quantile for this run
    quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])  # For multi-quantile
    
    if regression_type == 'mean':
        criterion = nn.MSELoss()
        print(f"Loss function: MSE (mean regression)")
    elif regression_type == 'quantile':
        if current_quantile is None:
            raise ValueError("current_quantile must be specified for quantile regression")
        print(f"Loss function: Quantile loss (quantile={current_quantile})")
    elif regression_type == 'multi-quantile':
        print(f"Loss function: Multi-quantile loss (quantiles={quantile_levels})")
        quantile_levels_tensor = torch.tensor(quantile_levels, device=device).float()
    else:
        raise ValueError(f"Unknown regression_type: {regression_type}")
    
    epochs = config.get('epochs', 100)
    best_val_loss = float('inf')
    patience = config.get('patience', 15)
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': [],
        'lr': []
    }
    
    # Track basis centers evolution every 100 epochs
    basis_centers_history = []  # List of (epoch, centers) tuples
    center_record_interval = 100
    
    best_model_path = output_dir / 'model_best.pt'
    global_step = 0  # Track global training step for warmup
    
    for epoch in range(epochs):
        # Progressive unfreezing: adjust basis learning rate at specific epochs
        if config.get('spatial_learnable', False) and basis_unfreeze_epoch > 0:
            if epoch == basis_unfreeze_epoch:
                # Unfreeze basis centers
                for param_group in optimizer.param_groups:
                    if param_group.get('name') == 'basis':
                        if basis_lr_rampup_epochs > 0:
                            # Start with small lr, will ramp up gradually
                            param_group['lr'] = param_group['target_lr'] * 0.1
                            print(f"\n🔓 Epoch {epoch}: Basis UNFROZEN (starting lr={param_group['lr']:.2e})")
                        else:
                            # Jump to target lr immediately
                            param_group['lr'] = param_group['target_lr']
                            print(f"\n🔓 Epoch {epoch}: Basis UNFROZEN (lr={param_group['lr']:.2e})")
            
            elif basis_unfreeze_epoch < epoch < basis_unfreeze_epoch + basis_lr_rampup_epochs:
                # Gradually increase basis lr during rampup period
                rampup_progress = (epoch - basis_unfreeze_epoch) / basis_lr_rampup_epochs
                for param_group in optimizer.param_groups:
                    if param_group.get('name') == 'basis':
                        # Linear rampup from 10% to 100% of target lr
                        param_group['lr'] = param_group['target_lr'] * (0.1 + 0.9 * rampup_progress)
        
        # Training
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            X = batch['X'].to(device)
            coords = batch['coords'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X, coords, t)
            
            # Main loss
            if regression_type == 'mean':
                loss = criterion(y_pred, y)
            elif regression_type == 'quantile':
                loss = quantile_loss(y_pred, y, current_quantile)
            elif regression_type == 'multi-quantile':
                # y_pred: (B, Q), y: (B, 1)
                # Compute mean quantile loss across all quantiles
                losses = []
                for q_idx, q_level in enumerate(quantile_levels):
                    q_pred = y_pred[:, q_idx:q_idx+1]
                    losses.append(quantile_loss(q_pred, y, q_level))
                loss = torch.mean(torch.stack(losses))

                # Non-crossing penalty
                # If δ reparameterization is enabled, use P_nc(δ) (parameter-level)
                # Otherwise, use prediction-level penalty
                use_delta_reparam = config.get('use_delta_reparameterization', False)
                if use_delta_reparam:
                    # P_nc(δ) penalty on δ parameters (Section 3.2, Equation 3.10)
                    # NOTE: P_nc(δ) ≤ 0 always, so adding λ * P_nc(δ) to loss encourages
                    # more negative P_nc(δ) (better feasibility). The quantile loss term
                    # should prevent δ_k,0 from going to -infinity in practice.
                    # TODO: Verify sign convention with original paper [17] (Moon et al., 2021).
                    # Current implementation matches Equation 3.10 exactly, but the negative
                    # sign behavior (rewarding more negative J(δ_k)) may need empirical
                    # validation or adjustment (e.g., using -P_nc(δ) or max(0, -P_nc(δ)) instead).
                    non_crossing_lambda = config.get('non_crossing_lambda', 0.0)
                    if non_crossing_lambda > 0:
                        delta_params = model.get_delta_parameters()
                        if delta_params is not None:
                            # Use positive penalty formula if enabled
                            use_positive_penalty = config.get('use_positive_p_nc_penalty', False)
                            p_nc_delta = compute_p_nc_delta_penalty(delta_params, use_positive_penalty=use_positive_penalty)
                            loss = loss + non_crossing_lambda * p_nc_delta
                else:
                    # Prediction-level penalty (original method)
                    non_crossing_weight = config.get('non_crossing_weight', 0.0)
                    if non_crossing_weight > 0:
                        nc_power = int(config.get('non_crossing_power', 1))  # 1 or 2
                        nc_penalty = non_crossing_penalty(y_pred, reduction="mean", power=nc_power)
                        loss = loss + non_crossing_weight * nc_penalty
            
            # Add regularization penalties for learnable basis centers
            if config.get('spatial_learnable', False):
                # Domain penalty: prevent centers from going outside [0,1]^2
                domain_penalty_weight = config.get('domain_penalty_weight', 0.0)
                if domain_penalty_weight > 0:
                    domain_penalty = model.compute_domain_penalty()
                    loss = loss + domain_penalty_weight * domain_penalty
                
                # Movement penalty: prevent centers from moving too far from initialization
                movement_penalty_weight = config.get('movement_penalty_weight', 0.0)
                if movement_penalty_weight > 0:
                    movement_penalty = model.compute_movement_penalty()
                    loss = loss + movement_penalty_weight * movement_penalty
            
            # Add sparsity penalty on first layer weights
            sparsity_type = config.get('sparsity_penalty_type', 'none')
            if sparsity_type != 'none':
                lambda_l1 = config.get('sparsity_lambda_l1', 0.001)
                lambda_group = config.get('sparsity_lambda_group', 0.01)
                apply_spatial = config.get('sparsity_apply_to_spatial', True)
                apply_temporal = config.get('sparsity_apply_to_temporal', True)
                
                penalties = model.compute_sparsity_penalty(
                    penalty_type=sparsity_type,
                    lambda_l1=lambda_l1,
                    lambda_group=lambda_group
                )
                
                if apply_spatial:
                    loss = loss + penalties['spatial_penalty']
                if apply_temporal:
                    loss = loss + penalties['temporal_penalty']
            
            loss.backward()
            
            # Gradient clipping
            if config.get('grad_clip', 0) > 0:
                # Clip basis gradients more aggressively if learnable
                if config.get('spatial_learnable', False):
                    basis_params = list(model.spatial_basis.parameters())
                    mlp_params = [p for p in model.parameters() if not any(p is bp for bp in basis_params)]
                    
                    # Clip basis gradients 10x more aggressively
                    basis_clip = config['grad_clip'] * 0.1
                    torch.nn.utils.clip_grad_norm_(basis_params, basis_clip)
                    torch.nn.utils.clip_grad_norm_(mlp_params, config['grad_clip'])
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            
            # Update EMA after optimizer step (if enabled)
            if ema is not None:
                ema.update(model)
            
            # Apply warmup by manually adjusting learning rate
            if global_step < warmup_steps:
                warmup_factor = (global_step + 1) / warmup_steps
                for param_group in optimizer.param_groups:
                    param_group['lr'] = param_group['initial_lr'] * warmup_factor
            
            global_step += 1
            train_loss += loss.item()
            
            # Check for NaN
            if np.isnan(loss.item()):
                print(f"\n[WARNING] NaN detected at batch {len(train_loader)}!")
                print(f"  y_pred stats: min={y_pred.min().item():.3f}, max={y_pred.max().item():.3f}, "
                      f"mean={y_pred.mean().item():.3f}, std={y_pred.std().item():.3f}")
                print(f"  y_true stats: min={y.min().item():.3f}, max={y.max().item():.3f}")
                if config.get('spatial_learnable', False):
                    bw = model.spatial_basis.bandwidths
                    print(f"  Bandwidths: min={bw.min().item():.4f}, max={bw.max().item():.4f}, "
                          f"mean={bw.mean().item():.4f}")
                break
        
        train_loss /= len(train_loader)
        
        # Validation with EMA model (if enabled)
        # Skip validation if validation set is empty (train_ratio=1.0, matches paper)
        model.eval()
        if ema is not None:
            ema.apply_shadow()  # Use EMA parameters for validation
        
        val_loss = 0.0
        val_preds = []
        val_trues = []
        
        has_validation = len(val_loader) > 0
        
        with torch.no_grad():
            if has_validation:
                for batch in val_loader:
                    X = batch['X'].to(device)
                    coords = batch['coords'].to(device)
                    t = batch['t'].to(device)
                    y = batch['y'].to(device)
                    
                    y_pred = model(X, coords, t)
                
                    # Compute validation loss
                    if regression_type == 'mean':
                        loss = criterion(y_pred, y)
                    elif regression_type == 'quantile':
                        loss = quantile_loss(y_pred, y, current_quantile)
                    elif regression_type == 'multi-quantile':
                        # Use mean quantile loss for validation
                        losses = []
                        for q_idx, q_level in enumerate(quantile_levels):
                            q_pred = y_pred[:, q_idx:q_idx+1]
                            losses.append(quantile_loss(q_pred, y, q_level))
                        loss = torch.mean(torch.stack(losses))

                        # Keep validation objective consistent with training objective
                        # Use same penalty method as training (δ-based or prediction-level)
                        use_delta_reparam = config.get('use_delta_reparameterization', False)
                        if use_delta_reparam:
                            # TODO: Verify sign convention - see compute_p_nc_delta_penalty() docstring.
                            non_crossing_lambda = config.get('non_crossing_lambda', 0.0)
                            if non_crossing_lambda > 0:
                                delta_params = model.get_delta_parameters()
                                if delta_params is not None:
                                    use_positive_penalty = config.get('use_positive_p_nc_penalty', False)
                                    p_nc_delta = compute_p_nc_delta_penalty(
                                        delta_params,
                                        use_positive_penalty=use_positive_penalty
                                    )
                                    loss = loss + non_crossing_lambda * p_nc_delta
                        else:
                            non_crossing_weight = config.get('non_crossing_weight', 0.0)
                            if non_crossing_weight > 0:
                                nc_power = int(config.get('non_crossing_power', 1))  # 1 or 2
                                nc_penalty = non_crossing_penalty(y_pred, reduction="mean", power=nc_power)
                                loss = loss + non_crossing_weight * nc_penalty
                    
                    val_loss += loss.item()
                    
                    val_preds.append(y_pred.cpu().numpy())
                    val_trues.append(y.cpu().numpy())
            else:
                # No validation set (train_ratio=1.0, matches paper)
                # Use train loss as proxy for validation
                val_loss = train_loss
                val_preds = []
                val_trues = []
        
        if ema is not None:
            ema.restore()  # Restore original parameters for training
        
        if has_validation:
            val_loss /= len(val_loader)
        else:
            val_loss = train_loss  # Use train loss when no validation set
        
        # Compute RMSE
        if has_validation and len(val_preds) > 0:
            val_preds = np.concatenate(val_preds, axis=0)
            val_trues = np.concatenate(val_trues, axis=0)
            
            # For multi-quantile, use median quantile (typically 0.5) for RMSE
            if regression_type == 'multi-quantile':
                # Find median quantile index
                median_idx = len(quantile_levels) // 2
                val_preds_for_rmse = val_preds[:, median_idx:median_idx+1]
            else:
                val_preds_for_rmse = val_preds
            
            val_rmse = np.sqrt(np.mean((val_preds_for_rmse - val_trues) ** 2))
        else:
            # No validation set, use train RMSE as proxy
            val_rmse = 0.0  # Will be computed from train if needed
        
        # Compute check_loss for early stopping (if enabled)
        # This is the metric that CRPS is based on, without penalty
        use_check_loss_early_stop = config.get('use_check_loss_early_stop', False)
        if has_validation and len(val_preds) > 0:
            if use_check_loss_early_stop and regression_type == 'multi-quantile':
                # Compute mean check loss across all quantiles (same as CRPS computation)
                check_losses = []
                for q_idx, q_level in enumerate(quantile_levels):
                    q_pred = val_preds[:, q_idx:q_idx+1].flatten()
                    q_true = val_trues.flatten()
                    check_loss_q = check_loss_numpy(q_pred, q_true, q_level)
                    check_losses.append(check_loss_q)
                val_check_loss = np.mean(check_losses)
            elif use_check_loss_early_stop and regression_type == 'quantile':
                current_quantile = config.get('current_quantile', 0.5)
                val_check_loss = check_loss_numpy(val_preds.flatten(), val_trues.flatten(), current_quantile)
            else:
                val_check_loss = None
        else:
            # No validation set, cannot compute check_loss for early stopping
            val_check_loss = None
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        if val_check_loss is not None:
            if 'val_check_loss' not in history:
                history['val_check_loss'] = []
            history['val_check_loss'].append(val_check_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        history['lr'].append(current_lr)
        
        # Compact output: Epoch | Train Loss | Val Loss | Val RMSE | Status
        if use_check_loss_early_stop and val_check_loss is not None:
            output_str = f"Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, ValCheck={val_check_loss:.6f}, RMSE={val_rmse:.6f}"
        else:
            output_str = f"Epoch {epoch+1}/{epochs}: Train={train_loss:.6f}, Val={val_loss:.6f}, RMSE={val_rmse:.6f}"
        
        # Learning rate scheduling (only after warmup)
        if scheduler is not None and epoch >= warmup_epochs:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
            output_str += f", LR={current_lr:.6f}"
        elif epoch < warmup_epochs:
            # During warmup, show current LR
            output_str += f", LR={current_lr:.6f}(warmup)"
        
        # Save best model (EMA version)
        # Skip early stopping if no validation set (train_ratio=1.0, matches paper)
        if not has_validation:
            # No validation set: save model every epoch (or use final model)
            # For paper setting, we'll use final model (no early stopping)
            patience_counter = 0  # Reset counter, no early stopping
            output_str += " (No validation, using final model)"
        elif use_check_loss_early_stop and val_check_loss is not None:
            # Initialize best_val_check_loss if first epoch
            if epoch == 0:
                best_val_check_loss = float('inf')
            # Use check_loss for best model selection
            if not np.isnan(val_check_loss) and val_check_loss < best_val_check_loss:
                best_val_check_loss = val_check_loss
                best_val_loss = val_loss  # Keep track for logging
                patience_counter = 0
                # Save model (EMA if enabled, otherwise current)
                if ema is not None:
                    ema.apply_shadow()
                torch.save(model.state_dict(), best_model_path)
                if ema is not None:
                    ema.restore()
                output_str += " [Best]"
            else:
                patience_counter += 1
                output_str += f" ({patience_counter}/{patience})"
        else:
            # Original logic: use val_loss
            if not np.isnan(val_loss) and val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save model (EMA if enabled, otherwise current)
                if ema is not None:
                    ema.apply_shadow()
                torch.save(model.state_dict(), best_model_path)
                if ema is not None:
                    ema.restore()
                output_str += " [Best]"
            else:
                patience_counter += 1
                output_str += f" ({patience_counter}/{patience})"
        
        try:
            print(output_str)
        except (ValueError, OSError):
            pass  # stdout closed in parallel mode
        
        # Record basis centers every 100 epochs (if learnable)
        if config.get('spatial_learnable', False) and (epoch + 1) % center_record_interval == 0:
            with torch.no_grad():
                centers = model.spatial_basis.centers.cpu().numpy().copy()
                basis_centers_history.append((epoch + 1, centers))
        
        # Skip early stopping if no validation set
        if has_validation and patience_counter >= patience:
            try:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
            except (ValueError, OSError):
                pass
            break
    
    # Load best model (EMA version) if it exists
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        if use_check_loss_early_stop and 'best_val_check_loss' in locals():
            print(f"\nTraining Complete! Best Val Check Loss: {best_val_check_loss:.6f}, Best Val Loss: {best_val_loss:.6f} (EMA model)")
        else:
            print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.6f} (EMA model)")
    else:
        # If no best model, use final model (EMA if enabled)
        if ema is not None:
            ema.apply_shadow()
            print(f"\n[WARNING] No best model saved, using final EMA model")
        else:
            print(f"\n[WARNING] No best model saved, using final model")
    
    # Save training history
    import pandas as pd
    history_df = pd.DataFrame({
        'epoch': list(range(1, len(history['train_loss']) + 1)),
        'train_loss': history['train_loss'],
        'val_loss': history['val_loss'],
        'val_rmse': history['val_rmse'],
        'lr': history['lr']
    })
    history_df.to_csv(output_dir / 'training_history.csv', index=False)
    print(f"Training history saved to: {output_dir / 'training_history.csv'}")
    
    # Return history and basis centers trajectory
    return model, history, basis_centers_history


def evaluate_model(model, data_loader, device, config=None):
    """
    Evaluate model on a dataset
    
    Returns:
        metrics: dict with 'mse', 'mae', 'rmse', and optionally 'check_loss', 'crps' for quantile/multi-quantile
    """
    model.eval()
    all_preds = []
    all_trues = []
    
    # Handle empty dataset (e.g., when train_ratio=1.0, no validation set)
    if len(data_loader) == 0 or len(data_loader.dataset) == 0:
        # Return dummy metrics for empty dataset
        regression_type = config.get('regression_type', 'mean') if config is not None else 'mean'
        if regression_type == 'multi-quantile':
            quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
            n_quantiles = len(quantile_levels)
            return {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'crps': 0.0,
                'mean_check_loss': 0.0,
                'check_loss': 0.0,
                'coverage_90': 0.0
            }
        else:
            return {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'check_loss': 0.0
            }
    
    with torch.no_grad():
        for batch in data_loader:
            X = batch['X'].to(device)
            coords = batch['coords'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            y_pred = model(X, coords, t)
            
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y.cpu().numpy())
    
    # Handle case where no batches were processed (empty dataset)
    if len(all_preds) == 0:
        regression_type = config.get('regression_type', 'mean') if config is not None else 'mean'
        if regression_type == 'multi-quantile':
            return {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'crps': 0.0,
                'mean_check_loss': 0.0,
                'check_loss': 0.0,
                'coverage_90': 0.0
            }
        else:
            return {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'check_loss': 0.0
            }
    
    preds = np.concatenate(all_preds, axis=0)  # (N, 1) or (N, Q)
    trues = np.concatenate(all_trues, axis=0)  # (N, 1)
    
    regression_type = config.get('regression_type', 'mean') if config is not None else 'mean'
    
    # For multi-quantile, use median quantile for MSE/MAE/RMSE
    if regression_type == 'multi-quantile':
        quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
        median_idx = len(quantile_levels) // 2
        preds_for_metrics = preds[:, median_idx:median_idx+1]
    else:
        preds_for_metrics = preds
    
    mse = np.mean((preds_for_metrics - trues) ** 2)
    mae = np.mean(np.abs(preds_for_metrics - trues))
    
    metrics = {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse))
    }
    
    # Add check loss for quantile regression
    if config is not None and config.get('regression_type') == 'quantile' and 'current_quantile' in config:
        q = config['current_quantile']
        check_loss = quantile_loss(
            torch.tensor(preds, dtype=torch.float32),
            torch.tensor(trues, dtype=torch.float32),
            q
        ).item()
        metrics['check_loss'] = float(check_loss)
    
    # Add CRPS and mean check loss for multi-quantile regression
    if config is not None and config.get('regression_type') == 'multi-quantile':
        quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
        
        # Compute CRPS using the new multi-quantile function
        crps = compute_crps_multi_quantile(preds, trues, quantile_levels)
        metrics['crps'] = float(crps)
        
        # Compute mean check loss across all quantiles
        check_losses = []
        for q_idx, q_level in enumerate(quantile_levels):
            q_pred = preds[:, q_idx:q_idx+1]
            check_loss_q = quantile_loss(
                torch.tensor(q_pred, dtype=torch.float32),
                torch.tensor(trues, dtype=torch.float32),
                q_level
            ).item()
            check_losses.append(check_loss_q)
        
        metrics['mean_check_loss'] = float(np.mean(check_losses))
        metrics['check_loss'] = float(np.mean(check_losses))  # Alias for compatibility

        # Empirical coverage of 90% prediction interval [q_0.05, q_0.95]
        coverage_90 = compute_coverage(preds, trues, quantile_levels, alpha=0.1)
        metrics['coverage_90'] = float(coverage_90)
    
    return metrics


def save_results(results, output_dir):
    """Save results to JSON file"""
    # Convert numpy types to native Python types
    def convert_to_json_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_json_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_json_serializable(results)
    
    result_file = output_dir / 'results.json'
    with open(result_file, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    print(f"\nResults saved to: {result_file}")


def plot_training_curves(history, save_path):
    """Plot training curves"""
    matplotlib.use('Agg')
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('MSE Loss', fontsize=16)
    ax.set_title('Training and Validation Loss', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits based on epochs 2+ (ignore epoch 1)
    if len(history['train_loss']) > 1:
        train_loss_from_2 = history['train_loss'][1:]  # Skip first epoch
        val_loss_from_2 = history['val_loss'][1:]
        
        y_min = min(min(train_loss_from_2), min(val_loss_from_2))
        y_max = max(max(train_loss_from_2), max(val_loss_from_2))
        
        # Add 10% margin
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
    
    # RMSE
    ax = axes[1]
    ax.plot(epochs, history['val_rmse'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('RMSE', fontsize=16)
    ax.set_title('Validation RMSE', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    
    # Set y-axis limits based on epochs 2+ (ignore epoch 1)
    if len(history['val_rmse']) > 1:
        rmse_from_2 = history['val_rmse'][1:]  # Skip first epoch
        
        y_min = min(rmse_from_2)
        y_max = max(rmse_from_2)
        
        # Add 10% margin
        margin = (y_max - y_min) * 0.1
        ax.set_ylim(y_min - margin, y_max + margin)
    
    # Learning Rate
    ax = axes[2]
    ax.plot(epochs, history['lr'], 'purple', linewidth=2)
    ax.set_xlabel('Epoch', fontsize=16)
    ax.set_ylabel('Learning Rate', fontsize=16)
    ax.set_title('Learning Rate Schedule', fontsize=18, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')  # Log scale to see changes better
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training curves saved to {save_path}")


def plot_predictions(model, z_full, coords, train_mask, device, output_dir, n_times=3):
    """
    Plot true/pred/bias heatmaps for random time points
    
    Args:
        model: trained model
        z_full: (T, S) full data
        coords: (S, 2) coordinates
        train_mask: (T, S) training mask
        device: torch device
        output_dir: output directory
        n_times: number of time points to plot
    """
    matplotlib.use('Agg')
    
    T, S = z_full.shape
    
    # Select random time points
    np.random.seed(42)
    time_indices = np.random.choice(T, size=min(n_times, T), replace=False)
    time_indices = sorted(time_indices)
    
    # Get spatial basis centers from model
    spatial_centers = model.spatial_basis.centers.detach().cpu().numpy()  # (k_spatial, 2)
    spatial_bandwidths = model.spatial_basis.bandwidths.detach().cpu().numpy()  # (k_spatial,)
    
    # Size basis markers proportional to bandwidth
    # Normalize to reasonable marker size range [10, 100]
    bw_normalized = (spatial_bandwidths - spatial_bandwidths.min()) / (spatial_bandwidths.max() - spatial_bandwidths.min() + 1e-8)
    basis_sizes = 10 + bw_normalized * 90  # Range [10, 100]
    
    # Generate predictions for all sites at selected times
    model.eval()
    predictions = {}
    
    with torch.no_grad():
        for t_idx in time_indices:
            t_normalized = t_idx / (T - 1) if T > 1 else 0.0
            t_tensor = torch.tensor([[t_normalized]], dtype=torch.float32).repeat(S, 1).to(device)
            coords_tensor = torch.from_numpy(coords).float().to(device)
            X_tensor = torch.zeros(S, 0).to(device)  # No covariates
            
            y_pred = model(X_tensor, coords_tensor, t_tensor).cpu().numpy()  # (S, output_dim)
            
            # For multi-quantile, use median quantile; otherwise use single output
            if y_pred.shape[1] > 1:  # Multi-quantile
                median_idx = y_pred.shape[1] // 2
                y_pred = y_pred[:, median_idx]
            else:
                y_pred = y_pred.flatten()
            
            predictions[t_idx] = y_pred
    
    # Create grid for interpolation (higher resolution for smoother heatmap)
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Create plots
    fig, axes = plt.subplots(n_times, 3, figsize=(20, 5 * n_times))
    if n_times == 1:
        axes = axes.reshape(1, -1)
    
    for i, t_idx in enumerate(time_indices):
        y_true = z_full[t_idx, :]
        y_pred = predictions[t_idx]
        bias = y_pred - y_true
        
        # Get train sites at this time
        train_sites_t = np.where(train_mask[t_idx, :])[0]
        train_coords_t = coords[train_sites_t]
        
        # Valid indices (non-NaN)
        valid_idx = ~np.isnan(y_true)
        coords_valid = coords[valid_idx]
        
        # Interpolate to grid using nearest neighbor (to fill space like Voronoi)
        y_true_grid = griddata(coords_valid, y_true[valid_idx], (xi_grid, yi_grid), method='nearest')
        y_pred_grid = griddata(coords_valid, y_pred[valid_idx], (xi_grid, yi_grid), method='nearest')
        bias_grid = griddata(coords_valid, bias[valid_idx], (xi_grid, yi_grid), method='nearest')
        
        # True values
        ax = axes[i, 0]
        im = ax.pcolormesh(xi_grid, yi_grid, y_true_grid, cmap='viridis', shading='auto')
        ax.scatter(train_coords_t[:, 0], train_coords_t[:, 1], 
                  c='black', s=20, alpha=0.6, label='Train sites', edgecolors='white', linewidths=0.5)
        ax.scatter(spatial_centers[:, 0], spatial_centers[:, 1], 
                  c='red', s=basis_sizes, marker='x', alpha=0.5, label='Basis centers', linewidths=1.5)
        ax.set_title(f't={t_idx+1} - True', fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=11)
        plt.colorbar(im, ax=ax)
        
        # Predicted values
        ax = axes[i, 1]
        im = ax.pcolormesh(xi_grid, yi_grid, y_pred_grid, cmap='viridis', shading='auto')
        ax.scatter(train_coords_t[:, 0], train_coords_t[:, 1], 
                  c='black', s=20, alpha=0.6, label='Train sites', edgecolors='white', linewidths=0.5)
        ax.scatter(spatial_centers[:, 0], spatial_centers[:, 1], 
                  c='red', s=basis_sizes, marker='x', alpha=0.5, label='Basis centers', linewidths=1.5)
        ax.set_title(f't={t_idx+1} - Predicted', fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=11)
        plt.colorbar(im, ax=ax)
        
        # Bias (pred - true)
        ax = axes[i, 2]
        bias_max = np.nanmax(np.abs(bias[valid_idx]))
        im = ax.pcolormesh(xi_grid, yi_grid, bias_grid, cmap='RdBu_r', shading='auto',
                          vmin=-bias_max, vmax=bias_max)
        ax.scatter(train_coords_t[:, 0], train_coords_t[:, 1], 
                  c='black', s=20, alpha=0.6, label='Train sites', edgecolors='white', linewidths=0.5)
        ax.scatter(spatial_centers[:, 0], spatial_centers[:, 1], 
                  c='red', s=basis_sizes, marker='x', alpha=0.5, label='Basis centers', linewidths=1.5)
        ax.set_title(f't={t_idx+1} - Bias (Pred - True)', fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=11)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    save_path = output_dir / 'prediction_maps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction maps saved to {save_path}")



def get_spatial_quantile_predictions(model, z_full, coords, device):
    """Get full (T, S, Q) quantile predictions for all grid points. Returns None if not multi-quantile."""
    T, S = z_full.shape
    model.eval()
    all_preds = []
    with torch.no_grad():
        for t_idx in range(T):
            t_normalized = t_idx / (T - 1) if T > 1 else 0.0
            t_tensor = torch.tensor([[t_normalized]], dtype=torch.float32).repeat(S, 1).to(device)
            coords_tensor = torch.from_numpy(coords).float().to(device)
            X_tensor = torch.zeros(S, 0).to(device)
            y_pred = model(X_tensor, coords_tensor, t_tensor).cpu().numpy()  # (S, Q)
            all_preds.append(y_pred)
    preds = np.stack(all_preds, axis=0)  # (T, S, Q)
    if preds.shape[2] <= 1:
        return None
    return preds


def plot_spatial_mse(model, z_full, coords, train_mask, device, output_dir, 
                     return_predictions=False, valid_mask=None, test_mask=None):
    """
    Plot spatial MSE heatmap averaged over all time points
    
    Args:
        model: trained model
        z_full: (T, S) full data
        coords: (S, 2) coordinates
        train_mask: (T, S) training mask
        device: torch device
        output_dir: output directory
        return_predictions: if True, return true and pred values
        valid_mask: (T, S) validation mask (optional, for saving)
        test_mask: (T, S) test mask (optional, for saving)
    
    Returns:
        If return_predictions=True: (all_predictions, z_full, coords, train_mask, valid_mask, test_mask)
        Otherwise: None
    """
    matplotlib.use('Agg')
    
    T, S = z_full.shape
    
    # Get spatial basis centers from model
    spatial_centers = model.spatial_basis.centers.detach().cpu().numpy()  # (k_spatial, 2)
    spatial_bandwidths = model.spatial_basis.bandwidths.detach().cpu().numpy()  # (k_spatial,)
    
    # Size basis markers proportional to bandwidth
    bw_normalized = (spatial_bandwidths - spatial_bandwidths.min()) / (spatial_bandwidths.max() - spatial_bandwidths.min() + 1e-8)
    basis_sizes = 10 + bw_normalized * 90  # Range [10, 100]
    
    # Generate predictions for all sites at all times
    model.eval()
    all_predictions = np.zeros((T, S))
    
    with torch.no_grad():
        for t_idx in range(T):
            t_normalized = t_idx / (T - 1) if T > 1 else 0.0
            t_tensor = torch.tensor([[t_normalized]], dtype=torch.float32).repeat(S, 1).to(device)
            coords_tensor = torch.from_numpy(coords).float().to(device)
            X_tensor = torch.zeros(S, 0).to(device)  # No covariates
            
            y_pred = model(X_tensor, coords_tensor, t_tensor).cpu().numpy()  # (S, output_dim)
            
            # For multi-quantile, use median quantile; otherwise use single output
            if y_pred.shape[1] > 1:  # Multi-quantile
                median_idx = y_pred.shape[1] // 2
                y_pred = y_pred[:, median_idx]
            else:
                y_pred = y_pred.flatten()
            
            all_predictions[t_idx, :] = y_pred
    
    # Compute MSE per site (averaged over time)
    squared_errors = (all_predictions - z_full) ** 2
    site_mse = np.nanmean(squared_errors, axis=0)  # (S,)
    
    # Get all train sites (any time)
    train_sites_any = np.where(train_mask.any(axis=0))[0]
    train_coords_any = coords[train_sites_any]
    
    # Valid sites (not all NaN)
    valid_sites = ~np.isnan(site_mse)
    coords_valid = coords[valid_sites]
    site_mse_valid = site_mse[valid_sites]
    
    # Create grid for interpolation
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate MSE to grid using nearest neighbor
    mse_grid = griddata(coords_valid, site_mse_valid, (xi_grid, yi_grid), method='nearest')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(xi_grid, yi_grid, mse_grid, cmap='YlOrRd', shading='auto')
    ax.scatter(train_coords_any[:, 0], train_coords_any[:, 1], 
              c='black', s=25, alpha=0.6, label='Train sites', edgecolors='white', linewidths=0.5)
    ax.scatter(spatial_centers[:, 0], spatial_centers[:, 1], 
              c='red', s=basis_sizes, marker='x', alpha=0.5, label='Basis centers', linewidths=1.5)
    
    ax.set_title('Spatial MSE (Averaged over Time)', fontsize=18, fontweight='bold')
    ax.set_xlabel('x', fontsize=16)
    ax.set_ylabel('y', fontsize=16)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=13)
    cbar = plt.colorbar(im, ax=ax, label='MSE')
    cbar.ax.tick_params(labelsize=12)
    cbar.set_label('MSE', fontsize=14)
    
    plt.tight_layout()
    save_path = output_dir / 'spatial_mse.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spatial MSE plot saved to {save_path}")
    
    if return_predictions:
        return all_predictions, z_full, coords, train_mask, valid_mask, test_mask
    return None


def plot_spatial_coverage_and_qhat(output_dir):
    """
    Plot spatial coverage (nominal vs cluster-aware) and qhat spatial distribution.
    Reads quantile_levels and conformal_alpha from conformal_info.npz (source of truth).
    Requires: predictions.npz with predictions_quantile, conformal_info.npz.
    """
    output_dir = Path(output_dir)
    pred_path = output_dir / 'predictions.npz'
    conformal_path = output_dir / 'conformal_info.npz'
    if not pred_path.exists() or not conformal_path.exists():
        return
    preds_npz = np.load(pred_path, allow_pickle=True)
    conf_npz = np.load(conformal_path, allow_pickle=True)
    if 'predictions_quantile' not in preds_npz:
        return
    preds_q = preds_npz['predictions_quantile']  # (T, S, Q)
    z_full = preds_npz['true']
    coords = preds_npz['coords']
    test_mask = preds_npz['test_mask']
    qhat_per_center = conf_npz['qhat_per_center']  # (C,)
    spatial_centers = conf_npz['spatial_centers']  # (C, 2)
    quantile_levels = np.asarray(conf_npz['quantile_levels'])
    conformal_alpha = float(conf_npz['conformal_alpha']) if 'conformal_alpha' in conf_npz.files else 0.1
    T, S, Q = preds_q.shape
    q_lo, q_hi = conformal_alpha / 2, 1.0 - conformal_alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    q_lo_grid = preds_q[:, :, idx_lo]
    q_hi_grid = preds_q[:, :, idx_hi]
    cluster_ids = _assign_nearest_center(coords, spatial_centers)  # (S,)
    qhat_per_site = qhat_per_center[cluster_ids]  # (S,)

    test_mask = test_mask.astype(bool)
    n_test_per_site = test_mask.sum(axis=0)  # (S,)
    inside_nominal = (z_full >= q_lo_grid) & (z_full <= q_hi_grid)
    qhat_expanded = np.broadcast_to(qhat_per_site, (T, S))
    inside_cluster = (z_full >= q_lo_grid - qhat_expanded) & (z_full <= q_hi_grid + qhat_expanded)
    cov_nominal = np.where(n_test_per_site > 0,
                           np.where(test_mask, inside_nominal, 0).sum(axis=0) / (n_test_per_site + 1e-10),
                           np.nan)
    cov_cluster = np.where(n_test_per_site > 0,
                           np.where(test_mask, inside_cluster, 0).sum(axis=0) / (n_test_per_site + 1e-10),
                           np.nan)
    valid_sites = n_test_per_site > 0
    coords_valid = coords[valid_sites]
    cov_nominal_valid = cov_nominal[valid_sites]
    cov_cluster_valid = cov_cluster[valid_sites]
    qhat_valid = qhat_per_site[valid_sites]
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    cov_nominal_grid = griddata(coords_valid, cov_nominal_valid, (xi_grid, yi_grid), method='nearest')
    cov_cluster_grid = griddata(coords_valid, cov_cluster_valid, (xi_grid, yi_grid), method='nearest')
    qhat_grid = griddata(coords, qhat_per_site, (xi_grid, yi_grid), method='nearest')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title, vmin, vmax in [
        (axes[0], cov_nominal_grid, 'Spatial Coverage (Nominal 90% PI)', 0.5, 1.0),
        (axes[1], cov_cluster_grid, 'Spatial Coverage (Cluster-aware)', 0.5, 1.0),
        (axes[2], qhat_grid, 'Conformal qhat (per cluster)', 0, None),
    ]:
        vmax = vmax if vmax is not None else np.nanmax(data) * 1.05
        im = ax.pcolormesh(xi_grid, yi_grid, data, cmap='RdYlGn' if 'Coverage' in title else 'viridis',
                           shading='auto', vmin=vmin, vmax=vmax)
        ax.scatter(spatial_centers[:, 0], spatial_centers[:, 1], c='red', s=15, marker='x', alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path = output_dir / 'spatial_coverage_and_qhat.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spatial coverage/qhat plot saved to {save_path}")


def plot_temporal_series(model, z_full, coords, train_mask, device, output_dir, 
                         valid_mask=None, test_mask=None, n_sites=4, quantile_models=None, quantile_levels=None,
                         conformal_qhat=None, conformal_alpha=0.1):
    """
    Plot temporal series for selected spatial locations
    
    Args:
        model: trained model (for mean regression or single quantile), can be None if only quantile_models provided
        z_full: (T, S) full data
        coords: (S, 2) coordinates
        train_mask: (T, S) training mask
        device: computation device
        output_dir: output directory
        valid_mask: (T, S) validation mask (optional)
        test_mask: (T, S) test mask (optional)
        n_sites: number of sites to plot
        quantile_models: dict of {quantile_level: model} for quantile regression (optional)
        quantile_levels: list of quantile levels (optional)
        conformal_qhat: CQR expansion (optional); if set, plot conformal 90% PI band [q_lo-qhat, q_hi+qhat]
        conformal_alpha: nominal miscoverage for conformal (0.1 -> 90% PI)
    """
    matplotlib.use('Agg')
    
    T, S = z_full.shape
    
    if valid_mask is None:
        valid_mask = np.zeros_like(train_mask, dtype=bool)
    if test_mask is None:
        test_mask = ~(train_mask | valid_mask)
    
    # Select sites with good spatial coverage
    # Choose sites from different regions
    coords_np = coords.cpu().numpy() if torch.is_tensor(coords) else coords
    
    # First, ensure at least one site with training samples
    sites_with_train = np.where(train_mask.sum(axis=0) > 0)[0]
    selected_sites = []
    
    if len(sites_with_train) > 0:
        # Pick a train site near the center
        center = np.array([0.5, 0.5])
        dists_to_center = np.linalg.norm(coords_np[sites_with_train] - center, axis=1)
        train_site = sites_with_train[np.argmin(dists_to_center)]
        selected_sites.append(train_site)
    
    # Then select remaining sites with spatial coverage
    n_grid = int(np.ceil(np.sqrt(n_sites)))
    
    for i in range(n_grid):
        for j in range(n_grid):
            if len(selected_sites) >= n_sites:
                break
            
            # Define region boundaries
            x_min, x_max = i / n_grid, (i + 1) / n_grid
            y_min, y_max = j / n_grid, (j + 1) / n_grid
            
            # Find sites in this region
            in_region = (
                (coords_np[:, 0] >= x_min) & (coords_np[:, 0] < x_max) &
                (coords_np[:, 1] >= y_min) & (coords_np[:, 1] < y_max)
            )
            
            if in_region.sum() > 0:
                # Pick site closest to region center
                region_center = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2])
                dists = np.linalg.norm(coords_np[in_region] - region_center, axis=1)
                local_idx = np.argmin(dists)
                global_idx = np.where(in_region)[0][local_idx]
                if global_idx not in selected_sites:
                    selected_sites.append(global_idx)
    
    # Get predictions
    all_predictions = None
    quantile_predictions = {}  # Initialize for multi-quantile or separate quantile models
    
    if model is not None:
        model.eval()
        with torch.no_grad():
            # Create meshgrid for all (t, s) pairs
            t_grid = torch.linspace(0, 1, T, device=device)
            coords_tensor = torch.tensor(coords_np, dtype=torch.float32, device=device)
            
            # Expand to (T*S, 1) and (T*S, 2)
            t_expanded = t_grid.repeat_interleave(S).unsqueeze(1)  # (T*S, 1)
            coords_expanded = coords_tensor.repeat(T, 1)  # (T*S, 2)
            
            # Create empty covariate tensor if p_covariates == 0
            if hasattr(model, 'p') and model.p > 0:
                X_expanded = torch.zeros(T * S, model.p, device=device)
            else:
                X_expanded = torch.zeros(T * S, 0, device=device)
            
            # Get predictions for mean or single quantile or multi-quantile
            all_predictions_raw = model(X_expanded, coords_expanded, t_expanded)  # (T*S, output_dim)
            
            # For multi-quantile model (output_dim > 1), also extract individual quantile predictions
            if all_predictions_raw.shape[1] > 1 and quantile_levels is not None:
                # Multi-quantile model: extract each quantile
                for q_idx, q_level in enumerate(quantile_levels):
                    q_pred_flat = all_predictions_raw[:, q_idx]
                    quantile_predictions[q_level] = q_pred_flat.reshape(T, S).cpu().numpy()
                
                # Use median quantile for basic plot
                median_idx = all_predictions_raw.shape[1] // 2
                all_predictions_flat = all_predictions_raw[:, median_idx]
            else:
                all_predictions_flat = all_predictions_raw.squeeze()
            
            all_predictions = all_predictions_flat.reshape(T, S).cpu().numpy()  # (T, S)
    
    # Get quantile predictions from separate models if available
    if quantile_models is not None and quantile_levels is not None:
        # Create meshgrid for quantile predictions
        t_grid = torch.linspace(0, 1, T, device=device)
        coords_tensor = torch.tensor(coords_np, dtype=torch.float32, device=device)
        t_expanded = t_grid.repeat_interleave(S).unsqueeze(1)
        coords_expanded = coords_tensor.repeat(T, 1)
        
        for q_level, q_model in quantile_models.items():
            q_model.eval()
            with torch.no_grad():
                # Create empty covariate tensor for each quantile model
                if hasattr(q_model, 'p_covariates') and q_model.p_covariates > 0:
                    X_q = torch.zeros(T * S, q_model.p_covariates, device=device)
                else:
                    X_q = torch.zeros(T * S, 0, device=device)
                q_predictions_flat = q_model(X_q, coords_expanded, t_expanded).squeeze()  # (T*S,)
                quantile_predictions[q_level] = q_predictions_flat.reshape(T, S).cpu().numpy()  # (T, S)
    
    # Only create basic plot if model is provided
    if model is not None and all_predictions is not None:
        # Create subplots: 4 rows x 1 column for wide horizontal plots
        n_rows = len(selected_sites)
        n_cols = 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        if len(selected_sites) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        time_points = np.arange(1, T + 1)  # Original time scale: 1, 2, ..., T
        
        for idx, site_idx in enumerate(selected_sites):
            ax = axes[idx]
            
            # Get data for this site
            true_values = z_full[:, site_idx]
            pred_values = all_predictions[:, site_idx]
            
            train_obs = train_mask[:, site_idx]
            valid_obs = valid_mask[:, site_idx]
            test_obs = test_mask[:, site_idx]
            
            # Plot predictions (line)
            ax.plot(time_points, pred_values, 'b-', linewidth=2, label='Prediction', alpha=0.8)
            
            # Plot observed data (all circles) - Test: gray, Train+Valid: black
            if test_obs.sum() > 0:
                ax.scatter(time_points[test_obs], true_values[test_obs], 
                          c='gray', s=40, marker='o', alpha=0.7, label='Test (unobserved)', zorder=3)
            # Combine train and valid as "observed" in black
            observed_mask = train_obs | valid_obs
            if observed_mask.sum() > 0:
                ax.scatter(time_points[observed_mask], true_values[observed_mask], 
                          c='black', s=40, marker='o', alpha=0.7, label='Train (observed)', zorder=3)
            
            site_coord = coords_np[site_idx]
            ax.set_title(f'Site {site_idx} at ({site_coord[0]:.3f}, {site_coord[1]:.3f})', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(selected_sites), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend on the right
        save_path = output_dir / 'temporal_series.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Temporal series plot saved to {save_path}")
    
    # If quantile regression, create combined quantile plot ONLY
    if quantile_predictions and quantile_levels:
        # Create subplots: 4 rows x 1 column for wide horizontal plots
        n_rows = len(selected_sites)
        n_cols = 1
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, 3.5 * n_rows))
        if len(selected_sites) == 1:
            axes = np.array([axes])
        axes = axes.flatten()
        
        # Use vivid rainbow colors for quantiles (more saturated and visible)
        if len(quantile_levels) == 3:
            # For 3 quantiles: blue, green, red
            colors = ['#0000FF', '#00CC00', '#FF0000']  # Vivid blue, green, red
        elif len(quantile_levels) == 5:
            # For 5 quantiles: blue, cyan, green, orange, red
            colors = ['#0000FF', '#00CCCC', '#00CC00', '#FF8800', '#FF0000']
        elif len(quantile_levels) == 7:
            # For 7 quantiles: full rainbow
            colors = ['#8B00FF', '#0000FF', '#00CCCC', '#00CC00', '#FFCC00', '#FF8800', '#FF0000']
        else:
            # General case: use tab10 colormap (more distinct colors)
            colors = plt.cm.tab10(np.linspace(0, 0.9, len(quantile_levels)))
        
        time_points = np.arange(1, T + 1)  # Original time scale: 1, 2, ..., T
        
        for idx, site_idx in enumerate(selected_sites):
            ax = axes[idx]
            
            true_values = z_full[:, site_idx]
            train_obs = train_mask[:, site_idx]
            valid_obs = valid_mask[:, site_idx]
            test_obs = test_mask[:, site_idx]
            
            # Plot each quantile with rainbow colors
            for q_idx, q_level in enumerate(quantile_levels):
                pred_values = quantile_predictions[q_level][:, site_idx]
                ax.plot(time_points, pred_values, 
                       color=colors[q_idx], linewidth=2, 
                       label=f'τ={q_level}', alpha=0.8)
            
            # Conformal 90% PI: draw only the *expansion margin* (qhat) so it's clearly distinct from quantile band
            if conformal_qhat is not None and conformal_qhat > 0:
                q_lo = conformal_alpha / 2
                q_hi = 1.0 - conformal_alpha / 2
                q_levels_arr = np.asarray(quantile_levels)
                idx_lo = np.argmin(np.abs(q_levels_arr - q_lo))
                idx_hi = np.argmin(np.abs(q_levels_arr - q_hi))
                q_lo_level = quantile_levels[idx_lo]
                q_hi_level = quantile_levels[idx_hi]
                q_lo_line = quantile_predictions[q_lo_level][:, site_idx]
                q_hi_line = quantile_predictions[q_hi_level][:, site_idx]
                conf_low = q_lo_line - conformal_qhat
                conf_high = q_hi_line + conformal_qhat
                # Fill only the conformal *margin* (below q_lo and above q_hi) so it's obvious conformal = quantile + expansion
                ax.fill_between(time_points, conf_low, q_lo_line, alpha=0.4, color='purple', 
                                label='90% PI (conformal margin)' if idx == 0 else None, zorder=1)
                ax.fill_between(time_points, q_hi_line, conf_high, alpha=0.4, color='purple', zorder=1)
                ax.plot(time_points, conf_low, '--', color='purple', linewidth=2, alpha=0.9, 
                        label='conformal bound' if idx == 0 else None, zorder=2)
                ax.plot(time_points, conf_high, '--', color='purple', linewidth=2, alpha=0.9, zorder=2)
            
            # Plot observed data (all circles) - Test: gray, Train+Valid: black
            if test_obs.sum() > 0:
                ax.scatter(time_points[test_obs], true_values[test_obs], 
                          c='gray', s=40, marker='o', alpha=0.7, 
                          label='Test', zorder=3)
            # Combine train and valid as "observed" in black
            observed_mask = train_obs | valid_obs
            if observed_mask.sum() > 0:
                ax.scatter(time_points[observed_mask], true_values[observed_mask], 
                          c='black', s=40, marker='o', alpha=0.7, 
                          label='Train', zorder=3)
            
            site_coord = coords_np[site_idx]
            ax.set_title(f'Site {site_idx} at ({site_coord[0]:.3f}, {site_coord[1]:.3f}) - All Quantiles', 
                        fontsize=12, fontweight='bold')
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Value', fontsize=10)
            ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
            ax.grid(True, alpha=0.3)
        
        # Hide unused subplots
        for idx in range(len(selected_sites), len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout(rect=[0, 0, 0.85, 1])  # Leave space for legend on the right
        save_path = output_dir / 'temporal_series_quantiles_combined.png'
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Combined quantile temporal series plot saved to {save_path}")


def plot_observation_pattern(coords, obs_mask, train_mask, valid_mask, output_dir):
    """
    Plot the spatial pattern of observations (train/valid/test)
    
    Args:
        coords: (S, 2) coordinates
        obs_mask: (T, S) observation mask
        train_mask: (T, S) training mask
        valid_mask: (T, S) validation mask
        output_dir: output directory
    """
    matplotlib.use('Agg')
    
    T, S = obs_mask.shape
    
    # Adaptive point size based on number of sites
    # Base size: 13 for 1000 sites, scale inversely with sqrt(S/1000)
    point_size = max(5, min(100, 13 * np.sqrt(1000 / S)))
    
    # Count observations per site
    obs_counts = obs_mask.sum(axis=0)  # (S,)
    train_counts = train_mask.sum(axis=0)  # (S,)
    valid_counts = valid_mask.sum(axis=0)  # (S,)
    test_counts = T - obs_counts  # Test = unobserved
    
    # Create 2x2 subplot
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Total observations
    ax = axes[0, 0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=obs_counts, cmap='viridis', s=point_size, alpha=0.7)
    ax.set_title(f'Total Observations per Site\n(Total: {obs_mask.sum()} obs)', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar(scatter, ax=ax, label='# observations')
    cbar.ax.tick_params(labelsize=11)
    
    # Train observations
    ax = axes[0, 1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=train_counts, cmap='Blues', s=point_size, alpha=0.7)
    ax.set_title(f'Train Observations per Site\n(Total: {train_mask.sum()} obs)', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar(scatter, ax=ax, label='# observations')
    cbar.ax.tick_params(labelsize=11)
    
    # Valid observations
    ax = axes[1, 0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=valid_counts, cmap='Greens', s=point_size, alpha=0.7)
    ax.set_title(f'Valid Observations per Site\n(Total: {valid_mask.sum()} obs)', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar(scatter, ax=ax, label='# observations')
    cbar.ax.tick_params(labelsize=11)
    
    # Test (unobserved) count
    ax = axes[1, 1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=test_counts, cmap='Reds', s=point_size, alpha=0.7)
    ax.set_title(f'Test (Unobserved) per Site\n(Total: {(~obs_mask).sum()} obs)', fontsize=16, fontweight='bold')
    ax.set_xlabel('x', fontsize=14)
    ax.set_ylabel('y', fontsize=14)
    ax.tick_params(axis='both', which='major', labelsize=12)
    cbar = plt.colorbar(scatter, ax=ax, label='# unobserved')
    cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    save_path = output_dir / 'observation_pattern.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Observation pattern plot saved to {save_path}")


def plot_basis_evolution(model_initial, model_final, train_coords, output_dir, config, basis_centers_history=None):
    """
    Plot spatial basis centers before and after training
    
    Args:
        model_initial: model with initial basis centers
        model_final: trained model with final basis centers
        train_coords: (N, 2) training coordinates
        output_dir: output directory
        config: configuration dict
        basis_centers_history: list of (epoch, centers) tuples recording trajectory every 100 epochs
    """
    matplotlib.use('Agg')
    
    # Extract basis information
    centers_init = model_initial.spatial_basis.centers.detach().cpu().numpy()
    centers_final = model_final.spatial_basis.centers.detach().cpu().numpy()
    
    bandwidths_init = model_initial.spatial_basis.bandwidths.detach().cpu().numpy()
    bandwidths_final = model_final.spatial_basis.bandwidths.detach().cpu().numpy()
    
    learnable = config.get('spatial_learnable', False)
    init_method = config.get('spatial_init_method', 'uniform')
    k_spatial_centers = config.get('k_spatial_centers', [25, 81, 121])
    
    # Detect inactive basis (zero weights from Group Lasso)
    inactive_basis_mask = None
    sparsity_type = config.get('sparsity_penalty_type', 'none')
    if sparsity_type in ['group', 'sparse_group']:
        # Extract first layer weights for spatial basis
        # In δ reparameterization mode, use mlp_trunk; otherwise use mlp
        if hasattr(model_final, 'use_delta_reparameterization') and model_final.use_delta_reparameterization and model_final.mlp_trunk is not None:
            first_layer_weight = model_final.mlp_trunk[0].weight.data  # (hidden_dim, input_dim)
        else:
            first_layer_weight = model_final.mlp[0].weight.data  # (hidden_dim, input_dim)
        
        # Get spatial basis weights
        p = config.get('p_covariates', 0)
        k_spatial = model_final.k_spatial
        spatial_weights = first_layer_weight[:, p:p+k_spatial].T  # (k_spatial, hidden_dim)
        
        # Compute L2 norm for each basis (row)
        basis_norms = torch.norm(spatial_weights, dim=1).cpu().numpy()  # (k_spatial,)
        
        # Mark basis as inactive if norm is very small relative to max
        # Use configurable threshold (default: 1% of max norm)
        # This accounts for the fact that Group Lasso creates "small but nonzero" values
        # due to (1) bias terms, (2) Adam momentum, (3) numerical stability
        if basis_norms.max() > 0:
            relative_threshold = config.get('sparsity_threshold_ratio', 1e-2)  # Default: 1% of max
            threshold = relative_threshold * basis_norms.max()
        else:
            threshold = 0.0
        
        inactive_basis_mask = basis_norms < threshold
        
        n_inactive = inactive_basis_mask.sum()
        n_active = (~inactive_basis_mask).sum()
        
        # Show detailed statistics
        print(f"\n  [INFO] Sparsity Analysis ({sparsity_type} penalty):")
        print(f"     Active basis: {n_active}/{len(inactive_basis_mask)}")
        print(f"     Removed basis: {n_inactive}/{len(inactive_basis_mask)} (norm < {threshold:.4f})")
        print(f"     Basis norms: min={basis_norms.min():.4f}, max={basis_norms.max():.4f}, "
              f"median={np.median(basis_norms):.4f}, mean={basis_norms.mean():.4f}")
        
        # Show distribution of norms
        sorted_norms = np.sort(basis_norms)
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = np.percentile(sorted_norms, percentiles)
        print(f"     Percentiles: ", end="")
        for p, v in zip(percentiles, percentile_values):
            print(f"P{p}={v:.4f} ", end="")
        print()
        
        # Count how many are "very small" (different thresholds)
        very_small_1e3 = (basis_norms < 1e-3 * basis_norms.max()).sum()
        very_small_1e2 = (basis_norms < 1e-2 * basis_norms.max()).sum()
        very_small_1e1 = (basis_norms < 1e-1 * basis_norms.max()).sum()
        print(f"     Small norms: <0.1%max: {very_small_1e3}, <1%max: {very_small_1e2}, <10%max: {very_small_1e1}")

    # Sample training data for visualization (max 20000 points)
    max_train_vis = 20000
    if len(train_coords) > max_train_vis:
        indices = np.random.choice(len(train_coords), max_train_vis, replace=False)
        train_coords_vis = train_coords[indices]
    else:
        train_coords_vis = train_coords
    
    # Create figure
    if learnable:
        fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(16, 7))
        axes = [axes[0], axes[1], None]
    
    # Normalize bandwidth for marker size
    def bw_to_size(bw):
        bw_norm = (bw - bw.min()) / (bw.max() - bw.min() + 1e-8)
        return 20 + bw_norm * 180  # Range [20, 200]
    
    sizes_init = bw_to_size(bandwidths_init)
    sizes_final = bw_to_size(bandwidths_final)
    
    # Color by resolution
    colors = []
    offset = 0
    color_map = ['red', 'blue', 'green']
    for i, k in enumerate(k_spatial_centers):
        colors.extend([color_map[i % 3]] * k)
        offset += k
    
    # Plot 1: Initial basis
    ax = axes[0]
    ax.scatter(train_coords_vis[:, 0], train_coords_vis[:, 1], 
              c='lightgray', s=2, alpha=0.3, label='Train data', rasterized=True)
    
    # Draw domain boundary
    from matplotlib.patches import Rectangle
    rect = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', linestyle='--', label='Domain [0,1]²')
    ax.add_patch(rect)
    
    for i, (center, size, color) in enumerate(zip(centers_init, sizes_init, colors)):
        ax.scatter(center[0], center[1], c=color, s=size, marker='o', 
                  alpha=0.6, edgecolors='black', linewidths=0.5)
    
    # Legend for resolutions
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=color_map[i], label=f'Resolution {i+1} (k={k})') 
                      for i, k in enumerate(k_spatial_centers)]
    legend_elements.insert(0, Patch(facecolor='lightgray', label='Train data'))
    
    ax.set_title(f'Initial Basis Centers\n({init_method} initialization)', fontsize=20, fontweight='bold')
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    ax.grid(True, alpha=0.2)
    
    # Plot 2: Final basis (after training)
    ax = axes[1]
    ax.scatter(train_coords_vis[:, 0], train_coords_vis[:, 1], 
              c='lightgray', s=2, alpha=0.3, label='Train data', rasterized=True)
    
    # Draw domain boundary
    rect = Rectangle((0, 0), 1, 1, linewidth=2, edgecolor='black', facecolor='none', linestyle='--')
    ax.add_patch(rect)
    
    # Count out-of-domain centers
    out_of_domain = ((centers_final < 0) | (centers_final > 1)).any(axis=1).sum()
    
    # Plot basis centers with different alpha for inactive ones
    for i, (center, size, color) in enumerate(zip(centers_final, sizes_final, colors)):
        # Determine alpha based on whether basis is active or inactive
        if inactive_basis_mask is not None and inactive_basis_mask[i]:
            alpha = 0.15  # Very faint for inactive basis
            edge_alpha = 0.3
        else:
            alpha = 0.6  # Normal for active basis
            edge_alpha = 1.0
        
        ax.scatter(center[0], center[1], c=color, s=size, marker='o', 
                  alpha=alpha, edgecolors='black', linewidths=0.5)
    
    # Update title with sparsity info
    title_suffix = ' (LEARNED)' if learnable else ' (FIXED - same as initial)'
    if learnable and out_of_domain > 0:
        title_suffix += f'\n[WARNING] {out_of_domain} centers out-of-domain'
    if inactive_basis_mask is not None:
        n_inactive = inactive_basis_mask.sum()
        n_active = (~inactive_basis_mask).sum()
        title_suffix += f'\n[INFO] {n_active} active, {n_inactive} removed (sparsity)'
    
    ax.set_title(f'Final Basis Centers{title_suffix}', fontsize=20, fontweight='bold')
    ax.set_xlabel('x', fontsize=18)
    ax.set_ylabel('y', fontsize=18)
    ax.tick_params(axis='both', which='major', labelsize=14)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.legend(handles=legend_elements, loc='upper right', fontsize=14)
    ax.grid(True, alpha=0.2)
    
    # Plot 3: Movement (only if learnable)
    if learnable and axes[2] is not None:
        ax = axes[2]
        ax.scatter(train_coords_vis[:, 0], train_coords_vis[:, 1], 
                  c='lightgray', s=2, alpha=0.3, label='Train data', rasterized=True)
        
        # Plot movement paths as polylines
        if basis_centers_history is not None and len(basis_centers_history) > 0:
            # Build complete trajectory: initial -> intermediate points -> final
            trajectory = []
            trajectory.append((0, centers_init))  # Epoch 0: initial
            trajectory.extend(basis_centers_history)  # Intermediate epochs (every 100)
            trajectory.append((-1, centers_final))  # Final epoch
            
            # Draw polyline for each basis center
            for i in range(len(centers_init)):
                # Skip inactive basis in movement visualization
                if inactive_basis_mask is not None and inactive_basis_mask[i]:
                    continue
                
                # Extract trajectory for this basis
                path = np.array([traj[1][i] for traj in trajectory])
                
                # Compute total path length to decide if we should show it
                total_distance = np.sum(np.linalg.norm(path[1:] - path[:-1], axis=1))
                
                if total_distance > 0.005:  # Only show significant movements
                    # Draw polyline (no markers at intermediate points)
                    ax.plot(path[:, 0], path[:, 1], 
                           color=colors[i], alpha=0.5, linewidth=1.5, zorder=1)
        else:
            # Fallback: draw straight arrows (original behavior)
            for i, (c_init, c_final, color) in enumerate(zip(centers_init, centers_final, colors)):
                # Skip inactive basis in movement visualization
                if inactive_basis_mask is not None and inactive_basis_mask[i]:
                    continue
                
                # Draw arrow from initial to final
                dx = c_final[0] - c_init[0]
                dy = c_final[1] - c_init[1]
                distance = np.sqrt(dx**2 + dy**2)
                
                if distance > 0.005:  # Only show significant movements
                    ax.arrow(c_init[0], c_init[1], dx, dy,
                            head_width=0.015, head_length=0.01, 
                            fc=color, ec='black', alpha=0.5, linewidth=0.5)
        
        # Plot initial and final positions
        # Separate active and inactive basis
        if inactive_basis_mask is not None:
            # Plot active basis (normal)
            active_mask = ~inactive_basis_mask
            if active_mask.any():
                ax.scatter(centers_init[active_mask, 0], centers_init[active_mask, 1], 
                          c='white', s=16, marker='o', alpha=0.8, 
                          edgecolors='black', linewidths=1.5, label='Initial (active)', zorder=2)
                ax.scatter(centers_final[active_mask, 0], centers_final[active_mask, 1], 
                          c=np.array(colors)[active_mask], s=80, marker='o', alpha=0.8, 
                          edgecolors='black', linewidths=1.5, label='Final (active)', zorder=2)
            
            # Plot inactive basis (faded)
            if inactive_basis_mask.any():
                ax.scatter(centers_init[inactive_basis_mask, 0], centers_init[inactive_basis_mask, 1], 
                          c='white', s=8, marker='o', alpha=0.2, 
                          edgecolors='gray', linewidths=0.5, label='Removed by sparsity', zorder=2)
                ax.scatter(centers_final[inactive_basis_mask, 0], centers_final[inactive_basis_mask, 1], 
                          c=np.array(colors)[inactive_basis_mask], s=40, marker='o', alpha=0.2, 
                          edgecolors='gray', linewidths=0.5, zorder=2)
        else:
            # Original behavior when no sparsity
            ax.scatter(centers_init[:, 0], centers_init[:, 1], 
                      c='white', s=16, marker='o', alpha=0.8, 
                      edgecolors='black', linewidths=1.5, label='Initial', zorder=2)
            ax.scatter(centers_final[:, 0], centers_final[:, 1], 
                      c=colors, s=80, marker='o', alpha=0.8, 
                      edgecolors='black', linewidths=1.5, label='Final', zorder=2)
        
        # Compute movement statistics (only for active basis)
        if inactive_basis_mask is not None:
            active_mask = ~inactive_basis_mask
            movements = np.linalg.norm(centers_final[active_mask] - centers_init[active_mask], axis=1)
        else:
            movements = np.linalg.norm(centers_final - centers_init, axis=1)
        
        mean_movement = movements.mean() if len(movements) > 0 else 0
        max_movement = movements.max() if len(movements) > 0 else 0
        median_movement = np.median(movements) if len(movements) > 0 else 0
        
        # Get penalty weights from config
        movement_penalty = config.get('movement_penalty_weight', 0.0)
        
        title_text = f'Basis Movement\n'
        title_text += f'Mean: {mean_movement:.4f}, Max: {max_movement:.4f}, Median: {median_movement:.4f}\n'
        if movement_penalty > 0:
            title_text += f'Movement Penalty: λ={movement_penalty}'
        
        ax.set_title(title_text, fontsize=18, fontweight='bold')
        ax.set_xlabel('x', fontsize=18)
        ax.set_ylabel('y', fontsize=18)
        ax.tick_params(axis='both', which='major', labelsize=14)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_aspect('equal')
        ax.legend(loc='upper right', fontsize=14)
        ax.grid(True, alpha=0.2)
    
    plt.tight_layout()
    save_path = output_dir / 'basis_evolution.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Basis evolution plot saved to {save_path}")


def run_single_experiment(config: dict, experiment_id: int, output_dir: Path, device: str, verbose: bool = True, parallel_mode: bool = False, skip_existing: bool = False):
    """
    Run a single experiment
    
    Args:
        config: configuration dictionary
        experiment_id: experiment ID (1, 2, ..., M)
        output_dir: output directory for this experiment
        device: torch device
        verbose: whether to print detailed logs
        parallel_mode: whether running in parallel mode (disables DataLoader workers)
        skip_existing: if True, skip if results.json already exists
    
    Returns:
        results: dictionary containing all results (or dict of results per quantile)
    """
    # Check if this is quantile regression with multiple quantiles
    regression_type = config.get('regression_type', 'mean')
    quantile_levels = config.get('quantile_levels', [0.5])
    
    # For multi-quantile, train a single model
    if regression_type == 'multi-quantile':
        if verbose:
            print("\n" + "="*70)
            print(f"MULTI-QUANTILE REGRESSION: Training single model for {len(quantile_levels)} quantiles")
            print(f"Quantiles: {quantile_levels}")
            print("="*70)
        
        # Run single experiment with multi-quantile model
        result = _run_single_quantile_experiment(
            config, experiment_id, output_dir, device, 
            verbose=verbose, parallel_mode=parallel_mode
        )
        
        return result
    
    # For single-quantile regression, train separate models per quantile
    elif regression_type == 'quantile' and len(quantile_levels) > 1:
        # Run separate experiment for each quantile level
        if verbose:
            print("\n" + "="*70)
            print(f"QUANTILE REGRESSION: Training {len(quantile_levels)} models")
            print(f"Quantiles: {quantile_levels}")
            print("="*70)
        
        quantile_results = {}
        quantile_predictions = {}
        
        for q_idx, q_level in enumerate(quantile_levels, 1):
            if verbose:
                print(f"\n{'='*70}")
                print(f"QUANTILE {q_idx}/{len(quantile_levels)}: τ = {q_level}")
                print(f"{'='*70}")
            
            # Create config for this quantile
            q_config = config.copy()
            q_config['current_quantile'] = q_level
            q_config['regression_type'] = 'quantile'
            
            # Create quantile-specific output directory
            q_output_dir = output_dir / f'quantile_{q_level}'
            q_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this quantile already exists
            if skip_existing:
                q_result_file = q_output_dir / 'results.json'
                if q_result_file.exists():
                    if verbose:
                        print(f"[OK] Quantile {q_level} already completed, loading...")
                    with open(q_result_file, 'r') as f:
                        quantile_results[q_level] = json.load(f)
                    
                    # Load predictions
                    pred_file = q_output_dir / 'predictions.npz'
                    if pred_file.exists():
                        pred_data = np.load(pred_file)
                        quantile_predictions[q_level] = {
                            'test': pred_data['test_predictions'],
                            'valid': pred_data['valid_predictions']
                        }
                    continue
            
            # Run experiment for this quantile (recursive call with single quantile)
            q_result = _run_single_quantile_experiment(
                q_config, experiment_id, q_output_dir, device, 
                verbose=verbose, parallel_mode=parallel_mode
            )
            
            quantile_results[q_level] = q_result
            quantile_predictions[q_level] = {
                'train': q_result.get('train_predictions'),
                'test': q_result.get('test_predictions'),
                'valid': q_result.get('valid_predictions')
            }
        
        # Compute CRPS across quantiles
        if verbose:
            print(f"\n{'='*70}")
            print(f"COMPUTING CRPS ACROSS QUANTILES")
            print(f"{'='*70}")
        
        # Aggregate results
        train_true = quantile_results[quantile_levels[0]].get('train_true')
        test_true = quantile_results[quantile_levels[0]].get('test_true')
        valid_true = quantile_results[quantile_levels[0]].get('valid_true')
        
        train_preds_dict = {q: quantile_predictions[q]['train'] for q in quantile_levels}
        test_preds_dict = {q: quantile_predictions[q]['test'] for q in quantile_levels}
        valid_preds_dict = {q: quantile_predictions[q]['valid'] for q in quantile_levels}
        
        train_crps = compute_crps(train_preds_dict, train_true)
        test_crps = compute_crps(test_preds_dict, test_true)
        valid_crps = compute_crps(valid_preds_dict, valid_true)
        
        # Average check loss across quantiles (use check_loss from each quantile result)
        test_check_loss = np.mean([quantile_results[q].get('test_check_loss', quantile_results[q].get('test_mse')) for q in quantile_levels])
        valid_check_loss = np.mean([quantile_results[q].get('valid_check_loss', quantile_results[q].get('valid_mse')) for q in quantile_levels])
        train_check_loss = np.mean([quantile_results[q].get('train_check_loss', quantile_results[q].get('train_mse')) for q in quantile_levels])
        
        # Also compute average time
        total_time = np.sum([quantile_results[q].get('total_time_seconds', 0) for q in quantile_levels])
        
        if verbose:
            print(f"\nTrain CRPS: {train_crps:.6f}")
            print(f"Test  CRPS: {test_crps:.6f}")
            print(f"Valid CRPS: {valid_crps:.6f}")
            print(f"Train Check Loss (avg): {train_check_loss:.6f}")
            print(f"Test  Check Loss (avg): {test_check_loss:.6f}")
            print(f"Valid Check Loss (avg): {valid_check_loss:.6f}")
        
        # Save aggregated results
        aggregated_results = {
            'experiment_id': experiment_id,
            'regression_type': 'quantile',
            'quantile_levels': quantile_levels,
            'quantile_results': quantile_results,
            'train_crps': float(train_crps),
            'test_crps': float(test_crps),
            'valid_crps': float(valid_crps),
            'train_check_loss': float(train_check_loss),
            'test_check_loss': float(test_check_loss),
            'valid_check_loss': float(valid_check_loss),
            # Store as standard metric names for summary generation
            'test_mse': float(test_check_loss),
            'valid_mse': float(valid_check_loss),
            'train_mse': float(train_check_loss),
            'test_rmse': float(np.sqrt(test_check_loss)),
            'valid_rmse': float(np.sqrt(valid_check_loss)),
            'train_rmse': float(np.sqrt(train_check_loss)),
            'test_mae': float(np.mean([quantile_results[q].get('test_mae', 0) for q in quantile_levels])),
            'valid_mae': float(np.mean([quantile_results[q].get('valid_mae', 0) for q in quantile_levels])),
            'train_mae': float(np.mean([quantile_results[q].get('train_mae', 0) for q in quantile_levels])),
            'total_time_seconds': float(total_time),
        }
        
        # Use save_results function to handle JSON serialization
        save_results(aggregated_results, output_dir)
        
        # Generate combined quantile temporal plots
        if verbose:
            print(f"\n{'='*70}")
            print(f"GENERATING COMBINED QUANTILE TEMPORAL PLOTS")
            print(f"{'='*70}")
        
        # Load models for plotting
        quantile_models = {}
        for q_level in quantile_levels:
            q_output_dir = output_dir / f'quantile_{q_level}'
            model_path = q_output_dir / 'model_final.pt'
            if model_path.exists():
                # Recreate model (use first quantile's result to get config info)
                first_result = quantile_results[quantile_levels[0]]
                model_config = first_result.get('config', config)
                
                # Import model class
                from stnf.models.st_interp import STInterpMLP
                
                q_model = STInterpMLP(
                    k_spatial_centers=model_config['k_spatial_centers'],
                    k_temporal_centers=model_config['k_temporal_centers'],
                    spatial_basis_function=model_config.get('spatial_basis_function', 'wendland'),
                    spatial_init_method='uniform',  # Use uniform to avoid train_coords requirement
                    spatial_learnable=model_config.get('spatial_learnable', False),
                    hidden_dims=model_config['hidden_dims'],
                    p=model_config.get('p_covariates', 0),  # Use 'p' parameter name
                    dropout=model_config.get('dropout', 0.0),
                    layernorm=model_config.get('layernorm', False)
                ).to(device)
                
                q_model.load_state_dict(torch.load(model_path, map_location=device))
                quantile_models[q_level] = q_model
        
        # Get data from first quantile result
        first_q_dir = output_dir / f'quantile_{quantile_levels[0]}'
        pred_file = first_q_dir / 'predictions.npz'
        if pred_file.exists():
            pred_data = np.load(pred_file)
            z_full_np = pred_data['true']
            coords_np = pred_data['coords']
            train_mask_np = pred_data['train_mask']
            valid_mask_np = pred_data['valid_mask']
            test_mask_np = pred_data['test_mask']
            
            # Convert to tensors
            z_full_tensor = torch.tensor(z_full_np, dtype=torch.float32)
            coords_tensor = torch.tensor(coords_np, dtype=torch.float32)
            
            # Plot combined quantile temporal series
            plot_temporal_series(
                None,  # No base model, only quantile models
                z_full_tensor, coords_tensor, train_mask_np, device, output_dir,
                valid_mask=valid_mask_np, test_mask=test_mask_np, n_sites=4,
                quantile_models=quantile_models, quantile_levels=quantile_levels
            )
        
        return aggregated_results
    
    else:
        # Single quantile or mean regression - use existing logic
        if regression_type == 'quantile':
            config['current_quantile'] = quantile_levels[0]
        
        return _run_single_quantile_experiment(
            config, experiment_id, output_dir, device, 
            verbose=verbose, parallel_mode=parallel_mode
        )


def _run_single_quantile_experiment(config: dict, experiment_id: int, output_dir: Path, device: str, verbose: bool = True, parallel_mode: bool = False):
    """
    Run a single experiment for one quantile (or mean regression)
    This is the original run_single_experiment logic
    """
    # Check if results already exist (removed skip_existing logic, handled by parent)
    
    start_time = time.time()
    
    if verbose:
        print("\n" + "="*70)
        print(f"EXPERIMENT {experiment_id}")
        print("="*70)
    
    # Set seed for this experiment
    experiment_seed = config.get('base_seed', 42) + experiment_id - 1
    set_seed(experiment_seed)
    if verbose:
        print(f"Experiment seed: {experiment_seed}")
    
    # Load data (without normalization first, to avoid test data leakage)
    use_provider_split = config.get('use_provider_split', False)
    data_file = config.get('data_file', 'data/2b/2b_7.csv')

    if use_provider_split:
        # Use provider's train/test files (e.g. 2b_8_train.csv, 2b_8_test.csv); test z from full CSV
        if verbose:
            print("\nLoading data (provider train/test split)...")
        base = str(Path(data_file).with_suffix(''))
        if base.endswith('_train'):
            base = base[:-6]  # strip _train
        train_path = base + '_train.csv'
        test_path = base + '_test.csv'
        full_path = base + '.csv'
        z_full, coords, metadata = load_kaust_csv_with_test_gt(
            train_path, test_path, full_path,
            normalize=False  # we normalize below using train only
        )
        T_full, S = z_full.shape
        T_tr = metadata['T_tr']
        if verbose:
            print(f"Full data shape: {z_full.shape} (train t=1..{T_tr}, test t={T_tr+1}..{T_full})")
        obs_mask = np.zeros((T_full, S), dtype=bool)
        obs_mask[:T_tr, :] = True
        train_mask = obs_mask.copy()
        valid_mask = np.zeros_like(obs_mask, dtype=bool)
        cal_mask = np.zeros_like(obs_mask, dtype=bool)
        test_mask = np.zeros_like(obs_mask, dtype=bool)
        test_mask[T_tr:, :] = True
        print("Using provider train/test split (no observation sampling)")
        print(f"Train: {train_mask.sum()} samples (t=1..{T_tr})")
        print(f"Valid: {valid_mask.sum()} samples")
        print(f"Test: {test_mask.sum()} samples (t={T_tr+1}..{T_full})")
    else:
        if verbose:
            print("\nLoading data...")
        z_full, coords, metadata = load_kaust_csv_single(
            data_file,
            normalize=False  # Don't normalize yet - we'll normalize based on observed data only
        )
        if verbose:
            print(f"Full data shape: {z_full.shape}, Coords: {coords.shape}")

        # Sample observations (train + valid)
        if verbose:
            print("\nSampling observations...")
        obs_method = config.get('obs_method', 'site-wise')
        obs_ratio = config.get('obs_ratio', 0.5)

        # Define observation probability function
        obs_spatial_pattern = config.get('obs_spatial_pattern', 'uniform')
        obs_spatial_intensity = config.get('obs_spatial_intensity', 1.0)

        obs_prob_fn = create_spatial_obs_prob_fn(
            pattern=obs_spatial_pattern,
            intensity=obs_spatial_intensity
        )

        obs_mask, obs_sites = sample_observations(
            z_full, coords,
            obs_method=obs_method,
            obs_ratio=obs_ratio,
            obs_prob_fn=obs_prob_fn,
            seed=experiment_seed  # Use experiment-specific seed
        )

        n_obs_total = obs_mask.sum()
        print(f"Observation method: {obs_method}")
        if obs_spatial_pattern != 'uniform':
            print(f"Spatial pattern: {obs_spatial_pattern} (intensity={obs_spatial_intensity})")
        print(f"Observed: {n_obs_total} / {z_full.size} ({n_obs_total/z_full.size*100:.1f}%)")
        print(f"Observed sites: {len(obs_sites)} / {coords.shape[0]}")

        # Split train and validation (from observed data)
        # Paper: Use all 10% observations for training, no validation split
        # Default to train_ratio=1.0 to match paper (all obs for training)
        print("\nSplitting train/valid...")
        split_method = config.get('split_method', 'site-wise')
        train_ratio = config.get('train_ratio', 1.0)  # Changed default to 1.0 to match paper

        if train_ratio >= 1.0:
            # Use all observations for training, no validation set (matches paper)
            train_mask = obs_mask.copy()
            valid_mask = np.zeros_like(obs_mask, dtype=bool)
            print("Using all observations for training (train_ratio=1.0, matches paper)")
            # Option B: split a calibration set from train for conformal when valid is empty
            calibration_ratio_from_train = config.get('calibration_ratio_from_train', 0.0)
            calibration_split_method = config.get('calibration_split_method', 'random')
            if calibration_ratio_from_train > 0 and config.get('regression_type') == 'multi-quantile':
                if calibration_split_method == 'site-wise':
                    # Use a subset of observed *sites* as calibration (cal more like test: different sites).
                    # calibration_ratio_from_train here = fraction of *sites*, not samples; cal sample count depends on obs per site.
                    if len(obs_sites) < 2:
                        cal_mask = np.zeros_like(train_mask, dtype=bool)
                        print("[WARNING] calibration_split_method=site-wise but only 1 observed site; skipping calibration split.")
                    else:
                        n_cal_sites = max(1, min(int(len(obs_sites) * calibration_ratio_from_train), len(obs_sites) - 1))  # keep >= 1 site for train
                        rng = np.random.default_rng(experiment_seed + 20000)
                        site_order = np.array(obs_sites, dtype=int)
                        rng.shuffle(site_order)
                        cal_sites = set(site_order[:n_cal_sites])
                        cal_mask = np.zeros_like(train_mask, dtype=bool)
                        train_mask_new = np.zeros_like(train_mask, dtype=bool)
                        for s in obs_sites:
                            if s in cal_sites:
                                cal_mask[:, s] = train_mask[:, s]
                            else:
                                train_mask_new[:, s] = train_mask[:, s]
                        train_mask = train_mask_new
                        pct_sites = 100.0 * len(cal_sites) / len(obs_sites) if len(obs_sites) > 0 else 0.0
                        cal_ratio_samples = cal_mask.sum() / n_obs_total if n_obs_total > 0 else 0.0
                        print(
                            "Calibration from train (site-wise): "
                            f"{cal_mask.sum()} samples ({len(cal_sites)} sites, "
                            f"{pct_sites:.0f}% of sites, {cal_ratio_samples*100:.1f}% of obs) for conformal"
                        )
                else:
                    # random: random (t,s) from train (default; calibration_ratio_from_train = fraction of *samples*)
                    flat_idx = np.where(train_mask.ravel())[0]
                    if len(flat_idx) < 2:
                        cal_mask = np.zeros_like(train_mask, dtype=bool)
                        print("[WARNING] calibration_split_method=random but <2 observed samples; skipping calibration split.")
                    else:
                        n_cal = max(1, min(int(len(flat_idx) * calibration_ratio_from_train), len(flat_idx) - 1))  # keep >= 1 sample for train
                        rng = np.random.default_rng(experiment_seed + 20000)
                        rng.shuffle(flat_idx)
                        cal_flat = flat_idx[:n_cal]
                        train_flat = flat_idx[n_cal:]
                        cal_mask = np.zeros_like(train_mask, dtype=bool)
                        cal_mask.ravel()[cal_flat] = True
                        train_mask = np.zeros_like(train_mask, dtype=bool)
                        train_mask.ravel()[train_flat] = True
                        cal_ratio_samples = cal_mask.sum() / n_obs_total if n_obs_total > 0 else 0.0
                        print(
                            "Calibration from train (random): "
                            f"{cal_mask.sum()} samples ({cal_ratio_samples*100:.1f}% of obs) for conformal"
                        )
            else:
                cal_mask = np.zeros_like(obs_mask, dtype=bool)
        else:
            train_mask, valid_mask = split_train_valid(
                obs_mask, obs_sites,
                split_method=split_method,
                train_ratio=train_ratio,
                seed=experiment_seed + 10000  # Different seed for split
            )
            cal_mask = np.zeros_like(obs_mask, dtype=bool)  # no cal split when we have valid

        print(f"Split method: {split_method}")
        print(f"Train: {train_mask.sum()} samples")
        print(f"Valid: {valid_mask.sum()} samples")
        if cal_mask.sum() > 0:
            print(f"Calibration (from train): {cal_mask.sum()} samples")
        if train_mask.sum() + valid_mask.sum() > 0:
            print(f"Actual train ratio: {train_mask.sum() / (train_mask.sum() + valid_mask.sum()):.3f}")

        # Test set: all non-observed data
        test_mask = ~obs_mask
        print(f"Test: {test_mask.sum()} samples (all unobserved data)")
    
    # Normalize based on observed data only (train + valid), then apply to all data
    # This prevents test data statistics from leaking into normalization
    # Strategy C1: Option to normalize on all data (including test) if normalize_on_all_data=True
    # Strategy C4: Option to center-only (subtract mean, don't divide by std) for zero-mean GP
    center_only = config.get('center_only', False)
    if config.get('normalize_target', False) or center_only:
        normalize_on_all_data = config.get('normalize_on_all_data', False)
        
        if normalize_on_all_data:
            if verbose:
                print("\nNormalizing data based on ALL data (including test)...")
            # Compute normalization stats from all data (including test)
            z_all = z_full[~np.isnan(z_full)]
            z_mean = z_all.mean()
            z_std = z_all.std() + 1e-8  # Add small epsilon to avoid division by zero
            
            # Apply normalization to all data
            if center_only:
                # Center-only: subtract mean, don't divide by std (for zero-mean GP)
                z_full = z_full - z_mean
                metadata['z_mean'] = z_mean
                metadata['z_std'] = 1.0  # No scaling
                if verbose:
                    print(f"[INFO] Center-only normalization (based on ALL data): mean={z_mean:.4f}, std=1.0 (no scaling)")
                    print(f"  All data range: [{z_all.min():.4f}, {z_all.max():.4f}]")
                    print(f"  Centered all data range: [{z_full[~np.isnan(z_full)].min():.4f}, {z_full[~np.isnan(z_full)].max():.4f}]")
            else:
                # Full normalization: subtract mean and divide by std
                z_full = (z_full - z_mean) / z_std
                metadata['z_mean'] = z_mean
                metadata['z_std'] = z_std
                if verbose:
                    print(f"[INFO] Normalized z (based on ALL data): mean={z_mean:.4f}, std={z_std:.4f}")
                    print(f"  All data range: [{z_all.min():.4f}, {z_all.max():.4f}]")
                    print(f"  Normalized all data range: [{z_full[~np.isnan(z_full)].min():.4f}, {z_full[~np.isnan(z_full)].max():.4f}]")
        else:
            if verbose:
                print("\nNormalizing data based on observed (train+valid) data only...")
            # Compute normalization stats from observed data only
            # When train_ratio=1.0, valid_mask is empty, so use train_mask only
            if train_mask.sum() > 0:
                observed_mask = train_mask | valid_mask
            else:
                observed_mask = valid_mask  # Fallback if train_mask is also empty
            z_observed = z_full[observed_mask]
            z_mean = z_observed.mean()
            z_std = z_observed.std() + 1e-8  # Add small epsilon to avoid division by zero
            
            # Apply normalization to all data (including test)
            if center_only:
                # Center-only: subtract mean, don't divide by std (for zero-mean GP)
                z_full = z_full - z_mean
                metadata['z_mean'] = z_mean
                metadata['z_std'] = 1.0  # No scaling
                if verbose:
                    print(f"[INFO] Center-only normalization (based on observed data): mean={z_mean:.4f}, std=1.0 (no scaling)")
                    print(f"  Observed data range: [{z_observed.min():.4f}, {z_observed.max():.4f}]")
                    print(f"  Centered observed range: [{z_full[observed_mask].min():.4f}, {z_full[observed_mask].max():.4f}]")
            else:
                # Full normalization: subtract mean and divide by std
                z_full = (z_full - z_mean) / z_std
                metadata['z_mean'] = z_mean
                metadata['z_std'] = z_std
                if verbose:
                    print(f"[INFO] Normalized z (based on observed data): mean={z_mean:.4f}, std={z_std:.4f}")
                    print(f"  Observed data range: [{z_observed.min():.4f}, {z_observed.max():.4f}]")
                    print(f"  Normalized observed range: [{z_full[observed_mask].min():.4f}, {z_full[observed_mask].max():.4f}]")
    else:
        metadata['z_mean'] = 0.0
        metadata['z_std'] = 1.0
    
    # Create datasets
    print("\nCreating datasets...")
    p_covariates = config.get('p_covariates', 0)
    
    train_dataset = create_dataset_from_mask(z_full, coords, train_mask, p_covariates)
    val_dataset = create_dataset_from_mask(z_full, coords, valid_mask, p_covariates)
    cal_dataset = create_dataset_from_mask(z_full, coords, cal_mask, p_covariates)
    test_dataset = create_dataset_from_mask(z_full, coords, test_mask, p_covariates)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    if len(cal_dataset) > 0:
        print(f"Calibration dataset: {len(cal_dataset)} samples (from train split)")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    # In parallel mode with many jobs, disable DataLoader workers to avoid nested multiprocessing
    # If n_jobs is small (e.g., <= 16), we can safely use DataLoader workers
    if parallel_mode:
        # Check if we can use DataLoader workers
        num_workers_config = config.get('num_workers', 0)
        n_jobs = config.get('n_jobs', -1)
        
        # If n_jobs is set and small enough, allow DataLoader workers
        if n_jobs > 0 and n_jobs <= 16:
            num_workers = num_workers_config
        else:
            num_workers = 0
    else:
        num_workers = config.get('num_workers', 0)
    
    # Adjust batch size to ensure at least 10 batches per epoch
    batch_size = config.get('batch_size', 256)
    n_train_samples = len(train_dataset)
    min_batches_per_epoch = 10
    
    while n_train_samples / batch_size < min_batches_per_epoch and batch_size > 1:
        old_batch_size = batch_size
        batch_size = batch_size // 2
        batches_per_epoch = n_train_samples / batch_size
        print(f"[WARNING] Batch size {old_batch_size} would result in {n_train_samples / old_batch_size:.1f} batches/epoch")
        print(f"   Reducing to {batch_size} → {batches_per_epoch:.1f} batches/epoch")
    
    if batch_size != config.get('batch_size', 256):
        print(f"[OK] Final train batch size: {batch_size} ({n_train_samples / batch_size:.1f} batches/epoch)")
    
    # For validation/test, use larger batch size (no gradient computation needed)
    # Use 4x train batch size or all data at once, whichever is smaller
    # Handle empty validation set (train_ratio=1.0, matches paper)
    if len(val_dataset) > 0:
        val_batch_size = min(max(batch_size * 16, 32768), len(val_dataset))
    else:
        val_batch_size = 1  # Dummy batch size for empty dataset
    
    test_batch_size = min(max(batch_size * 16, 32768), len(test_dataset))
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Only create val_loader if validation set is not empty
    if len(val_dataset) > 0:
        val_loader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        # Create empty DataLoader for compatibility (won't be used)
        from torch.utils.data import TensorDataset
        dummy_dataset = TensorDataset(torch.empty(0, 1))
        val_loader = DataLoader(
            dummy_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
    
    # Calibration loader (Option B: from-train split when valid is empty)
    if len(cal_dataset) > 0:
        cal_batch_size = min(max(batch_size * 16, 32768), len(cal_dataset))
        cal_loader = DataLoader(
            cal_dataset,
            batch_size=cal_batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=collate_fn
        )
    else:
        from torch.utils.data import TensorDataset
        cal_loader = DataLoader(
            TensorDataset(torch.empty(0, 1)),
            batch_size=1,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_fn
        )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=test_batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nCreating model...")
    
    # Prepare training coordinates for data-adaptive initialization if needed
    train_coords = None
    init_method = config.get('spatial_init_method', 'uniform')
    if init_method in ['gmm', 'random_site', 'kmeans_balanced']:
        # Extract spatial coordinates from training data (with temporal duplicates)
        # Note: Keep all (s,t) pairs to reflect spatio-temporal data density
        # Sites with more temporal observations will have higher weight in clustering
        train_coords_list = []
        for sample in train_dataset:
            train_coords_list.append(sample['coords'].numpy())
        train_coords = np.array(train_coords_list)  # (N_train, 2)
        print(f"Using {init_method} initialization with {len(train_coords)} training samples")
    
    model = create_model(config, train_coords=train_coords)
    model = model.to(device)
    
    # Save initial model for visualization (deep copy of basis parameters)
    import copy
    model_initial = copy.deepcopy(model)
    model_initial.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Spatial basis initialization: {config.get('spatial_init_method', 'uniform')}")
    print(f"Total spatial basis functions: {model.spatial_basis.k}")
    
    # Print basis statistics if GMM
    if config.get('spatial_init_method', 'uniform') == 'gmm':
        centers = model.spatial_basis.centers.detach().cpu().numpy()
        bandwidths = model.spatial_basis.bandwidths.detach().cpu().numpy()
        print(f"  Center range: x=[{centers[:,0].min():.3f}, {centers[:,0].max():.3f}], "
              f"y=[{centers[:,1].min():.3f}, {centers[:,1].max():.3f}]")
        print(f"  Bandwidth range: [{bandwidths.min():.4f}, {bandwidths.max():.4f}]")
        print(f"  Bandwidth mean: {bandwidths.mean():.4f}")
        
        # Print per-resolution statistics
        offset = 0
        for i, n in enumerate(config.get('k_spatial_centers', [25, 81, 121])):
            bw_res = bandwidths[offset:offset+n]
            print(f"  Resolution {i+1} (n={n}): bandwidth mean={bw_res.mean():.4f}, std={bw_res.std():.4f}")
            offset += n
    
    # Train model
    print("\n" + "="*50)
    print("Training Model")
    print("="*50)
    model, history, basis_centers_history = train_model(model, train_loader, val_loader, config, device, output_dir)
    
    # Evaluate on all sets
    print("\n" + "="*50)
    print("Evaluating Model")
    print("="*50)
    
    print("\nEvaluating on Train set...")
    train_metrics = evaluate_model(model, train_loader, device, config)
    if config.get('regression_type') == 'quantile':
        print(f"Train - Check Loss: {train_metrics.get('check_loss', train_metrics['mse']):.6f}, MAE: {train_metrics['mae']:.6f}")
    elif config.get('regression_type') == 'multi-quantile':
        print(f"Train - CRPS: {train_metrics['crps']:.6f}, Mean Check Loss: {train_metrics['mean_check_loss']:.6f}, MAE: {train_metrics['mae']:.6f}")
    else:
        print(f"Train - MSE: {train_metrics['mse']:.6f}, MAE: {train_metrics['mae']:.6f}, RMSE: {train_metrics['rmse']:.6f}")
    
    # Evaluate on validation set (if exists)
    if len(val_dataset) > 0:
        print("\nEvaluating on Valid set...")
        val_metrics = evaluate_model(model, val_loader, device, config)
        if config.get('regression_type') == 'quantile':
            print(f"Valid - Check Loss: {val_metrics.get('check_loss', val_metrics['mse']):.6f}, MAE: {val_metrics['mae']:.6f}")
        elif config.get('regression_type') == 'multi-quantile':
            print(f"Valid - CRPS: {val_metrics['crps']:.6f}, Mean Check Loss: {val_metrics['mean_check_loss']:.6f}, MAE: {val_metrics['mae']:.6f}")
        else:
            print(f"Valid - MSE: {val_metrics['mse']:.6f}, MAE: {val_metrics['mae']:.6f}, RMSE: {val_metrics['rmse']:.6f}")
    else:
        print("\nNo validation set (train_ratio=1.0, matches paper)")
        # Create dummy val_metrics for compatibility
        regression_type = config.get('regression_type', 'mean')
        if regression_type == 'multi-quantile':
            val_metrics = {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'crps': 0.0,
                'mean_check_loss': 0.0,
                'check_loss': 0.0,
                'coverage_90': 0.0
            }
        else:
            val_metrics = {
                'mse': 0.0,
                'mae': 0.0,
                'rmse': 0.0,
                'check_loss': 0.0
            }
    
    print("\nEvaluating on Test set...")
    test_metrics = evaluate_model(model, test_loader, device, config)
    if config.get('regression_type') == 'quantile':
        print(f"Test  - Check Loss: {test_metrics.get('check_loss', test_metrics['mse']):.6f}, MAE: {test_metrics['mae']:.6f}")
    elif config.get('regression_type') == 'multi-quantile':
        print(f"Test  - CRPS: {test_metrics['crps']:.6f}, Mean Check Loss: {test_metrics['mean_check_loss']:.6f}, MAE: {test_metrics['mae']:.6f}")
        if 'coverage_90' in test_metrics:
            print(f"Test  - Coverage (90% PI): {test_metrics['coverage_90']:.4f} (target 0.90)")
    else:
        print(f"Test  - MSE: {test_metrics['mse']:.6f}, MAE: {test_metrics['mae']:.6f}, RMSE: {test_metrics['rmse']:.6f}")
    
    # Conformal calibration (CQR): multi-quantile + (valid set or calibration-from-train)
    conformal_alpha = 0.1
    conformal_qhat = None
    calibration_n = None
    calibration_coverage_90 = None  # diagnostic: coverage of conformal interval on cal set (target ~0.90)
    test_coverage_90_conformal_global = None
    test_coverage_90_conformal_cluster = None
    mean_qhat_global = None
    mean_qhat_cluster = None
    qhat_per_cluster_saved = None  # for spatial viz: dict or None
    cluster_centers_saved = None   # (C, 2) for cluster assignment
    calibration_loader = val_loader if len(val_dataset) > 0 else (cal_loader if len(cal_dataset) > 0 else None)
    if (
        config.get('regression_type') == 'multi-quantile'
        and calibration_loader is not None
        and len(calibration_loader.dataset) > 0
    ):
        quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
        conformal_mode = config.get('conformal_mode', 'both')
        cal_preds, cal_y_true = _get_quantile_predictions(model, calibration_loader, device)
        if cal_preds.size > 0:
            conformal_qhat, calibration_n = compute_cqr_qhat(
                cal_preds, cal_y_true, quantile_levels, alpha=conformal_alpha
            )
            mean_qhat_global = conformal_qhat
            print(f"Conformal qhat (global): {conformal_qhat:.6f} (calibration n={calibration_n})  [small qhat => interval already good or cal/test mismatch]")
            calibration_coverage_90 = compute_conformal_coverage(
                cal_preds, cal_y_true, quantile_levels, conformal_qhat, alpha=conformal_alpha
            )
            print(f"Calibration - Coverage (90% PI, conformal): {calibration_coverage_90:.4f} (target 0.90)")
            test_preds, test_y_true = _get_quantile_predictions(model, test_loader, device)
            if conformal_mode in ('global', 'both'):
                coverage_90_conformal = compute_conformal_coverage(
                    test_preds, test_y_true, quantile_levels, conformal_qhat, alpha=conformal_alpha
                )
                test_metrics['coverage_90_conformal'] = coverage_90_conformal
                test_coverage_90_conformal_global = coverage_90_conformal
                print(f"Test  - Coverage (90% PI, conformal global): {coverage_90_conformal:.4f} (target 0.90)")
            if conformal_mode in ('global', 'both'):
                train_preds, train_y_true = _get_quantile_predictions(model, train_loader, device)
                train_metrics['coverage_90_conformal'] = compute_conformal_coverage(
                    train_preds, train_y_true, quantile_levels, conformal_qhat, alpha=conformal_alpha
                )
                val_metrics['coverage_90_conformal'] = compute_conformal_coverage(
                    cal_preds, cal_y_true, quantile_levels, conformal_qhat, alpha=conformal_alpha
                )

            # Cluster-aware conformal (same run, fair comparison)
            if conformal_mode in ('cluster', 'both'):
                try:
                    cal_preds2, cal_y_true2, cal_coords = _get_quantile_predictions_with_coords(
                        model, calibration_loader, device
                    )
                    test_preds2, test_y_true2, test_coords = _get_quantile_predictions_with_coords(
                        model, test_loader, device
                    )
                    center_source = config.get('conformal_center_source', 'init')
                    if center_source == 'init' and model_initial is not None:
                        centers = model_initial.spatial_basis.centers.detach().cpu().numpy()
                    else:
                        centers = model.spatial_basis.centers.detach().cpu().numpy()
                    min_n = config.get('conformal_cluster_min_n', 30)
                    qhat_per_cluster, _, mean_qhat_cluster, num_fallback_clusters = compute_cluster_aware_cqr(
                        cal_preds2, cal_y_true2, cal_coords, centers,
                        quantile_levels, alpha=conformal_alpha, min_n=min_n,
                        global_qhat_fallback=conformal_qhat
                    )
                    qhat_per_cluster_saved = qhat_per_cluster
                    cluster_centers_saved = centers
                    test_coverage_90_conformal_cluster = compute_cluster_conformal_coverage(
                        test_preds2, test_y_true2, test_coords, centers, qhat_per_cluster,
                        quantile_levels, alpha=conformal_alpha, global_qhat_fallback=conformal_qhat
                    )
                    n_centers = len(centers)
                    print(f"Test  - Coverage (90% PI, conformal cluster): {test_coverage_90_conformal_cluster:.4f} (target 0.90)")
                    print(f"  mean_qhat_global={mean_qhat_global:.4f}, mean_qhat_cluster={mean_qhat_cluster:.4f}, "
                          f"clusters: {n_centers} total, {num_fallback_clusters} fallback (n<{min_n})")
                except Exception as e:
                    print(f"[WARNING] Cluster-aware conformal failed: {e}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save results
    config_with_dir = config.copy()
    config_with_dir['output_dir'] = str(output_dir)
    
    # For quantile regression, extract check_loss as primary metrics
    if config.get('regression_type') == 'quantile':
        results = {
            'experiment_id': experiment_id,
            'experiment_seed': experiment_seed,
            'regression_type': 'quantile',
            'quantile_level': config.get('current_quantile'),
            'config': config_with_dir,
            'metrics': {
                'train': train_metrics,
                'valid': val_metrics,
                'test': test_metrics
            },
            # Use check_loss as primary metric for quantile regression
            'train_check_loss': train_metrics.get('check_loss', train_metrics['mse']),
            'valid_check_loss': val_metrics.get('check_loss', val_metrics['mse']),
            'test_check_loss': test_metrics.get('check_loss', test_metrics['mse']),
            # Keep compatibility with old format
            'train_mse': train_metrics['mse'],
            'valid_mse': val_metrics['mse'],
            'test_mse': test_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'valid_mae': val_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'valid_rmse': val_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'training_history': history,
            'total_time_seconds': total_time,
            'total_time_formatted': f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}",
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    elif config.get('regression_type') == 'multi-quantile':
        results = {
            'experiment_id': experiment_id,
            'experiment_seed': experiment_seed,
            'regression_type': 'multi-quantile',
            'quantile_levels': config.get('quantile_levels'),
            'config': config_with_dir,
            'metrics': {
                'train': train_metrics,
                'valid': val_metrics,
                'test': test_metrics
            },
            # Use CRPS and mean check loss as primary metrics
            'train_crps': train_metrics['crps'],
            'valid_crps': val_metrics['crps'],
            'test_crps': test_metrics['crps'],
            'train_check_loss': train_metrics['mean_check_loss'],
            'valid_check_loss': val_metrics['mean_check_loss'],
            'test_check_loss': test_metrics['mean_check_loss'],
            # Empirical coverage of 90% prediction interval (for conformal / calibration)
            'train_coverage_90': train_metrics.get('coverage_90', 0.0),
            'valid_coverage_90': val_metrics.get('coverage_90', 0.0),
            'test_coverage_90': test_metrics.get('coverage_90', 0.0),
            # Conformal (CQR) coverage and params; only when calibration set exists
            'conformal_qhat': conformal_qhat,
            'conformal_alpha': conformal_alpha if conformal_qhat is not None else None,
            'calibration_n': calibration_n,
            'calibration_coverage_90': calibration_coverage_90,  # diagnostic: conformal coverage on cal set (target ~0.90)
            'train_coverage_90_conformal': train_metrics.get('coverage_90_conformal'),
            'valid_coverage_90_conformal': val_metrics.get('coverage_90_conformal'),
            'test_coverage_90_conformal': test_metrics.get('coverage_90_conformal'),
            'test_coverage_90_conformal_global': test_coverage_90_conformal_global,
            'test_coverage_90_conformal_cluster': test_coverage_90_conformal_cluster,
            'mean_qhat_global': mean_qhat_global,
            'mean_qhat_cluster': mean_qhat_cluster,
            # Keep compatibility with old format (using median quantile)
            'train_mse': train_metrics['mse'],
            'valid_mse': val_metrics['mse'],
            'test_mse': test_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'valid_mae': val_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'valid_rmse': val_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'training_history': history,
            'total_time_seconds': total_time,
            'total_time_formatted': f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}",
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    else:
        results = {
            'experiment_id': experiment_id,
            'experiment_seed': experiment_seed,
            'config': config_with_dir,
            'metrics': {
                'train': train_metrics,
                'valid': val_metrics,
                'test': test_metrics
            },
            'train_mse': train_metrics['mse'],
            'valid_mse': val_metrics['mse'],
            'test_mse': test_metrics['mse'],
            'train_mae': train_metrics['mae'],
            'valid_mae': val_metrics['mae'],
            'test_mae': test_metrics['mae'],
            'train_rmse': train_metrics['rmse'],
            'valid_rmse': val_metrics['rmse'],
            'test_rmse': test_metrics['rmse'],
            'training_history': history,
            'total_time_seconds': total_time,
            'total_time_formatted': f"{int(total_time//3600):02d}:{int((total_time%3600)//60):02d}:{int(total_time%60):02d}",
            'model_parameters': sum(p.numel() for p in model.parameters()),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    save_results(results, output_dir)
    
    # Save final model
    torch.save(model.state_dict(), output_dir / 'model_final.pt')
    print(f"\nModel saved to: {output_dir / 'model_final.pt'}")
    
    # Plot training curves
    plot_training_curves(history, output_dir / 'training_curves.png')
    
    # Plot observation pattern
    print("\n" + "="*50)
    print("Generating Observation Pattern Visualization")
    print("="*50)
    plot_observation_pattern(coords, obs_mask, train_mask, valid_mask, output_dir)
    
    # Plot prediction maps
    print("\n" + "="*50)
    print("Generating Prediction Visualizations")
    print("="*50)
    plot_predictions(model, z_full, coords, train_mask, device, output_dir, n_times=3)
    pred_data = plot_spatial_mse(model, z_full, coords, train_mask, device, output_dir, 
                                 return_predictions=True, valid_mask=valid_mask, test_mask=test_mask)
    
    # Plot temporal series
    print("\n" + "="*50)
    print("Generating Temporal Series Visualizations")
    print("="*50)
    
    # Pass quantile_levels for multi-quantile models
    regression_type = config.get('regression_type', 'mean')
    if regression_type == 'multi-quantile':
        quantile_levels = config.get('quantile_levels', [0.1, 0.5, 0.9])
        plot_temporal_series(model, z_full, coords, train_mask, device, output_dir,
                            valid_mask=valid_mask, test_mask=test_mask, n_sites=4,
                            quantile_levels=quantile_levels,
                            conformal_qhat=conformal_qhat, conformal_alpha=conformal_alpha)
    else:
        plot_temporal_series(model, z_full, coords, train_mask, device, output_dir,
                            valid_mask=valid_mask, test_mask=test_mask, n_sites=4)
    
    # Save predictions for later aggregation
    train_predictions_array = None
    test_predictions_array = None
    valid_predictions_array = None
    train_true_array = None
    test_true_array = None
    valid_true_array = None
    
    if pred_data is not None:
        all_predictions, z_full_data, coords_data, train_mask_data, valid_mask_data, test_mask_data = pred_data
        save_dict = dict(
            predictions=all_predictions,
            true=z_full_data,
            coords=coords_data,
            train_mask=train_mask_data,
            valid_mask=valid_mask_data,
            test_mask=test_mask_data
        )
        if config.get('regression_type') == 'multi-quantile' and config.get('save_predictions_quantile', True):
            preds_quantile = get_spatial_quantile_predictions(model, z_full_data, coords_data, device)
            if preds_quantile is not None:
                save_dict['predictions_quantile'] = preds_quantile
        np.savez(output_dir / 'predictions.npz', **save_dict)
        print(f"Predictions saved to: {output_dir / 'predictions.npz'}")

        if (qhat_per_cluster_saved is not None and cluster_centers_saved is not None
                and config.get('regression_type') == 'multi-quantile'
                and config.get('save_predictions_quantile', True)):
            n_centers = len(cluster_centers_saved)
            qhat_per_center = np.array([qhat_per_cluster_saved.get(c, conformal_qhat or 0.0) for c in range(n_centers)])
            quantile_levels_arr = np.array(config.get('quantile_levels', [0.05, 0.25, 0.5, 0.75, 0.95]))
            np.savez(
                output_dir / 'conformal_info.npz',
                qhat_per_center=qhat_per_center,
                spatial_centers=cluster_centers_saved,
                global_qhat=conformal_qhat if conformal_qhat is not None else 0.0,
                quantile_levels=quantile_levels_arr,
                conformal_alpha=float(config.get('conformal_alpha', 0.1))
            )
            plot_spatial_coverage_and_qhat(output_dir)
        
        # Extract train, test and validation predictions for quantile regression
        train_predictions_array = all_predictions[train_mask_data]
        test_predictions_array = all_predictions[test_mask_data]
        valid_predictions_array = all_predictions[valid_mask_data]
        train_true_array = z_full_data[train_mask_data]
        test_true_array = z_full_data[test_mask_data]
        valid_true_array = z_full_data[valid_mask_data]
    
    # Save basis information (initial vs final)
    print("\n" + "="*50)
    print("Saving Basis Information")
    print("="*50)
    
    basis_info = {
        'spatial_centers_init': model_initial.spatial_basis.centers.detach().cpu().numpy(),
        'spatial_centers_final': model.spatial_basis.centers.detach().cpu().numpy(),
        'spatial_bandwidths_init': model_initial.spatial_basis.bandwidths.detach().cpu().numpy(),
        'spatial_bandwidths_final': model.spatial_basis.bandwidths.detach().cpu().numpy(),
        'temporal_centers_init': model_initial.temporal_basis.centers.detach().cpu().numpy(),
        'temporal_centers_final': model.temporal_basis.centers.detach().cpu().numpy(),
        'temporal_bandwidths_init': model_initial.temporal_basis.bandwidths.detach().cpu().numpy(),
        'temporal_bandwidths_final': model.temporal_basis.bandwidths.detach().cpu().numpy()
    }
    
    np.savez(output_dir / 'basis_info.npz', **basis_info)
    print(f"Basis info saved to: {output_dir / 'basis_info.npz'}")
    
    # Print summary if learnable
    if config.get('spatial_learnable', False):
        spatial_movement = np.linalg.norm(
            basis_info['spatial_centers_final'] - basis_info['spatial_centers_init'], 
            axis=1
        )
        print(f"Spatial centers movement: mean={spatial_movement.mean():.4f}, "
              f"max={spatial_movement.max():.4f}, min={spatial_movement.min():.4f}")
    
    # Plot basis evolution
    print("\n" + "="*50)
    print("Generating Basis Evolution Visualization")
    print("="*50)
    
    # Get training coordinates for visualization
    if train_coords is None:
        # Extract from training dataset if not already available
        train_coords_list = []
        for sample in train_dataset:
            train_coords_list.append(sample['coords'].numpy())
        train_coords = np.array(train_coords_list)
    
    plot_basis_evolution(model_initial, model, train_coords, output_dir, config, basis_centers_history)
    
    # Add predictions to results for quantile regression
    results['train_predictions'] = train_predictions_array
    results['test_predictions'] = test_predictions_array
    results['valid_predictions'] = valid_predictions_array
    results['train_true'] = train_true_array
    results['test_true'] = test_true_array
    results['valid_true'] = valid_true_array
    
    # For quantile regression, use check loss instead of MSE in results
    if config.get('regression_type') == 'quantile' and 'current_quantile' in config:
        q = config['current_quantile']
        results['test_mse'] = test_metrics.get('check_loss', test_metrics['mse'])
        results['valid_mse'] = val_metrics.get('check_loss', val_metrics['mse'])
    
    print("\n" + "="*70)
    print(f"EXPERIMENT {experiment_id} COMPLETE!")
    print(f"Total Time: {results['total_time_formatted']}")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    return results


def create_averaged_spatial_mse(all_results, summary_dir):
    """
    Create averaged spatial MSE map from all experiments
    
    Args:
        all_results: list of result dictionaries from each experiment
        summary_dir: directory to save the averaged map
    """
    import matplotlib
    matplotlib.use('Agg')
    from scipy.interpolate import griddata
    
    # Load predictions from all experiments and compute MSE per experiment
    all_site_mse_list = []
    all_train_masks = []
    z_full = None
    coords = None
    
    for result in all_results:
        # Find predictions.npz file
        # Handle both mean regression (nested 'config') and quantile (flat) formats
        if 'config' in result and 'output_dir' in result['config']:
            exp_dir = Path(result['config']['output_dir'])
        elif 'output_dir' in result:
            exp_dir = Path(result['output_dir'])
        else:
            continue
            
        pred_file = exp_dir / 'predictions.npz'
        if pred_file.exists():
            data = np.load(pred_file)
            predictions = data['predictions']  # (T, S)
            true_values = data['true']  # (T, S)
            train_mask = data['train_mask']  # (T, S)
            
            # Compute MSE per site for this experiment (averaged over time)
            squared_errors = (predictions - true_values) ** 2
            site_mse = np.nanmean(squared_errors, axis=0)  # (S,)
            all_site_mse_list.append(site_mse)
            all_train_masks.append(train_mask)
            
            # Use first experiment's metadata
            if z_full is None:
                z_full = true_values
                coords = data['coords']
    
    if len(all_site_mse_list) == 0:
        print("No prediction files found. Skipping averaged spatial MSE map.")
        return
    
    # Average MSE across experiments
    all_site_mse_array = np.array(all_site_mse_list)  # (n_exp, S)
    avg_site_mse = np.nanmean(all_site_mse_array, axis=0)  # (S,)
    
    # Get all train sites (any time)
    train_sites_any = np.where(train_mask.any(axis=0))[0]
    train_coords_any = coords[train_sites_any]
    
    # Valid sites (not all NaN)
    valid_sites = ~np.isnan(avg_site_mse)
    coords_valid = coords[valid_sites]
    site_mse_valid = avg_site_mse[valid_sites]
    
    # Create grid for interpolation
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate MSE to grid using nearest neighbor
    mse_grid = griddata(coords_valid, site_mse_valid, (xi_grid, yi_grid), method='nearest')
    
    # Plot MSE map (without observation sites)
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(xi_grid, yi_grid, mse_grid, cmap='YlOrRd', shading='auto')
    
    ax.set_title(f'Averaged Spatial MSE (n={len(all_site_mse_list)} experiments)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(im, ax=ax, label='MSE')
    
    plt.tight_layout()
    save_path = summary_dir / 'spatial_mse_all.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Averaged spatial MSE plot saved to {save_path}")
    
    # Create observation density map
    create_observation_density_map(all_train_masks, coords, summary_dir)


def create_observation_density_map(all_train_masks, coords, summary_dir):
    """
    Create observation density heatmap to compare with spatial MSE
    
    Args:
        all_train_masks: list of (T, S) training masks from all experiments
        coords: (S, 2) coordinates
        summary_dir: directory to save the map
    """
    import matplotlib
    matplotlib.use('Agg')
    from scipy.interpolate import griddata
    
    n_experiments = len(all_train_masks)
    
    if n_experiments == 0:
        print("No train masks found. Skipping observation density map.")
        return
    
    # Stack all train masks: (n_exp, T, S)
    all_masks_array = np.array(all_train_masks)
    
    # Get total number of timepoints across all experiments
    T, S = all_masks_array.shape[1], all_masks_array.shape[2]
    total_possible_obs = n_experiments * T  # Max possible observations per site
    
    # Count total observations per site across all experiments and time
    total_obs_per_site = all_masks_array.sum(axis=(0, 1))  # (S,)
    
    # Compute observation ratio per site (0 to 1)
    obs_ratio_per_site = total_obs_per_site / total_possible_obs
    
    # Create grid for interpolation (same as spatial_mse_all.png)
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Interpolate observation ratio to grid using nearest neighbor
    obs_ratio_grid = griddata(coords, obs_ratio_per_site, (xi_grid, yi_grid), method='nearest')
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    im = ax.pcolormesh(xi_grid, yi_grid, obs_ratio_grid, cmap='Blues', shading='auto', vmin=0, vmax=1)
    
    ax.set_title(f'Observation Ratio per Site (n={n_experiments} experiments)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.colorbar(im, ax=ax, label='Observation Ratio')
    
    plt.tight_layout()
    save_path = summary_dir / 'observation_density.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Observation density map saved to {save_path}")


def create_averaged_spatial_coverage(all_results, summary_dir, quantile_levels=None, conformal_alpha=0.1):
    """
    Create averaged spatial coverage map (nominal + cluster-aware) from all experiments.
    Uses first experiment's spatial_centers as reference for both coverage and qhat.
    Qhat panel: averaged across experiments when centers align; else representative (exp 1).
    quantile_levels/conformal_alpha from config (fallback); prefer first exp's conformal_info when loading.
    Skips if no experiment has predictions_quantile and conformal_info.
    """
    all_cov_nominal = []
    all_cov_cluster = []
    all_qhat_per_center = []  # for averaging
    coords_ref = None
    spatial_centers_ref = None
    first_exp_dir = None
    quantile_levels_set = False
    q_lo, q_hi, idx_lo, idx_hi = 0.05, 0.95, 0, -1  # defaults

    for result in all_results:
        exp_dir = Path(result.get('config', {}).get('output_dir', result.get('output_dir', '')))
        if not exp_dir:
            continue
        pred_path = exp_dir / 'predictions.npz'
        conf_path = exp_dir / 'conformal_info.npz'
        if not pred_path.exists() or not conf_path.exists():
            continue
        preds_npz = np.load(pred_path, allow_pickle=True)
        conf_npz = np.load(conf_path, allow_pickle=True)
        if 'predictions_quantile' not in preds_npz:
            continue
        if not quantile_levels_set:
            ql = np.asarray(conf_npz['quantile_levels']) if 'quantile_levels' in conf_npz.files else quantile_levels
            ca = float(conf_npz['conformal_alpha']) if 'conformal_alpha' in conf_npz.files else conformal_alpha
            ql = np.asarray(ql) if ql is not None else np.array([0.05, 0.25, 0.5, 0.75, 0.95])
            q_lo, q_hi = ca / 2, 1.0 - ca / 2
            idx_lo = np.argmin(np.abs(ql - q_lo))
            idx_hi = np.argmin(np.abs(ql - q_hi))
            quantile_levels_set = True
        if first_exp_dir is None:
            first_exp_dir = exp_dir
        preds_q = preds_npz['predictions_quantile']
        z_full = preds_npz['true']
        coords = preds_npz['coords']
        test_mask = preds_npz['test_mask'].astype(bool)
        qhat_per_center = conf_npz['qhat_per_center']
        spatial_centers = conf_npz['spatial_centers']
        T, S, Q = preds_q.shape
        q_lo_grid = preds_q[:, :, idx_lo]
        q_hi_grid = preds_q[:, :, idx_hi]
        cluster_ids = _assign_nearest_center(coords, spatial_centers)
        qhat_per_site = qhat_per_center[cluster_ids]
        n_test_per_site = test_mask.sum(axis=0)
        inside_nominal = (z_full >= q_lo_grid) & (z_full <= q_hi_grid)
        qhat_expanded = np.broadcast_to(qhat_per_site, (T, S))
        inside_cluster = (z_full >= q_lo_grid - qhat_expanded) & (z_full <= q_hi_grid + qhat_expanded)
        cov_nominal = np.where(n_test_per_site > 0,
                               np.where(test_mask, inside_nominal, 0).sum(axis=0) / (n_test_per_site + 1e-10),
                               np.nan)
        cov_cluster = np.where(n_test_per_site > 0,
                               np.where(test_mask, inside_cluster, 0).sum(axis=0) / (n_test_per_site + 1e-10),
                               np.nan)
        all_cov_nominal.append(cov_nominal)
        all_cov_cluster.append(cov_cluster)
        if coords_ref is None:
            coords_ref = coords
            spatial_centers_ref = spatial_centers
        if (
            spatial_centers_ref is not None
            and len(spatial_centers) == len(spatial_centers_ref)
            and np.allclose(spatial_centers, spatial_centers_ref, atol=1e-6)
        ):
            all_qhat_per_center.append(qhat_per_center)

    if len(all_cov_nominal) == 0:
        print("No experiments with conformal info found. Skipping averaged spatial coverage.")
        return

    cov_nominal_avg = np.nanmean(np.array(all_cov_nominal), axis=0)
    cov_cluster_avg = np.nanmean(np.array(all_cov_cluster), axis=0)
    valid_sites = ~np.isnan(cov_nominal_avg)
    S = len(coords_ref)
    cluster_ids = _assign_nearest_center(coords_ref, spatial_centers_ref)
    if len(all_qhat_per_center) > 0:
        qhat_avg = np.mean(np.array(all_qhat_per_center), axis=0)
        qhat_title = 'Conformal qhat (per cluster, mean over exps)'
    else:
        qhat_avg = np.load(first_exp_dir / 'conformal_info.npz')['qhat_per_center']
        qhat_title = 'Conformal qhat (per cluster, representative exp 1)'
    qhat_per_site = qhat_avg[cluster_ids]
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    cov_nominal_grid = griddata(coords_ref[valid_sites], cov_nominal_avg[valid_sites], (xi_grid, yi_grid), method='nearest')
    cov_cluster_grid = griddata(coords_ref[valid_sites], cov_cluster_avg[valid_sites], (xi_grid, yi_grid), method='nearest')
    qhat_grid = griddata(coords_ref, qhat_per_site, (xi_grid, yi_grid), method='nearest')

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, data, title in [
        (axes[0], cov_nominal_grid, 'Averaged Spatial Coverage (Nominal)'),
        (axes[1], cov_cluster_grid, 'Averaged Spatial Coverage (Cluster-aware)'),
        (axes[2], qhat_grid, qhat_title),
    ]:
        vmin = 0.5 if 'Coverage' in title else 0
        vmax = 1.0 if 'Coverage' in title else np.nanmax(data) * 1.05
        im = ax.pcolormesh(xi_grid, yi_grid, data, cmap='RdYlGn' if 'Coverage' in title else 'viridis',
                           shading='auto', vmin=vmin, vmax=vmax)
        ax.scatter(spatial_centers_ref[:, 0], spatial_centers_ref[:, 1], c='red', s=15, marker='x', alpha=0.7)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect('equal')
        plt.colorbar(im, ax=ax)
    plt.tight_layout()
    save_path = summary_dir / 'spatial_coverage_aggregated.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Averaged spatial coverage map saved to {save_path}")


def aggregate_results(all_results: list, summary_dir: Path):
    """
    Aggregate results from multiple experiments and compute statistics
    
    Args:
        all_results: list of result dictionaries from each experiment
        summary_dir: directory to save summary statistics
    """
    print("\n" + "="*70)
    print("AGGREGATING RESULTS")
    print("="*70)
    
    n_experiments = len(all_results)
    
    # Extract metrics from all experiments
    metrics_data = {
        'train_mse': [],
        'train_mae': [],
        'train_rmse': [],
        'valid_mse': [],
        'valid_mae': [],
        'valid_rmse': [],
        'test_mse': [],
        'test_mae': [],
        'test_rmse': [],
        'total_time_seconds': []
    }
    
    for result in all_results:
        # Handle both quantile and mean regression formats
        if 'metrics' in result:
            # Mean regression format
            metrics_data['train_mse'].append(result['metrics']['train']['mse'])
            metrics_data['train_mae'].append(result['metrics']['train']['mae'])
            metrics_data['train_rmse'].append(result['metrics']['train']['rmse'])
            metrics_data['valid_mse'].append(result['metrics']['valid']['mse'])
            metrics_data['valid_mae'].append(result['metrics']['valid']['mae'])
            metrics_data['valid_rmse'].append(result['metrics']['valid']['rmse'])
            metrics_data['test_mse'].append(result['metrics']['test']['mse'])
            metrics_data['test_mae'].append(result['metrics']['test']['mae'])
            metrics_data['test_rmse'].append(result['metrics']['test']['rmse'])
        else:
            # Quantile regression format (direct keys)
            metrics_data['train_mse'].append(result.get('train_mse', 0))
            metrics_data['train_mae'].append(result.get('train_mae', 0))
            metrics_data['train_rmse'].append(result.get('train_rmse', 0))
            metrics_data['valid_mse'].append(result.get('valid_mse', 0))
            metrics_data['valid_mae'].append(result.get('valid_mae', 0))
            metrics_data['valid_rmse'].append(result.get('valid_rmse', 0))
            metrics_data['test_mse'].append(result.get('test_mse', 0))
            metrics_data['test_mae'].append(result.get('test_mae', 0))
            metrics_data['test_rmse'].append(result.get('test_rmse', 0))
        
        metrics_data['total_time_seconds'].append(result['total_time_seconds'])
    
    # Compute statistics
    summary = {
        'n_experiments': n_experiments,
        'statistics': {}
    }
    
    for metric_name, values in metrics_data.items():
        values_array = np.array(values)
        summary['statistics'][metric_name] = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'median': float(np.median(values_array)),
            'values': [float(v) for v in values]
        }
    
    # Save summary
    summary_file = summary_dir / 'summary_statistics.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nSummary saved to: {summary_file}")
    
    # Generate averaged spatial MSE map
    print("\n" + "="*70)
    print("GENERATING AVERAGED SPATIAL MSE MAP")
    print("="*70)
    create_averaged_spatial_mse(all_results, summary_dir)

    # Generate averaged spatial coverage map (if conformal info available)
    cfg = all_results[0].get('config', all_results[0]) if all_results else {}
    if cfg.get('regression_type') == 'multi-quantile':
        create_averaged_spatial_coverage(
            all_results, summary_dir,
            quantile_levels=cfg.get('quantile_levels', [0.05, 0.25, 0.5, 0.75, 0.95]),
            conformal_alpha=cfg.get('conformal_alpha', 0.1)
        )
    
    # Print summary table
    print("\n" + "="*70)
    print("SUMMARY STATISTICS (across {} experiments)".format(n_experiments))
    print("="*70)
    print(f"{'Metric':<20} {'Mean':<12} {'Std':<12} {'Min':<12} {'Max':<12}")
    print("-"*70)
    
    for metric_name in ['test_mse', 'test_mae', 'test_rmse', 'valid_mse', 'valid_mae', 'valid_rmse']:
        stats = summary['statistics'][metric_name]
        print(f"{metric_name:<20} {stats['mean']:<12.6f} {stats['std']:<12.6f} "
              f"{stats['min']:<12.6f} {stats['max']:<12.6f}")
    
    print("\n" + f"{'total_time (sec)':<20} {summary['statistics']['total_time_seconds']['mean']:<12.2f} "
          f"{summary['statistics']['total_time_seconds']['std']:<12.2f} "
          f"{summary['statistics']['total_time_seconds']['min']:<12.2f} "
          f"{summary['statistics']['total_time_seconds']['max']:<12.2f}")
    
    # Create detailed CSV for easy analysis
    import pandas as pd
    
    df_data = {
        'experiment_id': [r.get('experiment_id', i+1) for i, r in enumerate(all_results)],
    }
    
    # Add experiment_seed only if present in results (not available for quantile experiments)
    if all_results and 'experiment_seed' in all_results[0]:
        df_data['experiment_seed'] = [r['experiment_seed'] for r in all_results]
    
    for metric_name in metrics_data.keys():
        df_data[metric_name] = metrics_data[metric_name]
    
    df = pd.DataFrame(df_data)
    csv_file = summary_dir / 'all_experiments.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {csv_file}")
    
    return summary


def run_multiple_experiments(config, output_dir, device, parallel=False, 
                            start_exp_id=None, end_exp_id=None, skip_existing=False):
    """
    Run multiple experiments for a given configuration
    
    Args:
        config: configuration dictionary
        output_dir: output directory
        device: torch device
        parallel: whether to run in parallel mode
        start_exp_id: Starting experiment ID (1-based). If None, starts from 1
        end_exp_id: Ending experiment ID (inclusive). If None, uses n_experiments
        skip_existing: if True, skip experiments that already have results.json
    
    Returns:
        summary: aggregated summary statistics
    """
    n_experiments = config.get('n_experiments', 10)
    n_jobs = config.get('n_jobs', 10) if parallel else 1
    
    # Determine experiment range
    start_id = start_exp_id if start_exp_id is not None else 1
    end_id = end_exp_id if end_exp_id is not None else n_experiments
    
    experiments_dir = Path(output_dir) / 'experiments'
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    if parallel and n_experiments > 1:
        # Parallel execution
        from joblib import Parallel, delayed
        import warnings
        import matplotlib
        matplotlib.use('Agg')
        
        def run_experiment_wrapper(exp_id):
            """Wrapper for parallel execution"""
            import os
            import sys
            
            exp_output_dir = experiments_dir / str(exp_id)
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Suppress output
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            devnull = open(os.devnull, 'w')
            sys.stdout = devnull
            sys.stderr = devnull
            warnings.filterwarnings('ignore')
            
            try:
                result = run_single_experiment(
                    config, exp_id, exp_output_dir, device, 
                    verbose=False, parallel_mode=True, skip_existing=skip_existing
                )
                return result
            except Exception as e:
                # Write error to file instead of printing
                error_file = exp_output_dir / 'error.txt'
                with open(error_file, 'w') as f:
                    f.write(f"Experiment {exp_id} FAILED\n")
                    f.write(f"Error: {str(e)}\n\n")
                    import traceback
                    f.write(traceback.format_exc())
                return None
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                if devnull and not devnull.closed:
                    devnull.close()
        
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_experiment_wrapper)(i) 
            for i in range(start_id, end_id + 1)
        )
        all_results = [r for r in results_list if r is not None]
        
    else:
        # Sequential execution
        for i in range(start_id, end_id + 1):
            exp_output_dir = experiments_dir / str(i)
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            try:
                result = run_single_experiment(
                    config, i, exp_output_dir, device, 
                    verbose=False, parallel_mode=False, skip_existing=skip_existing
                )
                all_results.append(result)
            except Exception as e:
                print(f"Experiment {i} FAILED: {e}")
                continue
    
    # Aggregate results from ALL experiments (not just the ones we just ran)
    summary_dir = Path(output_dir) / 'summary'
    summary_dir.mkdir(parents=True, exist_ok=True)
    
    # Load all existing experiment results
    all_exp_results = []
    for i in range(1, n_experiments + 1):
        result_file = experiments_dir / str(i) / 'results.json'
        if result_file.exists():
            with open(result_file, 'r') as f:
                result = json.load(f)
                all_exp_results.append(result)
    
    if len(all_exp_results) > 0:
        summary = aggregate_results(all_exp_results, summary_dir)
        return summary
    else:
        return None


def main():
    """Main function to run multiple experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_st_interp.yaml')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--n_experiments', type=int, default=None)
    parser.add_argument('--base_seed', type=int, default=None)
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all CPUs, 0 for sequential)')
    parser.add_argument('--start_exp_id', type=int, default=None, help='Starting experiment ID (1-based)')
    parser.add_argument('--end_exp_id', type=int, default=None, help='Ending experiment ID (inclusive)')
    parser.add_argument('--skip-existing', action='store_true', help='Skip experiments that already have results.json')
    parser.add_argument('--output_dir', type=str, default=None, help='Override base output directory (e.g. results/table_4_4_<ts>/Fixed_Uniform_STDK)')
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Override config with command line arguments
    if args.data_file is not None:
        config['data_file'] = args.data_file
    if args.n_experiments is not None:
        config['n_experiments'] = args.n_experiments
    if args.base_seed is not None:
        config['base_seed'] = args.base_seed
    
    # Get experiment settings
    n_experiments = config.get('n_experiments', 1)
    parallel = args.parallel or config.get('parallel', False)
    n_jobs = args.n_jobs if args.n_jobs != -1 else config.get('n_jobs', -1)
    
    # Get experiment tag
    tag = config.get('tag', 'default')
    
    # Base output directory: override with --output_dir or use results/<date>/<time>_<tag>/
    if args.output_dir is not None:
        base_output_dir = Path(args.output_dir)
    else:
        now = datetime.now()
        date_str = now.strftime('%Y%m%d')
        time_str = now.strftime('%H%M%S')
        base_output_dir = Path('results') / date_str / f"{time_str}_{tag}"
    experiments_dir = base_output_dir / 'experiments'
    experiments_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to base directory
    with open(base_output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("\n" + "="*70)
    print("MULTIPLE EXPERIMENT RUNNER")
    print("="*70)
    print(f"Experiment tag: {tag}")
    print(f"Number of experiments: {n_experiments}")
    print(f"Base output directory: {base_output_dir}")
    print(f"Base seed: {config.get('base_seed', 42)}")
    print(f"Parallel execution: {parallel}")
    if parallel:
        if n_jobs == -1:
            import multiprocessing
            n_jobs = multiprocessing.cpu_count()
        print(f"Number of parallel jobs: {n_jobs}")
    print("="*70)
    
    # Device
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Determine experiment range
    start_id = args.start_exp_id if args.start_exp_id is not None else 1
    end_id = args.end_exp_id if args.end_exp_id is not None else n_experiments
    skip_existing = args.skip_existing
    
    # Run experiments
    all_results = []
    
    if parallel and n_experiments > 1:
        # Parallel execution using joblib
        from joblib import Parallel, delayed
        import warnings
        
        # Set matplotlib backend globally before parallel execution
        matplotlib.use('Agg')
        
        def run_experiment_wrapper(exp_id):
            """Wrapper for parallel execution with suppressed output"""
            import os
            import sys
            
            exp_output_dir = experiments_dir / str(exp_id)
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Suppress all output during parallel execution
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Redirect to devnull
            devnull = open(os.devnull, 'w')
            sys.stdout = devnull
            sys.stderr = devnull
            
            # Suppress warnings
            warnings.filterwarnings('ignore')
            
            try:
                result = run_single_experiment(config, exp_id, exp_output_dir, device, 
                                              verbose=False, parallel_mode=True, skip_existing=skip_existing)
                return result
            except Exception as e:
                # Restore output to show errors
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                devnull.close()
                print(f"\n[ERROR] Experiment {exp_id} FAILED: {str(e)[:100]}")
                import traceback
                traceback.print_exc()
                return None
            finally:
                # Restore output
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                devnull.close()
        
        print(f"Running experiments {start_id} to {end_id} in parallel with {n_jobs} jobs...")
        print("(Detailed logs suppressed during parallel execution)\n")
        
        # Run in parallel with progress bar
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_experiment_wrapper)(i) for i in range(start_id, end_id + 1)
        )
        
        # Filter out None (failed experiments)
        all_results = [r for r in results_list if r is not None]
        
        print(f"\n✅ Completed {len(all_results)}/{end_id - start_id + 1} experiments successfully")
        
    else:
        # Sequential execution with full logging
        print(f"Running experiments {start_id} to {end_id} sequentially...\n")
        
        for i in range(start_id, end_id + 1):
            # Create experiment-specific output directory
            exp_output_dir = experiments_dir / str(i)
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run experiment with full logging
            try:
                results = run_single_experiment(config, i, exp_output_dir, device, 
                                              verbose=True, parallel_mode=False, skip_existing=skip_existing)
                all_results.append(results)
            except Exception as e:
                print(f"\n[ERROR] Experiment {i} FAILED with error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Aggregate results from ALL experiments (not just the ones we just ran)
    if n_experiments > 1:
        summary_dir = base_output_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        
        # Load all existing experiment results
        all_exp_results = []
        for i in range(1, n_experiments + 1):
            result_file = experiments_dir / str(i) / 'results.json'
            if result_file.exists():
                with open(result_file, 'r') as f:
                    import json
                    result = json.load(f)
                    all_exp_results.append(result)
        
        if len(all_exp_results) > 0:
            return aggregate_results(all_exp_results, summary_dir)
    
    return None
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Successful experiments: {len(all_results)} / {n_experiments}")
    print(f"Results directory: {base_output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
