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
from stnf.dataio.kaust_loader import load_kaust_csv_single
from stnf.utils import set_seed, compute_metrics


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
        # Heavy-tailed distribution with sharp peak at (0,0)
        # p(x,y) ∝ 1 / (1 + intensity * ||[x,y]||²)^2
        # This is a Cauchy-like distribution with heavier tails than Gaussian
        def obs_prob_fn(coord):
            x, y = coord
            dist_sq = x**2 + y**2
            # Power law decay: sharper peak, longer tail
            prob = 1.0 / (1.0 + intensity * dist_sq)**2
            return prob
        return obs_prob_fn
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def sample_observations(z_data, coords, obs_method='site-wise', obs_ratio=0.5, 
                       obs_prob_fn=None, seed=None):
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
        # Apply function to each coordinate
        obs_probs = np.array([obs_prob_fn(coords[i]) for i in range(S)])
    else:
        # Uniform probability
        obs_probs = np.ones(S) * obs_ratio
    
    if obs_method == 'site-wise':
        # Select obs_ratio fraction of sites, observe all times
        # Use obs_probs as sampling weights
        n_obs_sites = int(S * obs_ratio)
        obs_probs_normalized = obs_probs / obs_probs.sum()
        obs_sites = np.random.choice(S, size=n_obs_sites, replace=False, p=obs_probs_normalized)
        
        obs_mask = np.zeros((T, S), dtype=bool)
        obs_mask[:, obs_sites] = True
        
        return obs_mask, obs_sites
    
    elif obs_method == 'random':
        # Randomly observe each (time, site) pair
        # Each site has probability obs_probs[s]
        obs_probs_expanded = obs_probs[np.newaxis, :].repeat(T, axis=0)
        obs_mask = np.random.rand(T, S) < obs_probs_expanded
        
        # Get list of sites that have at least one observation
        obs_sites = np.where(obs_mask.any(axis=0))[0]
        
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
    
    if config.get('spatial_learnable', False):
        # Learnable basis: use differential learning rates
        basis_params = list(model.spatial_basis.parameters())
        mlp_params = [p for p in model.parameters() if not any(p is bp for bp in basis_params)]
        
        lr = float(config.get('lr', 1e-3))
        optimizer = optim.AdamW([
            {'params': mlp_params, 'lr': lr},
            {'params': basis_params, 'lr': lr * 0.5}  # 2x smaller LR
        ], weight_decay=float(config.get('weight_decay', 1e-5)))

        print(f"Spatial basis: LEARNABLE (MLP lr={lr:.2e}, Basis lr={lr*0.5:.2e})")
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
        
        print(f"Spatial basis: FIXED (only MLP trained, lr={lr:.2e})")
    
    # Scheduler
    scheduler = None
    if config.get('scheduler') == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.get('epochs', 100)
        )
    
    criterion = nn.MSELoss()
    epochs = config.get('epochs', 100)
    best_val_loss = float('inf')
    patience = config.get('patience', 15)
    patience_counter = 0
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'val_rmse': []
    }
    
    best_model_path = output_dir / 'model_best.pt'
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            X = batch['X'].to(device)
            coords = batch['coords'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            y_pred = model(X, coords, t)
            
            # Loss
            loss = criterion(y_pred, y)
            loss.backward()
            
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item()
            
            # Check for NaN
            if np.isnan(loss.item()):
                print(f"\n⚠️ NaN detected at batch {len(train_loader)}!")
                print(f"  y_pred stats: min={y_pred.min().item():.3f}, max={y_pred.max().item():.3f}, "
                      f"mean={y_pred.mean().item():.3f}, std={y_pred.std().item():.3f}")
                print(f"  y_true stats: min={y.min().item():.3f}, max={y.max().item():.3f}")
                if config.get('spatial_learnable', False):
                    bw = model.spatial_basis.bandwidths
                    print(f"  Bandwidths: min={bw.min().item():.4f}, max={bw.max().item():.4f}, "
                          f"mean={bw.mean().item():.4f}")
                break
        
        train_loss /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_trues = []
        
        with torch.no_grad():
            for batch in val_loader:
                X = batch['X'].to(device)
                coords = batch['coords'].to(device)
                t = batch['t'].to(device)
                y = batch['y'].to(device)
                
                y_pred = model(X, coords, t)
                
                loss = criterion(y_pred, y)
                val_loss += loss.item()
                
                val_preds.append(y_pred.cpu().numpy())
                val_trues.append(y.cpu().numpy())
        
        val_loss /= len(val_loader)
        
        # Compute RMSE
        val_preds = np.concatenate(val_preds, axis=0)
        val_trues = np.concatenate(val_trues, axis=0)
        val_rmse = np.sqrt(np.mean((val_preds - val_trues) ** 2))
        
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_rmse'].append(val_rmse)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f}, Val RMSE = {val_rmse:.6f}")
        
        # Learning rate scheduling
        if scheduler is not None:
            scheduler.step()
            print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Save best model
        if not np.isnan(val_loss) and val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"  → Best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model if it exists
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        print(f"\nTraining Complete! Best Val Loss: {best_val_loss:.6f}")
    else:
        print(f"\n⚠️ Training failed - no valid model saved!")
        print(f"Try: 1) Lower learning rate, 2) Set spatial_learnable=false, 3) Increase grad_clip")
    
    return model, history


def evaluate_model(model, data_loader, device):
    """
    Evaluate model on a dataset
    
    Returns:
        metrics: dict with 'mse' and 'mae'
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
    
    preds = np.concatenate(all_preds, axis=0)
    trues = np.concatenate(all_trues, axis=0)
    
    mse = np.mean((preds - trues) ** 2)
    mae = np.mean(np.abs(preds - trues))
    
    return {
        'mse': float(mse),
        'mae': float(mae),
        'rmse': float(np.sqrt(mse))
    }


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
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax = axes[0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1]
    ax.plot(epochs, history['val_rmse'], 'g-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('RMSE')
    ax.set_title('Validation RMSE')
    ax.grid(True, alpha=0.3)
    
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
            
            y_pred = model(X_tensor, coords_tensor, t_tensor).cpu().numpy().flatten()
            predictions[t_idx] = y_pred
    
    # Create grid for interpolation (higher resolution for smoother heatmap)
    grid_resolution = 200
    xi = np.linspace(0, 1, grid_resolution)
    yi = np.linspace(0, 1, grid_resolution)
    xi_grid, yi_grid = np.meshgrid(xi, yi)
    
    # Create plots
    fig, axes = plt.subplots(n_times, 3, figsize=(15, 5 * n_times))
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
        ax.set_title(f't={t_idx+1} - True')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax)
        
        # Predicted values
        ax = axes[i, 1]
        im = ax.pcolormesh(xi_grid, yi_grid, y_pred_grid, cmap='viridis', shading='auto')
        ax.scatter(train_coords_t[:, 0], train_coords_t[:, 1], 
                  c='black', s=20, alpha=0.6, label='Train sites', edgecolors='white', linewidths=0.5)
        ax.scatter(spatial_centers[:, 0], spatial_centers[:, 1], 
                  c='red', s=basis_sizes, marker='x', alpha=0.5, label='Basis centers', linewidths=1.5)
        ax.set_title(f't={t_idx+1} - Predicted')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
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
        ax.set_title(f't={t_idx+1} - Bias (Pred - True)')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(loc='upper right', fontsize=8)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    save_path = output_dir / 'prediction_maps.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Prediction maps saved to {save_path}")



def plot_spatial_mse(model, z_full, coords, train_mask, device, output_dir):
    """
    Plot spatial MSE heatmap averaged over all time points
    
    Args:
        model: trained model
        z_full: (T, S) full data
        coords: (S, 2) coordinates
        train_mask: (T, S) training mask
        device: torch device
        output_dir: output directory
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
            
            y_pred = model(X_tensor, coords_tensor, t_tensor).cpu().numpy().flatten()
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
    
    ax.set_title('Spatial MSE (Averaged over Time)')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper right', fontsize=10)
    plt.colorbar(im, ax=ax, label='MSE')
    
    plt.tight_layout()
    save_path = output_dir / 'spatial_mse.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Spatial MSE plot saved to {save_path}")


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
    ax.set_title(f'Total Observations per Site\n(Total: {obs_mask.sum()} obs)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter, ax=ax, label='# observations')
    
    # Train observations
    ax = axes[0, 1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=train_counts, cmap='Blues', s=point_size, alpha=0.7)
    ax.set_title(f'Train Observations per Site\n(Total: {train_mask.sum()} obs)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter, ax=ax, label='# observations')
    
    # Valid observations
    ax = axes[1, 0]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=valid_counts, cmap='Greens', s=point_size, alpha=0.7)
    ax.set_title(f'Valid Observations per Site\n(Total: {valid_mask.sum()} obs)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter, ax=ax, label='# observations')
    
    # Test (unobserved) count
    ax = axes[1, 1]
    scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                        c=test_counts, cmap='Reds', s=point_size, alpha=0.7)
    ax.set_title(f'Test (Unobserved) per Site\n(Total: {(~obs_mask).sum()} obs)', fontsize=12)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    plt.colorbar(scatter, ax=ax, label='# unobserved')
    
    plt.tight_layout()
    save_path = output_dir / 'observation_pattern.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Observation pattern plot saved to {save_path}")


def run_single_experiment(config: dict, experiment_id: int, output_dir: Path, device: str, verbose: bool = True):
    """
    Run a single experiment
    
    Args:
        config: configuration dictionary
        experiment_id: experiment ID (1, 2, ..., M)
        output_dir: output directory for this experiment
        device: torch device
        verbose: whether to print detailed logs
    
    Returns:
        results: dictionary containing all results
    """
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
    
    # Load data
    if verbose:
        print("\nLoading data...")
    z_full, coords, metadata = load_kaust_csv_single(
        config.get('data_file', 'data/2b/2b_7.csv'),
        normalize=config.get('normalize_target', False)
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
        seed=config.get('seed', 42)
    )
    
    n_obs_total = obs_mask.sum()
    print(f"Observation method: {obs_method}")
    if obs_spatial_pattern != 'uniform':
        print(f"Spatial pattern: {obs_spatial_pattern} (intensity={obs_spatial_intensity})")
    print(f"Observed: {n_obs_total} / {z_full.size} ({n_obs_total/z_full.size*100:.1f}%)")
    print(f"Observed sites: {len(obs_sites)} / {coords.shape[0]}")
    
    # Split train and validation (from observed data)
    print("\nSplitting train/valid...")
    split_method = config.get('split_method', 'site-wise')
    train_ratio = config.get('train_ratio', 0.8)
    
    train_mask, valid_mask = split_train_valid(
        obs_mask, obs_sites,
        split_method=split_method,
        train_ratio=train_ratio,
        seed=config.get('seed', 42) + 1
    )
    
    print(f"Split method: {split_method}")
    print(f"Train: {train_mask.sum()} samples")
    print(f"Valid: {valid_mask.sum()} samples")
    print(f"Actual train ratio: {train_mask.sum() / (train_mask.sum() + valid_mask.sum()):.3f}")
    
    # Test set: all non-observed data
    test_mask = ~obs_mask
    print(f"Test: {test_mask.sum()} samples (all unobserved data)")
    
    # Create datasets
    print("\nCreating datasets...")
    p_covariates = config.get('p_covariates', 0)
    
    train_dataset = create_dataset_from_mask(z_full, coords, train_mask, p_covariates)
    val_dataset = create_dataset_from_mask(z_full, coords, valid_mask, p_covariates)
    test_dataset = create_dataset_from_mask(z_full, coords, test_mask, p_covariates)
    
    print(f"Train dataset: {len(train_dataset)} samples")
    print(f"Val dataset: {len(val_dataset)} samples")
    print(f"Test dataset: {len(test_dataset)} samples")
    
    # Create dataloaders
    from torch.utils.data import DataLoader
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.get('batch_size', 256),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )
    
    # Create model
    print("\nCreating model...")
    
    # Prepare training coordinates for GMM initialization if needed
    train_coords = None
    if config.get('spatial_init_method', 'uniform') == 'gmm':
        # Extract unique coordinates from training data
        train_coords_list = []
        for sample in train_dataset:
            train_coords_list.append(sample['coords'].numpy())
        train_coords = np.array(train_coords_list)  # (N_train, 2)
        print(f"Using GMM initialization with {len(train_coords)} training coordinates")
    
    model = create_model(config, train_coords=train_coords)
    model = model.to(device)
    
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
    model, history = train_model(model, train_loader, val_loader, config, device, output_dir)
    
    # Evaluate on all sets
    print("\n" + "="*50)
    print("Evaluating Model")
    print("="*50)
    
    print("\nEvaluating on Train set...")
    train_metrics = evaluate_model(model, train_loader, device)
    print(f"Train - MSE: {train_metrics['mse']:.6f}, MAE: {train_metrics['mae']:.6f}, RMSE: {train_metrics['rmse']:.6f}")
    
    print("\nEvaluating on Valid set...")
    val_metrics = evaluate_model(model, val_loader, device)
    print(f"Valid - MSE: {val_metrics['mse']:.6f}, MAE: {val_metrics['mae']:.6f}, RMSE: {val_metrics['rmse']:.6f}")
    
    print("\nEvaluating on Test set...")
    test_metrics = evaluate_model(model, test_loader, device)
    print(f"Test  - MSE: {test_metrics['mse']:.6f}, MAE: {test_metrics['mae']:.6f}, RMSE: {test_metrics['rmse']:.6f}")
    
    # Calculate total time
    total_time = time.time() - start_time
    
    # Save results
    results = {
        'experiment_id': experiment_id,
        'experiment_seed': experiment_seed,
        'config': config,
        'metrics': {
            'train': train_metrics,
            'valid': val_metrics,
            'test': test_metrics
        },
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
    plot_spatial_mse(model, z_full, coords, train_mask, device, output_dir)
    
    print("\n" + "="*70)
    print(f"EXPERIMENT {experiment_id} COMPLETE!")
    print(f"Total Time: {results['total_time_formatted']}")
    print(f"Results saved to: {output_dir}")
    print("="*70)
    
    return results


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
        metrics_data['train_mse'].append(result['metrics']['train']['mse'])
        metrics_data['train_mae'].append(result['metrics']['train']['mae'])
        metrics_data['train_rmse'].append(result['metrics']['train']['rmse'])
        metrics_data['valid_mse'].append(result['metrics']['valid']['mse'])
        metrics_data['valid_mae'].append(result['metrics']['valid']['mae'])
        metrics_data['valid_rmse'].append(result['metrics']['valid']['rmse'])
        metrics_data['test_mse'].append(result['metrics']['test']['mse'])
        metrics_data['test_mae'].append(result['metrics']['test']['mae'])
        metrics_data['test_rmse'].append(result['metrics']['test']['rmse'])
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
        'experiment_id': [r['experiment_id'] for r in all_results],
        'experiment_seed': [r['experiment_seed'] for r in all_results],
    }
    
    for metric_name in metrics_data.keys():
        df_data[metric_name] = metrics_data[metric_name]
    
    df = pd.DataFrame(df_data)
    csv_file = summary_dir / 'all_experiments.csv'
    df.to_csv(csv_file, index=False)
    print(f"\nDetailed results saved to: {csv_file}")
    
    return summary


def main():
    """Main function to run multiple experiments"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config_st_interp.yaml')
    parser.add_argument('--data_file', type=str, default=None)
    parser.add_argument('--n_experiments', type=int, default=None)
    parser.add_argument('--base_seed', type=int, default=None)
    parser.add_argument('--parallel', action='store_true', help='Run experiments in parallel')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Number of parallel jobs (-1 for all CPUs, 0 for sequential)')
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
    
    # Create base output directory with date/time
    now = datetime.now()
    date_str = now.strftime('%Y%m%d')
    time_str = now.strftime('%H%M%S')
    base_output_dir = Path('results') / date_str / time_str
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config to base directory
    with open(base_output_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    print("\n" + "="*70)
    print("MULTIPLE EXPERIMENT RUNNER")
    print("="*70)
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
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}\n")
    
    # Run experiments
    all_results = []
    
    if parallel and n_experiments > 1:
        # Parallel execution using joblib
        from joblib import Parallel, delayed
        import warnings
        import sys
        import io
        
        def run_experiment_wrapper(exp_id):
            """Wrapper for parallel execution with suppressed output"""
            exp_output_dir = base_output_dir / str(exp_id)
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Suppress all output during parallel execution
            import os
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            
            # Redirect to devnull
            devnull = open(os.devnull, 'w')
            sys.stdout = devnull
            sys.stderr = devnull
            
            # Suppress matplotlib
            matplotlib.use('Agg')
            
            # Suppress warnings
            warnings.filterwarnings('ignore')
            
            try:
                result = run_single_experiment(config, exp_id, exp_output_dir, device, verbose=False)
                return result
            except Exception as e:
                # Restore output to show errors
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                devnull.close()
                print(f"\n❌ Experiment {exp_id} FAILED: {str(e)[:100]}")
                return None
            finally:
                # Restore output
                sys.stdout = old_stdout
                sys.stderr = old_stderr
                devnull.close()
        
        print(f"Running {n_experiments} experiments in parallel with {n_jobs} jobs...")
        print("(Detailed logs suppressed during parallel execution)\n")
        
        # Run in parallel with progress bar
        results_list = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_experiment_wrapper)(i) for i in range(1, n_experiments + 1)
        )
        
        # Filter out None (failed experiments)
        all_results = [r for r in results_list if r is not None]
        
        print(f"\n✅ Completed {len(all_results)}/{n_experiments} experiments successfully")
        
    else:
        # Sequential execution with full logging
        print(f"Running {n_experiments} experiments sequentially...\n")
        
        for i in range(1, n_experiments + 1):
            # Create experiment-specific output directory
            exp_output_dir = base_output_dir / str(i)
            exp_output_dir.mkdir(parents=True, exist_ok=True)
            
            # Run experiment with full logging
            try:
                results = run_single_experiment(config, i, exp_output_dir, device, verbose=True)
                all_results.append(results)
            except Exception as e:
                print(f"\n❌ Experiment {i} FAILED with error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Aggregate results if multiple experiments
    if n_experiments > 1 and len(all_results) > 0:
        summary_dir = base_output_dir / 'summary'
        summary_dir.mkdir(parents=True, exist_ok=True)
        aggregate_results(all_results, summary_dir)
    
    print("\n" + "="*70)
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"Successful experiments: {len(all_results)} / {n_experiments}")
    print(f"Results directory: {base_output_dir}")
    print("="*70)


if __name__ == '__main__':
    main()
