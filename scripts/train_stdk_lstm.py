"""
2-Stage Deep Kriging + LSTM 학습 스크립트

Stage 1: Deep Kriging으로 spatial interpolation 학습
Stage 2: LSTM으로 temporal forecasting 학습

Usage:
    python scripts/train_stdk_lstm.py --config configs/config.yaml
"""
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from stnf.models.stdk_lstm import STDKLSTM, create_model
from stnf.dataio import load_kaust_csv
from stnf.utils import set_seed, compute_metrics


def train_stage1(model, train_data, val_data, config, device):
    """
    Stage 1: Deep Kriging 학습
    목표: (X, phi_s, phi_t) → z 매핑 학습
    """
    print("\n" + "="*50)
    print("Stage 1: Training Deep Kriging")
    print("="*50)
    
    optimizer = optim.AdamW(
        model.kriging.parameters(),
        lr=config.get('kriging_lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    criterion = nn.MSELoss()
    epochs = config.get('kriging_epochs', 50)
    best_val_loss = float('inf')
    patience = config.get('patience', 10)
    patience_counter = 0
    
    for epoch in range(epochs):
        # Training
        model.kriging.train()
        train_loss = 0.0
        
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            y_hist_obs = batch['y_hist_obs'].to(device)  # (B, L, n_obs, 1)
            obs_coords = batch['obs_coords'].to(device)  # (B, n_obs, 2) or (n_obs, 2)
            
            B, L, n_obs, _ = y_hist_obs.shape
            
            # Get covariates if available
            if 'X_hist_obs' in batch:
                X_hist = batch['X_hist_obs'].to(device)  # (B, L, n_obs, p)
                p = X_hist.shape[-1]
            else:
                X_hist = torch.zeros(B, L, n_obs, 0, device=device)
                p = 0
            
            # Flatten batch and time dimensions for training
            X_flat = X_hist.view(B * L, n_obs, p)
            y_flat = y_hist_obs.view(B * L, n_obs, 1)
            
            # Time indices (normalized to [0, 1])
            t_flat = torch.linspace(0, 1, L, device=device).repeat(B).unsqueeze(-1).unsqueeze(-1)  # (B*L, 1, 1)
            t_flat = t_flat.expand(B * L, n_obs, 1)
            
            # Coordinates - handle both (B, n_obs, 2) and (n_obs, 2)
            if obs_coords.dim() == 2:
                obs_coords = obs_coords.unsqueeze(0).expand(B, -1, -1)  # (B, n_obs, 2)
            coords_flat = obs_coords.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, n_obs, 2)
            
            optimizer.zero_grad()
            
            # Kriging prediction
            z_pred = model.kriging(X_flat, coords_flat, t_flat)  # (B*L, n_obs, 1)
            
            # Loss
            loss = criterion(z_pred, y_flat)
            loss.backward()
            
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.kriging.parameters(), config['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_data)
        
        # Validation
        model.kriging.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch in val_data:
                y_hist_obs = batch['y_hist_obs'].to(device)
                obs_coords = batch['obs_coords'].to(device)
                
                B, L, n_obs, _ = y_hist_obs.shape
                
                if 'X_hist_obs' in batch:
                    X_hist = batch['X_hist_obs'].to(device)
                    p = X_hist.shape[-1]
                else:
                    X_hist = torch.zeros(B, L, n_obs, 0, device=device)
                    p = 0
                
                X_flat = X_hist.view(B * L, n_obs, p)
                y_flat = y_hist_obs.view(B * L, n_obs, 1)
                
                t_flat = torch.linspace(0, 1, L, device=device).repeat(B).unsqueeze(-1).unsqueeze(-1)
                t_flat = t_flat.expand(B * L, n_obs, 1)
                
                if obs_coords.dim() == 2:
                    obs_coords = obs_coords.unsqueeze(0).expand(B, -1, -1)
                coords_flat = obs_coords.unsqueeze(1).expand(-1, L, -1, -1).reshape(B * L, n_obs, 2)
                
                z_pred = model.kriging(X_flat, coords_flat, t_flat)
                loss = criterion(z_pred, y_flat)
                val_loss += loss.item()
        
        val_loss /= len(val_data)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, Val Loss = {val_loss:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.kriging.state_dict(), 'outputs/stdk_kriging_best.pt')
            print(f"  → Best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    model.kriging.load_state_dict(torch.load('outputs/stdk_kriging_best.pt'))
    print(f"\nStage 1 Complete! Best Val Loss: {best_val_loss:.6f}")
    
    return model


def train_stage2(model, train_data, val_data, config, device):
    """
    Stage 2: LSTM Forecasting 학습
    목표: 각 site별 시계열로 H-step ahead forecasting
    """
    print("\n" + "="*50)
    print("Stage 2: Training LSTM Forecaster")
    print("="*50)
    
    # Freeze kriging model
    for param in model.kriging.parameters():
        param.requires_grad = False
    
    optimizer = optim.AdamW(
        model.forecaster.parameters(),
        lr=config.get('lstm_lr', 1e-3),
        weight_decay=config.get('weight_decay', 1e-5)
    )
    
    criterion = nn.MSELoss()
    epochs = config.get('lstm_epochs', 100)
    best_val_loss = float('inf')
    patience = config.get('patience', 10)
    patience_counter = 0
    H = config.get('forecast_horizon', 10)
    
    for epoch in range(epochs):
        # Training
        model.train()
        model.kriging.eval()  # Keep kriging in eval mode
        train_loss = 0.0
        
        for batch in tqdm(train_data, desc=f"Epoch {epoch+1}/{epochs}"):
            y_hist_obs = batch['y_hist_obs'].to(device)  # (B, L, n_obs, 1)
            y_fut = batch['y_fut'].to(device)  # (B, H, n_obs, 1)
            obs_coords = batch['obs_coords'].to(device)  # (n_obs, 2) or (B, n_obs, 2)
            target_coords = batch['target_coords'].to(device)  # Same as obs_coords
            
            B, L, n_obs, _ = y_hist_obs.shape
            
            # Get covariates if available
            if 'X_hist_obs' in batch:
                X_hist = batch['X_hist_obs'].to(device)
                p = X_hist.shape[-1]
            else:
                X_hist = torch.zeros(B, L, n_obs, 0, device=device)
                p = 0
            
            # Handle coordinate dimensions
            if obs_coords.dim() == 2:
                obs_coords = obs_coords.unsqueeze(0).expand(B, -1, -1)
            if target_coords.dim() == 2:
                target_coords = target_coords.unsqueeze(0).expand(B, -1, -1)
            
            # Time for history
            t_hist = torch.linspace(0, 1, L, device=device)
            
            # For this dataset, obs and target are same
            # Combine observed and target coords (in this case, same)
            coords_all = obs_coords[0]  # (n_obs, 2)
            
            optimizer.zero_grad()
            
            # Stage 1: Interpolate all sites
            with torch.no_grad():
                z_all = model.interpolate_sites(
                    X_hist, obs_coords[0], y_hist_obs, coords_all, t_hist
                )  # (B, L, n_all, 1)
            
            # Stage 2: Forecast with LSTM
            y_pred = model.forecaster(z_all, H)  # (B, H, n_all, 1)
            
            # Loss (predict same obs sites in future)
            loss = criterion(y_pred, y_fut)
            loss.backward()
            
            if config.get('grad_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(model.forecaster.parameters(), config['grad_clip'])
            
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_data)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_predictions = []
        val_targets = []
        
        with torch.no_grad():
            for batch in val_data:
                y_hist_obs = batch['y_hist_obs'].to(device)
                y_fut = batch['y_fut'].to(device)
                obs_coords = batch['obs_coords'].to(device)
                target_coords = batch['target_coords'].to(device)
                
                B, L, n_obs, _ = y_hist_obs.shape
                
                if 'X_hist_obs' in batch:
                    X_hist = batch['X_hist_obs'].to(device)
                else:
                    X_hist = torch.zeros(B, L, n_obs, 0, device=device)
                
                if obs_coords.dim() == 2:
                    obs_coords = obs_coords.unsqueeze(0).expand(B, -1, -1)
                if target_coords.dim() == 2:
                    target_coords = target_coords.unsqueeze(0).expand(B, -1, -1)
                
                t_hist = torch.linspace(0, 1, L, device=device)
                coords_all = obs_coords[0]
                
                z_all = model.interpolate_sites(
                    X_hist, obs_coords[0], y_hist_obs, coords_all, t_hist
                )
                y_pred = model.forecaster(z_all, H)
                
                loss = criterion(y_pred, y_fut)
                val_loss += loss.item()
                
                val_predictions.append(y_pred.cpu())
                val_targets.append(y_fut.cpu())
        
        val_loss /= len(val_data)
        
        # Compute metrics
        val_preds = torch.cat(val_predictions, dim=0).numpy()
        val_tgts = torch.cat(val_targets, dim=0).numpy()
        metrics = compute_metrics(val_preds, val_tgts)
        
        print(f"Epoch {epoch+1}: Train Loss = {train_loss:.6f}, "
              f"Val Loss = {val_loss:.6f}, RMSE = {metrics['rmse']:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save(model.state_dict(), 'outputs/stdk_lstm_best.pt')
            print(f"  → Best model saved (val_loss: {val_loss:.6f})")
        else:
            patience_counter += 1
            print(f"  → No improvement ({patience_counter}/{patience})")
            
            if patience_counter >= patience:
                print(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # Load best model
    model.load_state_dict(torch.load('outputs/stdk_lstm_best.pt'))
    print(f"\nStage 2 Complete! Best Val Loss: {best_val_loss:.6f}")
    
    return model


def visualize_predictions(model, val_loader, coords, obs_indices, config, device, output_dir='outputs'):
    """
    학습 완료 후 예측 결과 시각화
    """
    print("\n" + "="*50)
    print("Visualizing Predictions")
    print("="*50)
    
    matplotlib.use('Agg')  # Non-interactive backend
    Path(output_dir).mkdir(exist_ok=True)
    
    model.eval()
    H = config.get('forecast_horizon', 10)
    
    # Use only observed site coordinates
    obs_coords = coords[obs_indices]
    
    # Collect predictions from validation set
    all_preds = []
    all_trues = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(val_loader, desc="Collecting predictions")):
            if batch_idx >= 5:  # Limit to 5 batches for visualization
                break
            
            y_hist_obs = batch['y_hist_obs'].to(device)
            y_fut = batch['y_fut'].to(device)
            batch_obs_coords = batch['obs_coords'].to(device)
            
            B, L, n_obs, _ = y_hist_obs.shape
            
            if 'X_hist_obs' in batch:
                X_hist = batch['X_hist_obs'].to(device)
            else:
                X_hist = torch.zeros(B, L, n_obs, 0, device=device)
            
            if batch_obs_coords.dim() == 2:
                batch_obs_coords = batch_obs_coords.unsqueeze(0).expand(B, -1, -1)
            
            t_hist = torch.linspace(0, 1, L, device=device)
            coords_all = batch_obs_coords[0]
            
            # Interpolate and forecast
            z_all = model.interpolate_sites(
                X_hist, batch_obs_coords[0], y_hist_obs, coords_all, t_hist
            )
            y_pred = model.forecaster(z_all, H)
            
            all_preds.append(y_pred.cpu().numpy())
            all_trues.append(y_fut.cpu().numpy())
    
    if len(all_preds) == 0:
        print("No predictions collected!")
        return
    
    preds = np.concatenate(all_preds, axis=0)  # (N, H, n_sites, 1)
    trues = np.concatenate(all_trues, axis=0)
    
    # 1. Spatial maps at different horizons
    plot_spatial_maps(preds, trues, obs_coords, H, f"{output_dir}/stdk_spatial_maps.png")
    
    # 2. Time series for selected sites
    plot_timeseries(preds, trues, H, f"{output_dir}/stdk_timeseries.png")
    
    # 3. Horizon-wise error analysis
    plot_horizon_errors(preds, trues, H, f"{output_dir}/stdk_horizon_errors.png")
    
    print(f"Visualizations saved to {output_dir}/")


def plot_spatial_maps(preds, trues, coords, H, save_path, n_horizons=5):
    """시간 horizon별 공간 분포 시각화"""
    # Select horizons to visualize
    if H > n_horizons:
        h_indices = np.linspace(0, H-1, n_horizons, dtype=int)
    else:
        h_indices = np.arange(H)
        n_horizons = H
    
    # Use first sample
    pred_sample = preds[0].squeeze(-1)  # (H, n_sites)
    true_sample = trues[0].squeeze(-1)
    
    fig, axes = plt.subplots(2, n_horizons, figsize=(4*n_horizons, 8))
    if n_horizons == 1:
        axes = axes.reshape(2, 1)
    
    vmin = min(true_sample.min(), pred_sample.min())
    vmax = max(true_sample.max(), pred_sample.max())
    
    for i, h in enumerate(h_indices):
        # True values
        ax = axes[0, i]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=true_sample[h], 
                           cmap='RdYlBu_r', s=20, vmin=vmin, vmax=vmax)
        ax.set_title(f'True (h={h+1})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
        
        # Predictions
        ax = axes[1, i]
        scatter = ax.scatter(coords[:, 0], coords[:, 1], c=pred_sample[h], 
                           cmap='RdYlBu_r', s=20, vmin=vmin, vmax=vmax)
        ax.set_title(f'Pred (h={h+1})')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_aspect('equal')
        plt.colorbar(scatter, ax=ax)
    
    fig.suptitle('Spatial Distribution at Different Horizons', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_timeseries(preds, trues, H, save_path, n_sites=10):
    """사이트별 시계열 비교"""
    N, H_dim, n_sites_total, _ = preds.shape
    
    # Randomly select sites
    if n_sites_total > n_sites:
        site_indices = np.random.choice(n_sites_total, n_sites, replace=False)
    else:
        site_indices = np.arange(n_sites_total)
        n_sites = n_sites_total
    
    # Use first sample
    pred_sample = preds[0].squeeze(-1)  # (H, n_sites)
    true_sample = trues[0].squeeze(-1)
    
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.flatten()
    
    for i, site_idx in enumerate(site_indices[:10]):
        ax = axes[i]
        horizons = np.arange(1, H+1)
        ax.plot(horizons, true_sample[:, site_idx], 'b-o', label='True', markersize=4)
        ax.plot(horizons, pred_sample[:, site_idx], 'r--s', label='Pred', markersize=4)
        ax.set_xlabel('Horizon')
        ax.set_ylabel('Value')
        ax.set_title(f'Site {site_idx}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Time Series Forecasts for Selected Sites', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def plot_horizon_errors(preds, trues, H, save_path):
    """Horizon별 에러 분석"""
    preds_flat = preds.squeeze(-1)  # (N, H, n_sites)
    trues_flat = trues.squeeze(-1)
    
    # Compute metrics per horizon
    mae_per_h = np.abs(preds_flat - trues_flat).mean(axis=(0, 2))  # (H,)
    rmse_per_h = np.sqrt(((preds_flat - trues_flat) ** 2).mean(axis=(0, 2)))
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    horizons = np.arange(1, H+1)
    
    # MAE
    ax = axes[0]
    ax.plot(horizons, mae_per_h, 'b-o', linewidth=2, markersize=6)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('MAE')
    ax.set_title('Mean Absolute Error by Horizon')
    ax.grid(True, alpha=0.3)
    
    # RMSE
    ax = axes[1]
    ax.plot(horizons, rmse_per_h, 'r-s', linewidth=2, markersize=6)
    ax.set_xlabel('Forecast Horizon')
    ax.set_ylabel('RMSE')
    ax.set_title('Root Mean Squared Error by Horizon')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--train_file', type=str, default='data/2a/2a_7_train.csv')
    parser.add_argument('--test_file', type=str, default='data/2a/2a_7_test.csv')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    # Load config
    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    from stnf.dataio.kaust_loader import load_kaust_csv, sample_observed_sites, create_dataloaders
    
    # Load training data
    print(f"Loading data from {args.train_file} and {args.test_file}...")
    z_train, z_test, coords, site_to_idx, metadata = load_kaust_csv(
        args.train_file, 
        args.test_file, 
        normalize=config.get('normalize_target', False)
    )
    print(f"Train shape: {z_train.shape}, Test shape: {z_test.shape}, Coords: {coords.shape}")
    
    # Sample observed sites
    obs_indices = sample_observed_sites(
        coords,
        obs_fraction=config.get('obs_ratio', 0.5),
        sampling_method='uniform'
    )
    print(f"Observed sites: {len(obs_indices)}/{coords.shape[0]}")
    
    # Create dataloaders
    train_loader, val_loader = create_dataloaders(
        z_train=z_train,
        coords=coords,
        obs_indices=obs_indices,
        config={
            'L': config.get('history_len', 10),
            'H': config.get('forecast_horizon', 10),
            'batch_size': config.get('batch_size', 16),
            'num_workers': 0,
            'use_coords_cov': config.get('use_coords', False),
            'use_time_cov': config.get('use_time', False),
            'time_encoding': 'linear'
        },
        val_ratio=0.2
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config)
    model = model.to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create output directory
    Path('outputs').mkdir(exist_ok=True)
    
    # Stage 1: Train Deep Kriging
    model = train_stage1(model, train_loader, val_loader, config, device)
    
    # Stage 2: Train LSTM Forecaster
    model = train_stage2(model, train_loader, val_loader, config, device)
    
    print("\n" + "="*50)
    print("Training Complete!")
    print("="*50)
    
    # Save final model
    torch.save(model.state_dict(), 'outputs/stdk_lstm_final.pt')
    print("Final model saved to outputs/stdk_lstm_final.pt")
    
    # Visualize predictions
    print("\nGenerating visualizations...")
    visualize_predictions(model, val_loader, coords, obs_indices, config, device, output_dir='outputs')


if __name__ == '__main__':
    main()
