"""
STNF-XAttn 학습 스크립트

Usage:
    python scripts/train_stnf_xattn.py \
        --train_csv data/2b/2b_7_train.csv \
        --test_csv data/2b/2b_7_test.csv \
        --cfg_data configs/kaust_data.yaml \
        --cfg_train configs/train.yaml \
        --out outputs/2b7_ckpt.pt
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
import pandas as pd
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from stnf.models.stnf_xattn import create_model
from stnf.dataio import load_kaust_csv, sample_observed_sites, create_dataloaders
from stnf.utils import set_seed, compute_metrics, print_metrics


class Trainer:
    """STNF-XAttn 학습 관리자"""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader,
        val_loader,
        config: dict,
        device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device
        
        # Optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config['lr'],
            weight_decay=config.get('weight_decay', 1e-5)
        )
        
        # Scheduler (optional)
        self.scheduler = None
        if config.get('scheduler') == 'cosine':
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config.get('scheduler_params', {}).get('T_max', config['epochs'])
            )
        elif config.get('scheduler') == 'step':
            self.scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=config.get('scheduler_params', {}).get('step_size', 10),
                gamma=config.get('scheduler_params', {}).get('gamma', 0.5)
            )
        
        # Loss
        self.criterion = nn.MSELoss()
        
        # Training state
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.patience = config.get('patience', 10)
        
        # Gradient clipping
        self.grad_clip = config.get('grad_clip', 0.0)
        
        # History
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_rmse': [],
            'val_mae': [],
            'lr': []
        }
    
    def train_epoch(self, epoch: int) -> float:
        """1 에폭 학습"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc=f"Epoch {epoch+1} [Train]")
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            obs_coords = batch['obs_coords'].to(self.device)
            target_coords = batch['target_coords'].to(self.device)
            y_hist_obs = batch['y_hist_obs'].to(self.device)
            y_fut = batch['y_fut'].to(self.device)
            
            B, L, n_obs, _ = y_hist_obs.shape
            H, S = y_fut.shape[1], y_fut.shape[2]
            
            # Forward
            self.optimizer.zero_grad()
            y_pred = self.model(obs_coords, target_coords, y_hist_obs, H)
            
            # Loss (ignore NaN)
            mask = ~torch.isnan(y_fut)
            if mask.sum() > 0:
                loss = self.criterion(y_pred[mask], y_fut[mask])
            else:
                loss = torch.tensor(0.0, device=self.device)
            
            # Backward
            loss.backward()
            
            # Gradient clipping
            if self.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
            
            self.optimizer.step()
            
            # Stats
            total_loss += loss.item()
            num_batches += 1
            
            # Progress bar
            if batch_idx % self.config.get('log_interval', 10) == 0:
                pbar.set_postfix({'loss': loss.item()})
        
        avg_loss = total_loss / max(num_batches, 1)
        return avg_loss
    
    def validate(self, epoch: int) -> dict:
        """검증"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        all_y_true = []
        all_y_pred = []
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f"Epoch {epoch+1} [Val]")
            for batch in pbar:
                obs_coords = batch['obs_coords'].to(self.device)
                target_coords = batch['target_coords'].to(self.device)
                y_hist_obs = batch['y_hist_obs'].to(self.device)
                y_fut = batch['y_fut'].to(self.device)
                
                H = y_fut.shape[1]
                
                # Forward
                y_pred = self.model(obs_coords, target_coords, y_hist_obs, H)
                
                # Loss
                mask = ~torch.isnan(y_fut)
                if mask.sum() > 0:
                    loss = self.criterion(y_pred[mask], y_fut[mask])
                    total_loss += loss.item()
                    num_batches += 1
                
                # Collect for metrics
                all_y_true.append(y_fut.cpu())
                all_y_pred.append(y_pred.cpu())
        
        avg_loss = total_loss / max(num_batches, 1)
        
        # Compute metrics (validation set이 비어있을 수 있음)
        if len(all_y_true) == 0:
            print(f"[WARNING] Validation set is empty!")
            metrics = {
                'loss': avg_loss,
                'rmse': float('inf'),
                'mae': float('inf'),
                'r2': -1.0
            }
        else:
            y_true_all = torch.cat(all_y_true, dim=0)
            y_pred_all = torch.cat(all_y_pred, dim=0)
            
            metrics = compute_metrics(y_true_all, y_pred_all, per_horizon=False)
            metrics['loss'] = avg_loss
        
        return metrics
    
    def fit(self, epochs: int, save_path: str):
        """전체 학습 루프"""
        print(f"\n{'='*60}")
        print(f"Starting training for {epochs} epochs")
        print(f"Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"{'='*60}\n")
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            val_loss = val_metrics['loss']
            
            # Scheduler step
            if self.scheduler is not None:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]['lr']
            else:
                current_lr = self.config['lr']
            
            # History
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_rmse'].append(val_metrics['rmse'])
            self.history['val_mae'].append(val_metrics['mae'])
            self.history['lr'].append(current_lr)
            
            # Print
            print(f"\nEpoch {epoch+1}/{epochs}")
            print(f"  Train Loss: {train_loss:.6f}")
            print(f"  Val Loss:   {val_loss:.6f}")
            print(f"  Val RMSE:   {val_metrics['rmse']:.6f}")
            print(f"  Val MAE:    {val_metrics['mae']:.6f}")
            print(f"  Val R²:     {val_metrics['r2']:.6f}")
            print(f"  LR:         {current_lr:.6e}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                
                # Save best model
                self.save_checkpoint(save_path, epoch, val_metrics)
                print(f"  → Best model saved! (Val Loss: {val_loss:.6f})")
            else:
                self.patience_counter += 1
                print(f"  → Patience: {self.patience_counter}/{self.patience}")
                
                if self.patience_counter >= self.patience:
                    print(f"\n{'='*60}")
                    print(f"Early stopping at epoch {epoch+1}")
                    print(f"Best Val Loss: {self.best_val_loss:.6f}")
                    print(f"{'='*60}\n")
                    break
        
        print(f"\n{'='*60}")
        print(f"Training completed!")
        print(f"Best Val Loss: {self.best_val_loss:.6f}")
        print(f"Model saved to: {save_path}")
        print(f"{'='*60}\n")
        
        # Save history to CSV and plot
        self.save_history(save_path)
        
        return self.history
    
    def save_history(self, model_path: str):
        """학습 history를 CSV와 plot으로 저장"""
        output_dir = Path(model_path).parent
        base_name = Path(model_path).stem  # e.g., "2a7_full_ckpt"
        
        # 1. Save CSV
        csv_path = output_dir / f"{base_name}_history.csv"
        df = pd.DataFrame({
            'epoch': list(range(1, len(self.history['train_loss']) + 1)),
            'train_loss': self.history['train_loss'],
            'val_loss': self.history['val_loss'],
            'val_rmse': self.history['val_rmse'],
            'val_mae': self.history['val_mae'],
            'lr': self.history['lr']
        })
        df.to_csv(csv_path, index=False)
        print(f"[INFO] History saved to: {csv_path}")
        
        # 2. Save plot
        plot_path = output_dir / f"{base_name}_loss_plot.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        fig.suptitle('Training History', fontsize=16)
        
        epochs = df['epoch'].values
        
        # Loss
        ax = axes[0, 0]
        ax.plot(epochs, df['train_loss'], 'b-', label='Train Loss', marker='o', markersize=3)
        ax.plot(epochs, df['val_loss'], 'r-', label='Val Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train/Val Loss')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # RMSE
        ax = axes[0, 1]
        ax.plot(epochs, df['val_rmse'], 'g-', label='Val RMSE', marker='^', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('RMSE')
        ax.set_title('Validation RMSE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # MAE
        ax = axes[1, 0]
        ax.plot(epochs, df['val_mae'], 'm-', label='Val MAE', marker='d', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MAE')
        ax.set_title('Validation MAE')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Learning Rate
        ax = axes[1, 1]
        ax.plot(epochs, df['lr'], 'c-', label='Learning Rate', marker='*', markersize=4)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('LR')
        ax.set_title('Learning Rate Schedule')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        plt.tight_layout()
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"[INFO] Plot saved to: {plot_path}")
    
    def save_checkpoint(self, path: str, epoch: int, val_metrics: dict):
        """체크포인트 저장"""
        # Create output directory if not exists
        output_dir = Path(path).parent
        output_dir.mkdir(parents=True, exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'val_metrics': val_metrics,
            'best_val_loss': self.best_val_loss,
            'history': self.history,
            'config': self.config
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        torch.save(checkpoint, path)


def main():
    parser = argparse.ArgumentParser(description='Train STNF-XAttn model')
    parser.add_argument('--config', type=str, default='configs/config.yaml', 
                       help='Path to config file (default: configs/config.yaml)')
    parser.add_argument('--mode', type=str, choices=['train', 'debug'], default='train',
                       help='Mode: train (default config) or debug (debug config)')
    args = parser.parse_args()
    
    # Load config based on mode
    if args.mode == 'debug':
        config_path = 'configs/config_debug.yaml'
        print("[INFO] Running in DEBUG mode")
    else:
        config_path = args.config
        print("[INFO] Running in TRAIN mode")
    
    print(f"[INFO] Loading config from: {config_path}")
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # Extract sub-configs for compatibility
    cfg_data = {
        'L': config['dataloader']['L'],
        'H': config['dataloader']['H'],
        'batch_size': config['dataloader']['batch_size'],
        'num_workers': config['dataloader']['num_workers'],
        'val_ratio_t': config['dataloader']['val_ratio_t'],
        'obs_fraction': config['sampling']['obs_fraction'],
        'obs_sampling': config['sampling']['obs_sampling'],
        'time_varying_obs': config['sampling']['time_varying_obs'],
        'bias_sigma': config['sampling']['bias_sigma'],
        'bias_temp': config['sampling']['bias_temp'],
        'normalize_target': config['preprocessing']['normalize_target'],
        'normalize_coords': config['preprocessing']['normalize_coords']
    }
    
    cfg_train = {
        'epochs': config['training']['epochs'],
        'lr': config['training']['lr'],
        'weight_decay': config['training']['weight_decay'],
        'patience': config['training']['patience'],
        'device': config['training']['device'],
        'seed': config['training']['seed'],
        'grad_clip': config['training']['grad_clip'],
        'scheduler': config['training']['scheduler'],
        'scheduler_params': config['training'].get('scheduler_params', {}),
        'd_site': config['model']['d_site'],
        'h_temporal': config['model']['h_temporal'],
        'dropout': config['model']['dropout'],
        'layernorm': config['model']['layernorm'],
        'n_heads': config['model']['n_heads'],
        'log_interval': config['logging']['log_interval'],
        'save_best_only': config['logging']['save_best_only']
    }
    
    # Data paths
    train_csv = config['data']['train_csv']
    test_csv = config['data']['test_csv']
    output_dir = config['data']['output_dir']
    experiment_name = config['data']['experiment_name']
    output_path = f"{output_dir}/{experiment_name}_ckpt.pt"
    
    # Set seed
    set_seed(cfg_train['seed'])
    
    # Device
    device = cfg_train.get('device', 'cuda')
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"Configuration")
    print(f"{'='*60}")
    print(f"Config:    {config_path}")
    print(f"Train CSV: {train_csv}")
    print(f"Test CSV:  {test_csv}")
    print(f"Output:    {output_path}")
    print(f"Device:    {device}")
    print(f"Seed:      {cfg_train['seed']}")
    print(f"{'='*60}\n")
    
    # Load data
    print("Loading data...")
    z_train, z_test, coords, site_to_idx, metadata = load_kaust_csv(
        train_csv,
        test_csv,
        normalize=cfg_data.get('normalize_target', True)
    )
    
    # Sample observed sites
    print("\nSampling observed sites...")
    obs_indices = sample_observed_sites(
        coords,
        obs_fraction=cfg_data['obs_fraction'],
        sampling_method=cfg_data.get('obs_sampling', 'uniform'),
        bias_sigma=cfg_data.get('bias_sigma', 0.15),
        bias_temp=cfg_data.get('bias_temp', 1.0),
        seed=cfg_train['seed']
    )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
    train_loader, val_loader = create_dataloaders(
        z_train,
        coords,
        obs_indices,
        cfg_data,
        val_ratio=cfg_data.get('val_ratio_t', 0.2)
    )
    
    # Create model
    print("\nCreating model...")
    model = create_model(cfg_train)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=cfg_train,
        device=device
    )
    
    # Train
    history = trainer.fit(
        epochs=cfg_train['epochs'],
        save_path=output_path
    )
    
    # Save metadata with checkpoint
    print("\nSaving metadata...")
    checkpoint = torch.load(output_path, map_location='cpu')
    checkpoint['metadata'] = metadata
    checkpoint['obs_indices'] = obs_indices
    checkpoint['cfg_data'] = cfg_data
    torch.save(checkpoint, output_path)
    
    print(f"\n{'='*60}")
    print(f"✓ Training complete!")
    print(f"✓ Checkpoint saved to: {output_path}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    main()
