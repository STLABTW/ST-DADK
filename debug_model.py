"""모델 forward 디버깅"""
import torch
import yaml
import numpy as np

from stnf.models.stnf_xattn import create_model
from stnf.dataio import load_kaust_csv, sample_observed_sites, create_dataloaders
from stnf.utils import set_seed

# Load config
with open('configs/config.yaml', encoding='utf-8') as f:
    config = yaml.safe_load(f)

set_seed(42)

# Load data
z_train, z_test, coords, site_to_idx, metadata = load_kaust_csv(
    config['data']['train_csv'],
    config['data']['test_csv']
)

# Sample observed sites
obs_indices = sample_observed_sites(coords, obs_fraction=config['sampling']['obs_fraction'])

# Create dataloaders
train_loader, val_loader = create_dataloaders(
    z_train, coords, obs_indices,
    config['dataloader'],
    config['dataloader']['val_ratio_t']
)

# Create model
model = create_model(config['model'])
model.eval()

# Get one batch
batch = next(iter(train_loader))
obs_coords = batch['obs_coords']
target_coords = batch['target_coords']
y_hist_obs = batch['y_hist_obs']
y_fut = batch['y_fut']

print("="*60)
print("INPUT SHAPES")
print("="*60)
print(f"obs_coords: {obs_coords.shape}")
print(f"target_coords: {target_coords.shape}")
print(f"y_hist_obs: {y_hist_obs.shape}")
print(f"y_fut: {y_fut.shape}")

print("\n" + "="*60)
print("INPUT STATISTICS")
print("="*60)
print(f"obs_coords range: [{obs_coords.min():.4f}, {obs_coords.max():.4f}]")
print(f"target_coords range: [{target_coords.min():.4f}, {target_coords.max():.4f}]")
print(f"y_hist_obs range: [{y_hist_obs.min():.4f}, {y_hist_obs.max():.4f}]")
print(f"y_hist_obs mean: {y_hist_obs.mean():.4f}, std: {y_hist_obs.std():.4f}")
print(f"y_fut range: [{y_fut.min():.4f}, {y_fut.max():.4f}]")
print(f"y_fut mean: {y_fut.mean():.4f}, std: {y_fut.std():.4f}")

# Check if all target coords are same
print(f"\nTarget coords unique: {target_coords[0].shape[0]} sites")
print(f"First 5 target sites:\n{target_coords[0, :5]}")

# Forward with intermediate outputs
print("\n" + "="*60)
print("FORWARD PASS - INTERMEDIATE OUTPUTS")
print("="*60)

with torch.no_grad():
    H = y_fut.shape[1]
    
    # 1. Basis embedding
    if model.basis_embedding is not None:
        phi_obs = model.basis_embedding(obs_coords)
        phi_tar = model.basis_embedding(target_coords)
        print(f"\n1. Basis Embedding:")
        print(f"   phi_obs: {phi_obs.shape}, range: [{phi_obs.min():.4f}, {phi_obs.max():.4f}]")
        print(f"   phi_tar: {phi_tar.shape}, range: [{phi_tar.min():.4f}, {phi_tar.max():.4f}]")
        print(f"   phi_obs mean: {phi_obs.mean():.4f}, std: {phi_obs.std():.4f}")
        print(f"   phi_tar mean: {phi_tar.mean():.4f}, std: {phi_tar.std():.4f}")
        
        # Check if phi varies across sites
        phi_var = phi_tar[0].var(dim=0).mean()
        print(f"   phi_tar variance across sites: {phi_var:.6f}")
    else:
        phi_obs = obs_coords
        phi_tar = target_coords
    
    # 2. SiteEncoder
    E_seq = model.site_encoder(phi_obs, y_hist_obs, None)
    print(f"\n2. SiteEncoder (E_seq):")
    print(f"   Shape: {E_seq.shape}")
    print(f"   Range: [{E_seq.min():.4f}, {E_seq.max():.4f}]")
    print(f"   Mean: {E_seq.mean():.4f}, Std: {E_seq.std():.4f}")
    
    # Check if E_seq varies across sites and time
    E_var_sites = E_seq[0].var(dim=1).mean()
    E_var_time = E_seq[0].var(dim=0).mean()
    print(f"   Variance across sites: {E_var_sites:.6f}")
    print(f"   Variance across time: {E_var_time:.6f}")
    
    # 3. GRURoll
    A, H_emb_obs = model.gru_roll(E_seq, H)
    print(f"\n3. GRURoll:")
    print(f"   A (rollout): {A.shape}, range: [{A.min():.4f}, {A.max():.4f}]")
    print(f"   A mean: {A.mean():.4f}, std: {A.std():.4f}")
    print(f"   H_emb_obs: {H_emb_obs.shape}, range: [{H_emb_obs.min():.4f}, {H_emb_obs.max():.4f}]")
    
    # Check if A varies across horizon
    A_var_h = A[0].var(dim=0).mean()
    print(f"   A variance across horizon: {A_var_h:.6f}")
    
    # 4. CrossAttnHead
    y_pred = model.cross_attn_head(A, phi_tar, H_emb_obs, None)
    print(f"\n4. CrossAttnHead (final prediction):")
    print(f"   Shape: {y_pred.shape}")
    print(f"   Range: [{y_pred.min():.4f}, {y_pred.max():.4f}]")
    print(f"   Mean: {y_pred.mean():.4f}, Std: {y_pred.std():.4f}")
    
    # Check if pred varies across sites
    y_pred_var_sites = y_pred[0, 0].var()
    print(f"   Variance across sites (h=0): {y_pred_var_sites:.6f}")
    
    print("\n" + "="*60)
    print("COMPARISON WITH TARGET")
    print("="*60)
    print(f"Target (y_fut) mean: {y_fut.mean():.4f}, std: {y_fut.std():.4f}")
    print(f"Pred (y_pred) mean: {y_pred.mean():.4f}, std: {y_pred.std():.4f}")
    print(f"MSE Loss: {((y_pred - y_fut)**2).mean():.6f}")
