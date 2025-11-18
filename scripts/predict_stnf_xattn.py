"""
STNF-XAttn 예측 스크립트

학습된 모델로 테스트 데이터 예측 및 제출용 CSV 생성

Usage:
    python scripts/predict_stnf_xattn.py \
        --train_csv data/2b/2b_7_train.csv \
        --test_csv data/2b/2b_7_test.csv \
        --ckpt outputs/2b7_ckpt.pt \
        --L 24 \
        --H 10 \
        --out_csv outputs/2b7_pred.csv
"""
import argparse
import torch
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from stnf.models.stnf_xattn import create_model
from stnf.dataio import load_kaust_csv, prepare_test_context, predictions_to_csv
from stnf.utils import set_seed


def predict(
    model: torch.nn.Module,
    obs_coords: torch.Tensor,
    target_coords: torch.Tensor,
    y_hist_obs: torch.Tensor,
    H: int,
    device: str = 'cuda',
    use_autoregressive: bool = False
) -> np.ndarray:
    """
    모델로 예측 수행
    
    Args:
        model: 학습된 모델
        obs_coords: (1, n_obs, 2)
        target_coords: (1, S, 2)
        y_hist_obs: (1, L, n_obs, 1)
        H: 예측 horizon
        device: 디바이스
        use_autoregressive: Autoregressive 예측 사용 여부
        
    Returns:
        y_pred: (H, S) - 예측값
    """
    model.eval()
    
    with torch.no_grad():
        obs_coords = obs_coords.to(device)
        target_coords = target_coords.to(device)
        y_hist_obs = y_hist_obs.to(device)
        
        if use_autoregressive:
            y_pred = model.predict_autoregressive(
                obs_coords, target_coords, y_hist_obs, H,
                use_decoder_feedback=True
            )
        else:
            y_pred = model(obs_coords, target_coords, y_hist_obs, H)
        
        # (1, H, S, 1) → (H, S)
        y_pred = y_pred.squeeze(0).squeeze(-1).cpu().numpy()
    
    return y_pred


def main():
    parser = argparse.ArgumentParser(description='Predict with STNF-XAttn model')
    parser.add_argument('--train_csv', type=str, required=True, help='Path to train.csv')
    parser.add_argument('--test_csv', type=str, required=True, help='Path to test.csv')
    parser.add_argument('--ckpt', type=str, required=True, help='Path to checkpoint')
    parser.add_argument('--L', type=int, required=True, help='Context length')
    parser.add_argument('--H', type=int, required=True, help='Forecast horizon')
    parser.add_argument('--out_csv', type=str, required=True, help='Output CSV path')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--autoregressive', action='store_true', help='Use autoregressive prediction')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print("[WARNING] CUDA not available, using CPU")
        device = 'cpu'
    
    print(f"\n{'='*60}")
    print(f"STNF-XAttn Prediction")
    print(f"{'='*60}")
    print(f"Train CSV:   {args.train_csv}")
    print(f"Test CSV:    {args.test_csv}")
    print(f"Checkpoint:  {args.ckpt}")
    print(f"Device:      {device}")
    print(f"L (context): {args.L}")
    print(f"H (horizon): {args.H}")
    print(f"Autoregressive: {args.autoregressive}")
    print(f"{'='*60}\n")
    
    # Load checkpoint
    print("Loading checkpoint...")
    checkpoint = torch.load(args.ckpt, map_location='cpu')
    
    # Extract metadata
    if 'metadata' not in checkpoint:
        raise ValueError("Checkpoint does not contain metadata. Please retrain the model.")
    
    metadata = checkpoint['metadata']
    obs_indices = checkpoint.get('obs_indices', None)
    
    if obs_indices is None:
        raise ValueError("Checkpoint does not contain obs_indices. Please retrain the model.")
    
    print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"  Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.6f}")
    print(f"  Observed sites: {len(obs_indices)}/{metadata['S']}")
    
    # Load data
    print("\nLoading data...")
    z_train, z_test, coords, site_to_idx, data_metadata = load_kaust_csv(
        args.train_csv,
        args.test_csv,
        normalize=True  # 항상 정규화 (checkpoint에 통계 저장됨)
    )
    
    # Use normalization stats from checkpoint
    z_mean = metadata.get('z_mean', data_metadata['z_mean'])
    z_std = metadata.get('z_std', data_metadata['z_std'])
    
    print(f"  z_train shape: {z_train.shape}")
    print(f"  coords shape: {coords.shape}")
    print(f"  Normalization: mean={z_mean:.4f}, std={z_std:.4f}")
    
    # Create model
    print("\nCreating model...")
    model_config = checkpoint.get('config', checkpoint.get('cfg_train', {}))
    model = create_model(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Prepare test context
    print("\nPreparing test context...")
    context = prepare_test_context(z_train, coords, obs_indices, args.L)
    
    print(f"  Context shape: {context['y_hist_obs'].shape}")
    print(f"  Observed coords: {context['obs_coords'].shape}")
    print(f"  Target coords: {context['target_coords'].shape}")
    
    # Predict
    print(f"\nPredicting {args.H} steps ahead...")
    y_pred = predict(
        model=model,
        obs_coords=context['obs_coords'],
        target_coords=context['target_coords'],
        y_hist_obs=context['y_hist_obs'],
        H=args.H,
        device=device,
        use_autoregressive=args.autoregressive
    )
    
    print(f"  Prediction shape: {y_pred.shape}")
    print(f"  Prediction range: [{y_pred.min():.4f}, {y_pred.max():.4f}] (normalized)")
    
    # Denormalize
    y_pred_denorm = y_pred * z_std + z_mean
    print(f"  Denormalized range: [{y_pred_denorm.min():.4f}, {y_pred_denorm.max():.4f}]")
    
    # Save to CSV
    print(f"\nSaving predictions to {args.out_csv}...")
    predictions_to_csv(
        y_pred=y_pred,
        test_csv_path=args.test_csv,
        output_path=args.out_csv,
        site_to_idx=site_to_idx,
        z_mean=z_mean,
        z_std=z_std,
        denormalize=True
    )
    
    # Summary statistics
    df_output = pd.read_csv(args.out_csv)
    print(f"\n{'='*60}")
    print(f"Prediction Summary")
    print(f"{'='*60}")
    print(f"  Total predictions: {len(df_output)}")
    print(f"  Mean: {df_output['z'].mean():.4f}")
    print(f"  Std:  {df_output['z'].std():.4f}")
    print(f"  Min:  {df_output['z'].min():.4f}")
    print(f"  Max:  {df_output['z'].max():.4f}")
    print(f"{'='*60}\n")
    
    print(f"✓ Predictions saved to: {args.out_csv}")


if __name__ == '__main__':
    main()
