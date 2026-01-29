"""
KAUST CSV data loader

Features:
1. Load train.csv, test.csv (x, y, t, z format)
2. Create site indices from (x, y) coordinates (train+test combined)
3. Reconstruct time series matrix (T, S)
4. Sample observed sites (Uniform/Biased)
5. Sliding window Dataset (L-step context, H-step forecast)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List
from pathlib import Path


def load_kaust_csv_single(
    data_path: str,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load KAUST CSV file (single file)
    
    Args:
        data_path: CSV file path
        normalize: Whether to normalize z values
        
    Returns:
        z_data: (T, S) - Complete time series
        coords: (S, 2) - Site coordinates [x, y], already in [0,1]
        metadata: dict - Normalization statistics, etc.
    """
    # Load CSV
    df = pd.read_csv(data_path)
    print(f"[INFO] Loaded data: {len(df)} rows")
    
    # 1. Create site indices
    all_coords = df[['x', 'y']].drop_duplicates().reset_index(drop=True)
    S = len(all_coords)
    print(f"[INFO] Total sites: {S}")
    
    # Site mapping: (x, y) → index
    site_to_idx = {
        (row['x'], row['y']): idx 
        for idx, row in all_coords.iterrows()
    }
    
    # Coordinate array: (S, 2), already in [0,1]^2
    coords = all_coords[['x', 'y']].values.astype(np.float32)
    
    # 2. Time indices
    t_vals = df['t'].values
    T = int(t_vals.max())
    print(f"[INFO] Time range: 1 ~ {T}")
    
    # 3. Reconstruct time series matrix: (T, S)
    z_data = np.full((T, S), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        t_idx = int(row['t']) - 1  # 0-based indexing
        site_idx = site_to_idx[(row['x'], row['y'])]
        z_data[t_idx, site_idx] = row['z']
    
    # 4. Normalize (z values only)
    metadata = {}
    if normalize:
        z_flat = z_data[~np.isnan(z_data)]
        z_mean = z_flat.mean()
        z_std = z_flat.std()
        z_data = (z_data - z_mean) / z_std
        metadata['z_mean'] = z_mean
        metadata['z_std'] = z_std
        print(f"[INFO] Normalized z: mean={z_mean:.4f}, std={z_std:.4f}")
    
    return z_data, coords, metadata


def load_kaust_csv(
    train_path: str,
    test_path: str,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load and preprocess KAUST CSV files
    
    Args:
        train_path: train.csv file path
        test_path: test.csv file path
        normalize: Whether to normalize z values
        
    Returns:
        z_train: (T_tr, S) - Training time series
        z_test: (T_te, S) - Test time series (initialized with NaN)
        coords: (S, 2) - Site coordinates [x, y]
        site_to_idx: dict - (x, y) → site index mapping
        metadata: dict - Normalization statistics, etc.
    """
    # Load CSV
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"[INFO] Loaded train: {len(df_train)} rows")
    print(f"[INFO] Loaded test: {len(df_test)} rows")
    
    # 1. Create site indices (train + test combined)
    # Define sites by unique combinations of (x, y) coordinates
    all_coords = pd.concat([
        df_train[['x', 'y']],
        df_test[['x', 'y']]
    ]).drop_duplicates().reset_index(drop=True)
    
    S = len(all_coords)
    print(f"[INFO] Total sites: {S}")
    
    # Site mapping: (x, y) → index
    site_to_idx = {
        (row['x'], row['y']): idx 
        for idx, row in all_coords.iterrows()
    }
    
    # Coordinate array: (S, 2)
    coords = all_coords[['x', 'y']].values.astype(np.float32)
    
    # 2. Time indices (assuming t starts from 1)
    t_train = df_train['t'].values
    t_test = df_test['t'].values
    
    T_tr = t_train.max()
    T_te_end = t_test.max()
    T_te_start = t_test.min()
    
    print(f"[INFO] Train time range: 1 ~ {T_tr}")
    print(f"[INFO] Test time range: {T_te_start} ~ {T_te_end}")
    
    # 3. Reconstruct time series matrix
    # z_train: (T_tr, S)
    z_train = np.full((T_tr, S), np.nan, dtype=np.float32)
    for _, row in df_train.iterrows():
        t_idx = int(row['t']) - 1  # 0-based indexing
        site_idx = site_to_idx[(row['x'], row['y'])]
        z_train[t_idx, site_idx] = row['z']
    
    # z_test: (T_te, S) - Initialized with NaN (prediction target)
    T_te = T_te_end - T_te_start + 1
    z_test = np.full((T_te, S), np.nan, dtype=np.float32)
    # test.csv doesn't have z values, so keep NaN
    
    # 4. Normalize (based on train data)
    metadata = {}
    if normalize:
        z_train_valid = z_train[~np.isnan(z_train)]
        z_mean = z_train_valid.mean()
        z_std = z_train_valid.std() + 1e-8
        
        z_train = (z_train - z_mean) / z_std
        
        metadata['z_mean'] = float(z_mean)
        metadata['z_std'] = float(z_std)
        print(f"[INFO] Normalized: mean={z_mean:.4f}, std={z_std:.4f}")
    else:
        metadata['z_mean'] = 0.0
        metadata['z_std'] = 1.0
    
    # 5. Metadata
    metadata.update({
        'S': S,
        'T_tr': T_tr,
        'T_te': T_te,
        'T_te_start': T_te_start,
        'coords': coords,
        'site_to_idx': site_to_idx
    })
    
    return z_train, z_test, coords, site_to_idx, metadata


def load_test_ground_truth_from_full(
    full_csv_path: str,
    site_to_idx: dict,
    T_te_start: int,
    T_te: int,
) -> np.ndarray:
    """
    Load test-period z values from the full CSV (e.g. 2b_8.csv) so we can
    evaluate on the provider's test set. Uses site_to_idx from load_kaust_csv
    so site order matches.

    Args:
        full_csv_path: path to full CSV (x, y, t, z) with all time steps
        site_to_idx: (x, y) -> index from load_kaust_csv
        T_te_start: first test time step (1-based, e.g. 91)
        T_te: number of test time steps

    Returns:
        z_test_gt: (T_te, S) float32 array
    """
    df = pd.read_csv(full_csv_path)
    S = len(site_to_idx)
    z_test_gt = np.full((T_te, S), np.nan, dtype=np.float32)
    for _, row in df.iterrows():
        t_val = int(row['t'])
        if t_val < T_te_start or t_val > T_te_start + T_te - 1:
            continue
        t_idx = t_val - T_te_start  # 0-based
        key = (float(row['x']), float(row['y']))
        if key not in site_to_idx:
            continue
        site_idx = site_to_idx[key]
        z_test_gt[t_idx, site_idx] = row['z']
    return z_test_gt


def load_kaust_csv_with_test_gt(
    train_path: str,
    test_path: str,
    full_csv_path: str,
    normalize: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Load provider train/test split and fill test z from full CSV for evaluation.

    Returns:
        z_full: (T_tr + T_te, S) - train then test time steps
        coords: (S, 2)
        metadata: includes z_mean, z_std (from train only), T_tr, T_te, T_te_start
    """
    z_train, z_test_empty, coords, site_to_idx, meta = load_kaust_csv(
        train_path, test_path, normalize=False
    )
    T_tr = meta['T_tr']
    T_te = meta['T_te']
    T_te_start = meta['T_te_start']

    z_test_gt = load_test_ground_truth_from_full(
        full_csv_path, site_to_idx, T_te_start, T_te
    )
    z_full = np.concatenate([z_train, z_test_gt], axis=0).astype(np.float32)

    if normalize:
        z_train_valid = z_train[~np.isnan(z_train)]
        z_mean = float(z_train_valid.mean())
        z_std = float(z_train_valid.std() + 1e-8)
        z_full = (z_full - z_mean) / z_std
        meta['z_mean'] = z_mean
        meta['z_std'] = z_std
    else:
        meta['z_mean'] = 0.0
        meta['z_std'] = 1.0

    return z_full, coords, meta


def sample_observed_sites(
    coords: np.ndarray,
    obs_fraction: float,
    sampling_method: str = 'uniform',
    bias_sigma: float = 0.15,
    bias_temp: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Sample observed sites
    
    Args:
        coords: (S, 2) - Site coordinates
        obs_fraction: Observation ratio (0~1)
        sampling_method: 'uniform' or 'biased'
        bias_sigma: Biased sampling distance scale
        bias_temp: Biased sampling temperature
        seed: Random seed
        
    Returns:
        obs_indices: (n_obs,) - Observed site index array
    """
    if seed is not None:
        np.random.seed(seed)
    
    S = len(coords)
    n_obs = max(1, int(S * obs_fraction))
    
    if sampling_method == 'uniform':
        # Uniform sampling
        obs_indices = np.random.choice(S, size=n_obs, replace=False)
        print(f"[INFO] Sampled {n_obs}/{S} sites (uniform)")
        
    elif sampling_method == 'biased':
        # Biased sampling (weighted near origin)
        # Calculate distances
        distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        
        # Gaussian weights
        weights = np.exp(- (distances**2) / (2 * bias_sigma**2))
        
        # Temperature scaling
        weights = weights ** (1.0 / bias_temp)
        
        # Normalize
        probs = weights / weights.sum()
        
        # Sampling
        obs_indices = np.random.choice(S, size=n_obs, replace=False, p=probs)
        
        avg_dist = distances[obs_indices].mean()
        print(f"[INFO] Sampled {n_obs}/{S} sites (biased, avg_dist={avg_dist:.4f})")
    
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    return np.sort(obs_indices)


class KAUSTWindowDataset(Dataset):
    """
    Sliding window Dataset
    
    During training:
    - Input: Observed site data from [t0-L, t0) interval
    - Target: All site data from [t0, t0+H) interval
    
    Args:
        z_full: (T, S) - Complete time series (train only)
        coords: (S, 2) - Site coordinates
        obs_indices: (n_obs,) - Observed site indices
        L: context length
        H: forecast horizon
        stride: Sliding window stride (default 1)
        t0_min: Minimum t0 (use L if None)
        t0_max: Maximum t0 (use T-H+1 if None)
        use_coords_cov: Use (x, y) as covariates
        use_time_cov: Use t as covariates
        time_encoding: Time encoding method {linear, sinusoidal}
    """
    def __init__(
        self,
        z_full: np.ndarray,
        coords: np.ndarray,
        obs_indices: np.ndarray,
        L: int,
        H: int,
        stride: int = 1,
        t0_min: int = None,
        t0_max: int = None,
        use_coords_cov: bool = False,
        use_time_cov: bool = False,
        time_encoding: str = 'linear'
    ):
        self.z_full = z_full  # (T, S)
        self.coords = coords  # (S, 2)
        self.obs_indices = obs_indices  # (n_obs,)
        self.L = L
        self.H = H
        self.stride = stride
        self.use_coords_cov = use_coords_cov
        self.use_time_cov = use_time_cov
        self.time_encoding = time_encoding
        
        self.T, self.S = z_full.shape
        self.n_obs = len(obs_indices)
        
        # Calculate covariates dimension
        self.p_covariates = 0
        if use_coords_cov:
            self.p_covariates += 2  # (x, y)
        if use_time_cov:
            if time_encoding == 'sinusoidal':
                self.p_covariates += 2  # (sin(t), cos(t))
            else:  # linear
                self.p_covariates += 1  # t
        
        # Valid window start points
        # t0-L >= 0 and t0+H <= T
        if t0_min is None:
            t0_min = L
        if t0_max is None:
            t0_max = self.T - H + 1
        
        self.valid_t0 = list(range(t0_min, t0_max, stride))
        
        cov_info = f", p_cov={self.p_covariates}" if self.p_covariates > 0 else ""
        print(f"[INFO] Dataset: {len(self.valid_t0)} windows (L={L}, H={H}, stride={stride}, t0=[{t0_min}, {t0_max}){cov_info})")
    
    def __len__(self):
        return len(self.valid_t0)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        t0 = self.valid_t0[idx]
        
        # 1. Context: Observed sites from [t0-L, t0)
        y_hist_obs = self.z_full[t0-self.L:t0, self.obs_indices]  # (L, n_obs)
        
        # 2. Target: Only observed sites from [t0, t0+H)
        y_fut = self.z_full[t0:t0+self.H, self.obs_indices]  # (H, n_obs)
        
        # 3. Coordinates
        obs_coords = self.coords[self.obs_indices]  # (n_obs, 2)
        target_coords = self.coords[self.obs_indices]  # (n_obs, 2) - Same!
        
        # 4. Create covariates
        result = {
            'obs_coords': torch.from_numpy(obs_coords).float(),      # (n_obs, 2)
            'target_coords': torch.from_numpy(target_coords).float(), # (n_obs, 2)
            'y_hist_obs': torch.from_numpy(y_hist_obs).float().unsqueeze(-1),  # (L, n_obs, 1)
            'y_fut': torch.from_numpy(y_fut).float().unsqueeze(-1),  # (H, n_obs, 1)
            't0': t0
        }
        
        # Covariates for history (observed sites)
        if self.p_covariates > 0:
            X_hist_list = []
            
            # Coordinate covariates: (n_obs, 2)
            if self.use_coords_cov:
                # Expand (n_obs, 2) to (L, n_obs, 2)
                coords_cov = np.tile(obs_coords[np.newaxis, :, :], (self.L, 1, 1))
                X_hist_list.append(coords_cov)
            
            # Time covariates
            if self.use_time_cov:
                # Time indices: [t0-L, t0) → normalized time
                t_indices = np.arange(t0 - self.L, t0).astype(np.float32)
                t_normalized = t_indices / self.T  # Normalize to [0, 1] range
                
                if self.time_encoding == 'sinusoidal':
                    # sin/cos encoding
                    t_sin = np.sin(2 * np.pi * t_normalized)  # (L,)
                    t_cos = np.cos(2 * np.pi * t_normalized)  # (L,)
                    # (L, n_obs, 2)
                    t_cov = np.stack([
                        np.tile(t_sin[:, np.newaxis], (1, self.n_obs)),
                        np.tile(t_cos[:, np.newaxis], (1, self.n_obs))
                    ], axis=-1)
                else:  # linear
                    # (L, n_obs, 1)
                    t_cov = np.tile(t_normalized[:, np.newaxis, np.newaxis], (1, self.n_obs, 1))
                
                X_hist_list.append(t_cov)
            
            # Concatenate: (L, n_obs, p)
            X_hist_obs = np.concatenate(X_hist_list, axis=-1)
            result['X_hist_obs'] = torch.from_numpy(X_hist_obs).float()
        
        # Covariates for future (target sites)
        if self.p_covariates > 0:
            X_fut_list = []
            
            # Coordinate covariates
            if self.use_coords_cov:
                # (n_obs, 2) - Target has same coordinates
                X_fut_list.append(target_coords)
            
            # Time covariates for future
            if self.use_time_cov:
                # Use only first time point of future (t0)
                t_future = float(t0) / self.T  # Normalize
                
                if self.time_encoding == 'sinusoidal':
                    # sin/cos encoding: (n_obs, 2)
                    t_sin = np.sin(2 * np.pi * t_future)
                    t_cos = np.cos(2 * np.pi * t_future)
                    t_fut_cov = np.tile(np.array([[t_sin, t_cos]]), (self.n_obs, 1))
                else:  # linear
                    # (n_obs, 1)
                    t_fut_cov = np.full((self.n_obs, 1), t_future, dtype=np.float32)
                
                X_fut_list.append(t_fut_cov)
            
            # Concatenate: (n_obs, p)
            if len(X_fut_list) > 0:
                X_fut_target = np.concatenate(X_fut_list, axis=-1)
                result['X_fut_target'] = torch.from_numpy(X_fut_target).float()
        
        return result


def create_dataloaders(
    z_train: np.ndarray,
    coords: np.ndarray,
    obs_indices: np.ndarray,
    config: dict,
    val_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    Create Train/Val DataLoaders (split by Target)
    
    Context is taken from entire z_train,
    but Target (prediction interval) is split into train/valid
    
    Example: T=90, L=24, H=10, val_ratio=0.2
        - Train: t0 = [24, 72), target = [24, 82)
        - Valid: t0 = [72, 80], target = [72, 90)
    
    Args:
        z_train: (T_tr, S) - Training time series
        coords: (S, 2) - Site coordinates
        obs_indices: (n_obs,) - Observed sites
        config: kaust_data.yaml configuration
        val_ratio: Validation ratio
        
    Returns:
        train_loader, val_loader
    """
    L = config['L']
    H = config['H']
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 0)
    
    # Extract covariates settings (if present)
    use_coords_cov = config.get('use_coords_cov', False)
    use_time_cov = config.get('use_time_cov', False)
    time_encoding = config.get('time_encoding', 'linear')
    
    T_tr = z_train.shape[0]
    
    # Split Train/Val by Target
    # Maximum t0: T_tr - H (since target is [t0, t0+H))
    t0_max = T_tr - H  # T=90, H=10 → t0_max = 80
    t0_split = int(t0_max * (1 - val_ratio))  # 0.8 → 64
    
    # Create datasets (share entire z_train, only t0 range differs)
    train_dataset = KAUSTWindowDataset(
        z_train, coords, obs_indices, L, H, stride=1,
        t0_min=L, t0_max=t0_split,  # t0 = [L, t0_split)
        use_coords_cov=use_coords_cov,
        use_time_cov=use_time_cov,
        time_encoding=time_encoding
    )
    
    val_dataset = KAUSTWindowDataset(
        z_train, coords, obs_indices, L, H, stride=1,  # stride=1 for temporal split
        t0_min=t0_split, t0_max=t0_max + 1,  # t0 = [t0_split, t0_max]
        use_coords_cov=use_coords_cov,
        use_time_cov=use_time_cov,
        time_encoding=time_encoding
    )
    
    # DataLoader
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"[INFO] Train: {len(train_dataset)} windows, Val: {len(val_dataset)} windows")
    
    return train_loader, val_loader


def prepare_test_context(
    z_train: np.ndarray,
    coords: np.ndarray,
    obs_indices: np.ndarray,
    L: int
) -> Dict[str, torch.Tensor]:
    """
    Prepare context for test prediction
    
    Use last L time points as context
    
    Args:
        z_train: (T_tr, S)
        coords: (S, 2)
        obs_indices: (n_obs,)
        L: context length
        
    Returns:
        context: dict with obs_coords, target_coords, y_hist_obs
    """
    T_tr, S = z_train.shape
    
    # Last L time points
    y_hist_obs = z_train[-L:, obs_indices]  # (L, n_obs)
    
    obs_coords = coords[obs_indices]  # (n_obs, 2)
    target_coords = coords  # (S, 2)
    
    return {
        'obs_coords': torch.from_numpy(obs_coords).float().unsqueeze(0),  # (1, n_obs, 2)
        'target_coords': torch.from_numpy(target_coords).float().unsqueeze(0),  # (1, S, 2)
        'y_hist_obs': torch.from_numpy(y_hist_obs).float().unsqueeze(0).unsqueeze(-1)  # (1, L, n_obs, 1)
    }


def predictions_to_csv(
    y_pred: np.ndarray,
    test_csv_path: str,
    output_path: str,
    site_to_idx: dict,
    z_mean: float,
    z_std: float,
    denormalize: bool = True
):
    """
    Save prediction results to CSV for submission
    
    Args:
        y_pred: (H, S) - Predictions
        test_csv_path: Original test.csv path (for row order reference)
        output_path: Output CSV path
        site_to_idx: (x, y) → site index mapping
        z_mean, z_std: Normalization statistics
        denormalize: Whether to denormalize
    """
    # Load test.csv
    df_test = pd.read_csv(test_csv_path)
    
    # Denormalize
    if denormalize:
        y_pred = y_pred * z_std + z_mean
    
    # Map predictions
    z_hat_list = []
    for _, row in df_test.iterrows():
        t = int(row['t'])
        site_idx = site_to_idx[(row['x'], row['y'])]
        
        # Convert t to relative index in test interval
        # Here, simply assume first test time point as 0
        t_rel = t - df_test['t'].min()
        
        if t_rel < len(y_pred):
            z_hat = y_pred[t_rel, site_idx]
        else:
            z_hat = np.nan
        
        z_hat_list.append(z_hat)
    
    # Save CSV
    df_output = pd.DataFrame({'z': z_hat_list})
    df_output.to_csv(output_path, index=False)
    print(f"[INFO] Saved predictions to {output_path}")


if __name__ == '__main__':
    # Test code
    train_path = 'data/2b/2b_7_train.csv'
    test_path = 'data/2b/2b_7_test.csv'
    
    # Load
    z_train, z_test, coords, site_to_idx, metadata = load_kaust_csv(
        train_path, test_path, normalize=True
    )
    
    # Sample observed sites
    obs_indices = sample_observed_sites(
        coords, obs_fraction=0.1, sampling_method='uniform', seed=42
    )
    
    print(f"Observed sites: {obs_indices[:10]}...")
    print(f"z_train shape: {z_train.shape}")
    print(f"coords shape: {coords.shape}")
