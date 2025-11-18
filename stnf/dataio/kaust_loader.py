"""
KAUST CSV 데이터 로더

기능:
1. train.csv, test.csv 로딩 (x, y, t, z 형식)
2. (x, y) 좌표로 사이트 인덱스 생성 (train+test 통합)
3. 시계열 매트릭스 (T, S) 재구성
4. 관측 사이트 샘플링 (Uniform/Biased)
5. 슬라이딩 윈도우 Dataset (L-step context, H-step forecast)
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, Optional, List
from pathlib import Path


def load_kaust_csv(
    train_path: str,
    test_path: str,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    KAUST CSV 파일 로딩 및 전처리
    
    Args:
        train_path: train.csv 경로
        test_path: test.csv 경로
        normalize: z 값 정규화 여부
        
    Returns:
        z_train: (T_tr, S) - 학습 시계열
        z_test: (T_te, S) - 테스트 시계열 (NaN으로 초기화)
        coords: (S, 2) - 사이트 좌표 [x, y]
        site_to_idx: dict - (x, y) → site index 매핑
        metadata: dict - 정규화 통계 등
    """
    # Load CSV
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    
    print(f"[INFO] Loaded train: {len(df_train)} rows")
    print(f"[INFO] Loaded test: {len(df_test)} rows")
    
    # 1. 사이트 인덱스 생성 (train + test 통합)
    # (x, y) 좌표의 고유 조합으로 사이트 정의
    all_coords = pd.concat([
        df_train[['x', 'y']],
        df_test[['x', 'y']]
    ]).drop_duplicates().reset_index(drop=True)
    
    S = len(all_coords)
    print(f"[INFO] Total sites: {S}")
    
    # 사이트 매핑: (x, y) → index
    site_to_idx = {
        (row['x'], row['y']): idx 
        for idx, row in all_coords.iterrows()
    }
    
    # 좌표 배열: (S, 2)
    coords = all_coords[['x', 'y']].values.astype(np.float32)
    
    # 2. 시간 인덱스 (t는 1부터 시작한다고 가정)
    t_train = df_train['t'].values
    t_test = df_test['t'].values
    
    T_tr = t_train.max()
    T_te_end = t_test.max()
    T_te_start = t_test.min()
    
    print(f"[INFO] Train time range: 1 ~ {T_tr}")
    print(f"[INFO] Test time range: {T_te_start} ~ {T_te_end}")
    
    # 3. 시계열 매트릭스 재구성
    # z_train: (T_tr, S)
    z_train = np.full((T_tr, S), np.nan, dtype=np.float32)
    for _, row in df_train.iterrows():
        t_idx = int(row['t']) - 1  # 0-based indexing
        site_idx = site_to_idx[(row['x'], row['y'])]
        z_train[t_idx, site_idx] = row['z']
    
    # z_test: (T_te, S) - NaN으로 초기화 (예측 타깃)
    T_te = T_te_end - T_te_start + 1
    z_test = np.full((T_te, S), np.nan, dtype=np.float32)
    # test.csv에는 z가 없으므로 NaN 유지
    
    # 4. 정규화 (train 기준)
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
    
    # 5. 메타데이터
    metadata.update({
        'S': S,
        'T_tr': T_tr,
        'T_te': T_te,
        'T_te_start': T_te_start,
        'coords': coords,
        'site_to_idx': site_to_idx
    })
    
    return z_train, z_test, coords, site_to_idx, metadata


def sample_observed_sites(
    coords: np.ndarray,
    obs_fraction: float,
    sampling_method: str = 'uniform',
    bias_sigma: float = 0.15,
    bias_temp: float = 1.0,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    관측 사이트 샘플링
    
    Args:
        coords: (S, 2) - 사이트 좌표
        obs_fraction: 관측 비율 (0~1)
        sampling_method: 'uniform' or 'biased'
        bias_sigma: biased 샘플링 거리 스케일
        bias_temp: biased 샘플링 temperature
        seed: 랜덤 시드
        
    Returns:
        obs_indices: (n_obs,) - 관측 사이트 인덱스 배열
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
        # Biased sampling (원점 근방 가중)
        # 거리 계산
        distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2)
        
        # 가우시안 가중치
        weights = np.exp(- (distances**2) / (2 * bias_sigma**2))
        
        # Temperature scaling
        weights = weights ** (1.0 / bias_temp)
        
        # 정규화
        probs = weights / weights.sum()
        
        # 샘플링
        obs_indices = np.random.choice(S, size=n_obs, replace=False, p=probs)
        
        avg_dist = distances[obs_indices].mean()
        print(f"[INFO] Sampled {n_obs}/{S} sites (biased, avg_dist={avg_dist:.4f})")
    
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")
    
    return np.sort(obs_indices)


class KAUSTWindowDataset(Dataset):
    """
    슬라이딩 윈도우 Dataset
    
    학습 시:
    - 입력: [t0-L, t0) 구간의 관측 사이트 데이터
    - 타깃: [t0, t0+H) 구간의 전체 사이트 데이터
    
    Args:
        z_full: (T, S) - 전체 시계열 (train만)
        coords: (S, 2) - 사이트 좌표
        obs_indices: (n_obs,) - 관측 사이트 인덱스
        L: context length
        H: forecast horizon
        stride: 슬라이딩 윈도우 stride (기본 1)
        t0_min: 최소 t0 (None이면 L 사용)
        t0_max: 최대 t0 (None이면 T-H+1 사용)
        use_coords_cov: (x, y)를 covariates로 사용
        use_time_cov: t를 covariates로 사용
        time_encoding: 시간 인코딩 방법 {linear, sinusoidal}
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
        
        # Covariates 차원 계산
        self.p_covariates = 0
        if use_coords_cov:
            self.p_covariates += 2  # (x, y)
        if use_time_cov:
            if time_encoding == 'sinusoidal':
                self.p_covariates += 2  # (sin(t), cos(t))
            else:  # linear
                self.p_covariates += 1  # t
        
        # 유효한 윈도우 시작점
        # t0-L >= 0 이고 t0+H <= T
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
        
        # 1. Context: [t0-L, t0)의 관측 사이트
        y_hist_obs = self.z_full[t0-self.L:t0, self.obs_indices]  # (L, n_obs)
        # transpose 제거 - 이미 올바른 shape
        
        # 2. Target: [t0, t0+H)의 관측 사이트만 (나머지는 어차피 NaN)
        y_fut = self.z_full[t0:t0+self.H, self.obs_indices]  # (H, n_obs)
        # transpose 제거 - 이미 올바른 shape
        
        # 3. 좌표 (관측 사이트만)
        obs_coords = self.coords[self.obs_indices]  # (n_obs, 2)
        target_coords = self.coords[self.obs_indices]  # (n_obs, 2) - 동일!
        
        # To torch
        return {
            'obs_coords': torch.from_numpy(obs_coords).float(),      # (n_obs, 2)
            'target_coords': torch.from_numpy(target_coords).float(), # (S, 2)
            'y_hist_obs': torch.from_numpy(y_hist_obs).float().unsqueeze(-1),  # (L, n_obs, 1)
            'y_fut': torch.from_numpy(y_fut).float().unsqueeze(-1),  # (H, S, 1)
            't0': t0
        }


def create_dataloaders(
    z_train: np.ndarray,
    coords: np.ndarray,
    obs_indices: np.ndarray,
    config: dict,
    val_ratio: float = 0.2
) -> Tuple[DataLoader, DataLoader]:
    """
    Train/Val DataLoader 생성 (Target 기준 분할)
    
    Context는 전체 z_train에서 가져오되,
    Target (예측 구간)만 train/valid로 분리
    
    예: T=90, L=24, H=10, val_ratio=0.2
        - Train: t0 = [24, 72), target = [24, 82)
        - Valid: t0 = [72, 80], target = [72, 90)
    
    Args:
        z_train: (T_tr, S) - 학습 시계열
        coords: (S, 2) - 사이트 좌표
        obs_indices: (n_obs,) - 관측 사이트
        config: kaust_data.yaml 설정
        val_ratio: 검증 비율
        
    Returns:
        train_loader, val_loader
    """
    L = config['L']
    H = config['H']
    batch_size = config['batch_size']
    num_workers = config.get('num_workers', 0)
    
    T_tr = z_train.shape[0]
    
    # Target 기준 Train/Val 분할
    # t0의 최대값: T_tr - H (target이 [t0, t0+H)이므로)
    t0_max = T_tr - H  # T=90, H=10 → t0_max = 80
    t0_split = int(t0_max * (1 - val_ratio))  # 0.8 → 64
    
    # Dataset 생성 (전체 z_train 공유, t0 범위만 다름)
    train_dataset = KAUSTWindowDataset(
        z_train, coords, obs_indices, L, H, stride=1,
        t0_min=L, t0_max=t0_split  # t0 = [L, t0_split)
    )
    
    val_dataset = KAUSTWindowDataset(
        z_train, coords, obs_indices, L, H, stride=1,  # 시간순 분할이므로 stride=1
        t0_min=t0_split, t0_max=t0_max + 1  # t0 = [t0_split, t0_max]
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
    테스트 예측용 컨텍스트 준비
    
    마지막 L개 시점을 컨텍스트로 사용
    
    Args:
        z_train: (T_tr, S)
        coords: (S, 2)
        obs_indices: (n_obs,)
        L: context length
        
    Returns:
        context: dict with obs_coords, target_coords, y_hist_obs
    """
    T_tr, S = z_train.shape
    
    # 마지막 L개 시점
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
    예측 결과를 제출용 CSV로 저장
    
    Args:
        y_pred: (H, S) - 예측값
        test_csv_path: 원본 test.csv 경로 (행 순서 참조)
        output_path: 출력 CSV 경로
        site_to_idx: (x, y) → site index 매핑
        z_mean, z_std: 정규화 통계
        denormalize: 역정규화 여부
    """
    # Load test.csv
    df_test = pd.read_csv(test_csv_path)
    
    # 역정규화
    if denormalize:
        y_pred = y_pred * z_std + z_mean
    
    # 예측값 매핑
    z_hat_list = []
    for _, row in df_test.iterrows():
        t = int(row['t'])
        site_idx = site_to_idx[(row['x'], row['y'])]
        
        # t는 test 구간의 상대 인덱스로 변환 필요
        # 여기서는 단순하게 첫 test 시점을 0으로 가정
        t_rel = t - df_test['t'].min()
        
        if t_rel < len(y_pred):
            z_hat = y_pred[t_rel, site_idx]
        else:
            z_hat = np.nan
        
        z_hat_list.append(z_hat)
    
    # CSV 저장
    df_output = pd.DataFrame({'z': z_hat_list})
    df_output.to_csv(output_path, index=False)
    print(f"[INFO] Saved predictions to {output_path}")


if __name__ == '__main__':
    # 테스트 코드
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
