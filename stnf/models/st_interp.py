"""
Spatio-Temporal Interpolation Model

Simple MLP-based model for spatio-temporal prediction:
Input: (Covariates X, coords s, time t)
Embedding: φ(s) via Wendland RBF, ψ(t) via Gaussian RBF
Output: ŷ(s,t) via MLP

No GRU, no Attention - pure function approximation
"""
import torch
import torch.nn as nn
import math
import numpy as np
from sklearn.mixture import GaussianMixture


class SpatialBasisEmbedding(nn.Module):
    """
    Wendland RBF with multi-resolution centers
    
    Two initialization methods:
    1. 'uniform': Regular grid (original method)
       - n_centers: list of integers (e.g., [25, 81, 121])
       - Each integer k -> sqrt(k) x sqrt(k) regular grid in [0,1]^2
       - Bandwidth: 2.5 × grid spacing for each resolution
    
    2. 'gmm': Gaussian Mixture Model (data-adaptive)
       - Fit GMM with n_components from n_centers list
       - Use GMM means as centers
       - Use GMM std * 2.5 as bandwidths
       - Multi-resolution: smaller n_components -> larger bandwidth
    """
    
    def __init__(self, n_centers: list = [25, 81, 121], learnable: bool = False,
                 init_method: str = 'uniform', train_coords: np.ndarray = None):
        super().__init__()
        self.n_centers = n_centers
        self.learnable = learnable
        self.init_method = init_method
        
        if init_method == 'uniform':
            centers, bandwidths = self._init_uniform()
        elif init_method == 'gmm':
            assert train_coords is not None, "train_coords required for GMM initialization"
            centers, bandwidths = self._init_gmm(train_coords)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        if learnable:
            self.centers = nn.Parameter(centers)
            # Store log(bandwidth) to ensure positivity: bandwidth = exp(log_bandwidth)
            self.log_bandwidths = nn.Parameter(torch.log(bandwidths))
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('_bandwidths', bandwidths)  # Use _ prefix to avoid property conflict
        
        self.k = self.centers.shape[0]
    
    @property
    def bandwidths(self):
        """Return bandwidths (always positive via exp transformation)"""
        if self.learnable:
            return torch.exp(self.log_bandwidths)
        else:
            return self._bandwidths
    
    def _init_uniform(self):
        """Original uniform grid initialization"""
        centers_list = []
        bandwidths_list = []
        
        for k in self.n_centers:
            side = int(math.sqrt(k))
            assert side * side == k, f"n_centers must be perfect squares, got {k}"
            
            # Regular grid in [0,1]^2 (including boundaries)
            x = torch.linspace(0, 1, side)
            y = torch.linspace(0, 1, side)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1)  # (k, 2)
            
            # Bandwidth: 2.5 × grid spacing
            spacing = 1.0 / (side - 1) if side > 1 else 1.0
            bandwidth = 2.5 * spacing
            
            centers_list.append(centers)
            bandwidths_list.append(torch.full((k,), bandwidth))
        
        # Concatenate all resolutions
        centers = torch.cat(centers_list, dim=0)  # (sum(n_centers), 2)
        bandwidths = torch.cat(bandwidths_list, dim=0)  # (sum(n_centers),)
        
        return centers, bandwidths
    
    def _init_gmm(self, train_coords: np.ndarray):
        """
        GMM-based data-adaptive initialization
        
        Args:
            train_coords: (N, 2) numpy array of training coordinates in [0,1]^2
        
        Returns:
            centers: (sum(n_centers), 2) tensor
            bandwidths: (sum(n_centers),) tensor
        """
        centers_list = []
        bandwidths_list = []
        
        # Compute uniform bandwidth reference for each resolution (for clipping)
        uniform_bandwidths = []
        for k in self.n_centers:
            side = int(math.sqrt(k))
            spacing = 1.0 / (side - 1) if side > 1 else 1.0
            uniform_bw = 2.5 * spacing
            uniform_bandwidths.append(uniform_bw)
        
        # Convert to float64 for better numerical stability
        train_coords_64 = train_coords.astype(np.float64)
        
        for i, n_components in enumerate(self.n_centers):
            # Fit GMM with spherical covariance (σ²I)
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='spherical',  # σ²I form - simplest
                random_state=42,
                max_iter=250,
                n_init=5,
                init_params='k-means++',  # Better initialization
                reg_covar=1e-6,  # Regularization to avoid singular covariance
                tol=1e-4,
                verbose=0
            )
            gmm.fit(train_coords_64)
            
            # Extract means as centers
            centers = torch.from_numpy(gmm.means_).float()  # (n_components, 2)
            
            # Extract standard deviations from spherical covariances
            # For spherical: covariances_ is (n_components,) array of variances
            # bandwidth = 4.23 * 2.5 * σ
            stds = np.sqrt(gmm.covariances_)  # (n_components,)
            bandwidths_raw = 4.23 * 2.5 * stds  # (n_components,)
            
            # Clip to [0.5 × uniform_bw, 4.0 × uniform_bw]
            uniform_bw = uniform_bandwidths[i]
            bw_min = 0.5 * uniform_bw
            bw_max = 4.0 * uniform_bw
            bandwidths_clipped = np.clip(bandwidths_raw, bw_min, bw_max)
            
            bandwidths = torch.from_numpy(bandwidths_clipped).float()  # (n_components,)
            
            centers_list.append(centers)
            bandwidths_list.append(bandwidths)
        
        # Concatenate all resolutions
        centers = torch.cat(centers_list, dim=0)  # (sum(n_centers), 2)
        bandwidths = torch.cat(bandwidths_list, dim=0)  # (sum(n_centers),)
        
        return centers, bandwidths
    
    def forward(self, coords: torch.Tensor):
        """
        coords: (B, 2) or (N, 2) - normalized coordinates in [0,1]^2
        Returns: (B, k) or (N, k) - Wendland basis
        """
        # Compute pairwise distances
        dist = torch.cdist(coords.unsqueeze(0) if coords.dim() == 2 else coords, 
                          self.centers.unsqueeze(0))  # (1, N, k) or (B, N, k)
        if coords.dim() == 2:
            dist = dist.squeeze(0)  # (N, k)
        
        # Normalize by bandwidth
        r = (dist / self.bandwidths).clamp(max=1.0)
        
        # Wendland C^4 function: (1 - r)^6 * (35*r^2 + 18*r + 3) / 3 for r in [0,1]
        phi = torch.pow(1 - r, 6) * (35 * r**2 + 18 * r + 3) / 3
        
        return phi


class TemporalBasisEmbedding(nn.Module):
    """
    Temporal basis using Gaussian RBF with multi-resolution centers
    Centers: regular grid with n_centers list (e.g., [10, 15, 45]) in [0,1]
    Bandwidth: 2.5 × grid spacing for each resolution
    Basis function: exp(-0.5 * (t - center)^2 / bandwidth^2)
    """
    
    def __init__(self, n_centers: list = [10, 15, 45]):
        super().__init__()
        self.n_centers = n_centers
        
        # Initialize centers and bandwidths for each resolution
        centers_list = []
        bandwidths_list = []
        
        for n in n_centers:
            # Regular grid including boundaries in [0,1]
            centers = torch.linspace(0.0, 1.0, n)
            centers_list.append(centers)
            
            # Bandwidth = 2.5 × grid spacing
            if n > 1:
                grid_spacing = 1.0 / (n - 1)
            else:
                grid_spacing = 1.0
            bandwidth = 2.5 * grid_spacing
            bandwidths_list.append(torch.full((n,), bandwidth))
        
        # Concatenate all centers and bandwidths
        self.register_buffer('centers', torch.cat(centers_list))  # (k_time,)
        self.register_buffer('bandwidths', torch.cat(bandwidths_list))  # (k_time,)
        self.k_time = self.centers.shape[0]
    
    def forward(self, t: torch.Tensor):
        """
        t: (B, 1) or (N, 1) - time values in [t_min, t_max]
        Returns: (B, k_time) or (N, k_time) - Gaussian RBF basis
        """
        # t: (B, 1) or (N, 1), centers: (k_time,)
        # Compute distances: (t - center)
        diff = t - self.centers.view(1, -1)  # (B, k_time) or (N, k_time)
        
        # Gaussian RBF: exp(-0.5 * (diff / bandwidth)^2)
        scaled_diff = diff / self.bandwidths.view(1, -1)  # (B, k_time) or (N, k_time)
        psi = torch.exp(-0.5 * scaled_diff ** 2)  # (B, k_time) or (N, k_time)
        
        return psi


class STInterpMLP(nn.Module):
    """
    Spatio-Temporal Interpolation via MLP
    
    Architecture:
    Input: [X (covariates), φ(s) (spatial basis), ψ(t) (temporal basis)]
    → MLP layers
    → Output: ŷ(s,t)
    """
    
    def __init__(self, 
                 p: int = 0,  # number of covariates
                 k_spatial_centers: list = [25, 81, 121],  # spatial basis centers (must be perfect squares)
                 k_temporal_centers: list = [10, 15, 45],  # temporal basis centers
                 hidden_dims: list = [256, 256, 128],
                 dropout: float = 0.1,
                 layernorm: bool = True,
                 spatial_learnable: bool = False,
                 spatial_init_method: str = 'uniform',
                 train_coords: np.ndarray = None):
        super().__init__()
        
        self.p = p
        self.k_spatial_centers = k_spatial_centers
        self.spatial_init_method = spatial_init_method
        
        # Spatial basis embedding
        self.spatial_basis = SpatialBasisEmbedding(
            n_centers=k_spatial_centers,
            learnable=spatial_learnable,
            init_method=spatial_init_method,
            train_coords=train_coords
        )
        
        # Temporal basis embedding
        self.temporal_basis = TemporalBasisEmbedding(
            n_centers=k_temporal_centers
        )
        self.k_spatial = self.spatial_basis.k
        self.k_temporal = self.temporal_basis.k_time
        
        # MLP: [X, φ(s), ψ(t)] → ŷ
        input_dim = p + self.k_spatial + self.k_temporal
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if layernorm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor, coords: torch.Tensor, t: torch.Tensor):
        """
        X: (B, p) or (N, p) - covariates (can be empty if p=0)
        coords: (B, 2) or (N, 2) - spatial coordinates (x, y) normalized to [0,1]
        t: (B, 1) or (N, 1) - temporal coordinate normalized to [t_min, t_max]
        
        Returns: (B, 1) or (N, 1) - predicted values ŷ(s,t)
        """
        # Spatial basis embedding
        phi_s = self.spatial_basis(coords)  # (B, k_spatial) or (N, k_spatial)
        
        # Temporal basis embedding
        psi_t = self.temporal_basis(t)  # (B, k_temporal) or (N, k_temporal)
        
        # Concatenate all features
        if X is not None and X.numel() > 0 and self.p > 0:
            features = torch.cat([X, phi_s, psi_t], dim=-1)  # (B, p + k_s + k_t)
        else:
            features = torch.cat([phi_s, psi_t], dim=-1)  # (B, k_s + k_t)
        
        # MLP prediction
        y_pred = self.mlp(features)  # (B, 1) or (N, 1)
        
        return y_pred


def create_model(config: dict, train_coords: np.ndarray = None) -> STInterpMLP:
    """
    Create model from config
    
    Args:
        config: configuration dictionary
        train_coords: (N, 2) numpy array of training coordinates for GMM initialization
    """
    return STInterpMLP(
        p=config.get('p_covariates', 0),
        k_spatial_centers=config.get('k_spatial_centers', [25, 81, 121]),
        k_temporal_centers=config.get('k_temporal_centers', [10, 15, 45]),
        hidden_dims=config.get('hidden_dims', [256, 256, 128]),
        dropout=config.get('dropout', 0.1),
        layernorm=config.get('layernorm', True),
        spatial_learnable=config.get('spatial_learnable', False),
        spatial_init_method=config.get('spatial_init_method', 'uniform'),
        train_coords=train_coords
    )
