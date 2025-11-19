"""
2-Stage Deep Kriging + Site-wise LSTM for Spatio-Temporal Forecasting

Stage 1: Deep Kriging - Spatial interpolation using (X, phi_s, phi_t)
Stage 2: Site-wise LSTM - Temporal forecasting per site (shared parameters)
"""
import torch
import torch.nn as nn
import math


class BasisEmbedding(nn.Module):
    """Wendland RBF for spatial basis"""
    
    def __init__(self, k: int = 250, initialize: str = 'regular', learnable: bool = False):
        super().__init__()
        self.k = k
        self.initialize = initialize
        self.learnable = learnable
        
        # Initialize centers
        centers = self._initialize_centers()
        if learnable:
            self.centers = nn.Parameter(centers)
        else:
            self.register_buffer('centers', centers)
    
    def _initialize_centers(self):
        """Initialize k centers in [0,1]^2"""
        if self.initialize == 'regular':
            side = int(math.ceil(math.sqrt(self.k)))
            x = torch.linspace(0, 1, side)
            y = torch.linspace(0, 1, side)
            xx, yy = torch.meshgrid(x, y, indexing='ij')
            centers = torch.stack([xx.flatten(), yy.flatten()], dim=-1)[:self.k]
        else:  # random
            centers = torch.rand(self.k, 2)
        return centers
    
    def forward(self, coords: torch.Tensor):
        """
        coords: (B, n, 2) or (n, 2) - normalized coordinates in [0,1]^2
        Returns: (B, n, k) or (n, k) - Wendland basis
        """
        if coords.dim() == 2:
            coords = coords.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, n, _ = coords.shape
        # coords: (B, n, 2), centers: (k, 2)
        # Compute pairwise distances
        dist = torch.cdist(coords, self.centers.unsqueeze(0).expand(B, -1, -1))  # (B, n, k)
        
        # Wendland function: (1 - r)^4 * (4*r + 1) for r in [0,1]
        r = dist.clamp(max=1.0)
        phi = torch.pow(1 - r, 4) * (4 * r + 1)
        
        if squeeze:
            phi = phi.squeeze(0)
        
        return phi


class TemporalBasisEmbedding(nn.Module):
    """
    Temporal basis using Gaussian RBF with multi-resolution centers
    Centers: regular grid with 10, 15, 45 points (total 70 basis functions)
    Bandwidth: 2.5 × grid spacing for each resolution
    Basis function: exp(-0.5 * (t - center)^2 / bandwidth^2)
    """
    
    def __init__(self, t_min: float = 0.0, t_max: float = 1.0, 
                 n_centers: list = [10, 15, 45]):
        super().__init__()
        self.t_min = t_min
        self.t_max = t_max
        self.n_centers = n_centers
        
        # Initialize centers and bandwidths for each resolution
        centers_list = []
        bandwidths_list = []
        
        for n in n_centers:
            # Regular grid including boundaries
            centers = torch.linspace(t_min, t_max, n)
            centers_list.append(centers)
            
            # Bandwidth = 2.5 × grid spacing
            if n > 1:
                grid_spacing = (t_max - t_min) / (n - 1)
            else:
                grid_spacing = t_max - t_min
            bandwidth = 2.5 * grid_spacing
            bandwidths_list.append(torch.full((n,), bandwidth))
        
        # Concatenate all centers and bandwidths
        self.register_buffer('centers', torch.cat(centers_list))  # (k_time,)
        self.register_buffer('bandwidths', torch.cat(bandwidths_list))  # (k_time,)
        self.k_time = self.centers.shape[0]
    
    def forward(self, t: torch.Tensor):
        """
        t: (B, n, 1) or (n, 1) - time values
        Returns: (B, n, k_time) or (n, k_time) - Gaussian RBF basis
        """
        if t.dim() == 2:
            t = t.unsqueeze(0)  # (1, n, 1)
            squeeze = True
        else:
            squeeze = False
        
        # t: (B, n, 1), centers: (k_time,)
        # Compute distances: (t - center)
        diff = t - self.centers.view(1, 1, -1)  # (B, n, k_time)
        
        # Gaussian RBF: exp(-0.5 * (diff / bandwidth)^2)
        scaled_diff = diff / self.bandwidths.view(1, 1, -1)  # (B, n, k_time)
        phi_t = torch.exp(-0.5 * scaled_diff ** 2)  # (B, n, k_time)
        
        if squeeze:
            phi_t = phi_t.squeeze(0)  # (n, k_time)
        
        return phi_t


class DeepKriging(nn.Module):
    """
    Stage 1: Deep Kriging for spatial interpolation
    Input: (X, phi_s(x,y), phi_t(t))
    Output: z (predicted value at location)
    """
    
    def __init__(self, p: int = 0, k: int = 250, t_min: float = 0.0, t_max: float = 1.0,
                 n_t_centers: list = [10, 15, 45], 
                 hidden_dims: list = [128, 128], dropout: float = 0.1):
        super().__init__()
        self.p = p
        self.k = k
        
        # Spatial basis
        self.spatial_basis = BasisEmbedding(k=k, initialize='regular', learnable=False)
        
        # Temporal basis (multi-resolution Gaussian RBF)
        self.temporal_basis = TemporalBasisEmbedding(
            t_min=t_min, t_max=t_max, n_centers=n_t_centers
        )
        self.k_time = self.temporal_basis.k_time
        
        # MLP: [X, phi_s, phi_t] → z
        input_dim = p + k + self.k_time
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
    
    def forward(self, X: torch.Tensor, coords: torch.Tensor, t: torch.Tensor):
        """
        X: (B, n, p) or (n, p) - covariates
        coords: (B, n, 2) or (n, 2) - spatial coordinates
        t: (B, n, 1) or (n, 1) - time values (normalized to [0, 1])
        
        Returns: z: (B, n, 1) or (n, 1) - predicted values
        """
        # Spatial basis
        phi_s = self.spatial_basis(coords)  # (B, n, k) or (n, k)
        
        # Temporal basis (Gaussian RBF)
        phi_t = self.temporal_basis(t)  # (B, n, k_time) or (n, k_time)
        
        # Concatenate features
        if X.numel() > 0 and self.p > 0:
            features = torch.cat([X, phi_s, phi_t], dim=-1)
        else:
            features = torch.cat([phi_s, phi_t], dim=-1)
        
        # MLP prediction
        z = self.mlp(features)
        
        return z


class SiteWiseLSTM(nn.Module):
    """
    Stage 2: Site-wise LSTM for temporal forecasting
    Each site uses shared LSTM parameters
    """
    
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, 
                 num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Shared LSTM for all sites
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        
        # Autoregressive forecasting with LSTMCell
        self.lstm_cell = nn.LSTMCell(input_dim, hidden_dim)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x_seq: torch.Tensor, H: int):
        """
        x_seq: (B, L, n_sites, input_dim) - input sequence for all sites
        H: forecast horizon
        
        Returns: y_pred: (B, H, n_sites, input_dim) - forecasted sequence
        """
        B, L, n_sites, input_dim = x_seq.shape
        
        # Reshape: (B, L, n_sites, input_dim) → (B*n_sites, L, input_dim)
        x_flat = x_seq.permute(0, 2, 1, 3).contiguous().view(B * n_sites, L, input_dim)
        
        # Encode sequence with LSTM
        _, (h_n, c_n) = self.lstm(x_flat)  # h_n: (num_layers, B*n_sites, hidden_dim)
        
        # Use last layer's hidden state for forecasting
        h_current = h_n[-1]  # (B*n_sites, hidden_dim)
        c_current = c_n[-1]  # (B*n_sites, hidden_dim)
        
        # Autoregressive forecasting
        predictions = []
        x_current = x_flat[:, -1, :]  # (B*n_sites, input_dim) - last observed
        
        for _ in range(H):
            # LSTM step
            h_current, c_current = self.lstm_cell(x_current, (h_current, c_current))
            h_current = self.dropout(h_current)
            
            # Output prediction
            y_t = self.output_proj(h_current)  # (B*n_sites, input_dim)
            predictions.append(y_t)
            x_current = y_t  # Autoregressive
        
        # Stack predictions: (H, B*n_sites, input_dim)
        y_flat = torch.stack(predictions, dim=0)
        
        # Reshape back: (H, B*n_sites, input_dim) → (B, H, n_sites, input_dim)
        y_pred = y_flat.permute(1, 0, 2).view(B, n_sites, H, input_dim).permute(0, 2, 1, 3)
        
        return y_pred


class STDKLSTM(nn.Module):
    """
    2-Stage model: Deep Kriging + Site-wise LSTM
    """
    
    def __init__(self, p: int = 0, k: int = 250, 
                 t_min: float = 0.0, t_max: float = 1.0,
                 n_t_centers: list = [10, 15, 45],
                 kriging_hidden: list = [128, 128], 
                 lstm_hidden: int = 64, lstm_layers: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        
        # Stage 1: Deep Kriging
        self.kriging = DeepKriging(
            p=p, k=k, 
            t_min=t_min, t_max=t_max,
            n_t_centers=n_t_centers,
            hidden_dims=kriging_hidden, 
            dropout=dropout
        )
        
        # Stage 2: Site-wise LSTM
        self.forecaster = SiteWiseLSTM(
            input_dim=1, 
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            dropout=dropout
        )
    
    def interpolate_sites(self, X_hist: torch.Tensor, coords_obs: torch.Tensor,
                          y_hist_obs: torch.Tensor, coords_all: torch.Tensor, 
                          t_hist: torch.Tensor):
        """
        Stage 1: Interpolate all sites using Deep Kriging
        
        X_hist: (B, L, n_obs, p) - observed covariates
        coords_obs: (n_obs, 2) - observed coordinates
        y_hist_obs: (B, L, n_obs, 1) - observed values
        coords_all: (n_all, 2) - all site coordinates
        t_hist: (L,) - time indices
        
        Returns: z_all: (B, L, n_all, 1) - interpolated values for all sites
        """
        B, L, n_obs, p = X_hist.shape
        n_all = coords_all.shape[0]
        
        z_list = []
        for t_idx in range(L):
            # For each time, predict at all locations
            # Use mean of observed X as global covariate (simple approach)
            if p > 0:
                X_mean = X_hist[:, t_idx].mean(dim=1, keepdim=True)  # (B, 1, p)
                X_t = X_mean.expand(B, n_all, p)  # (B, n_all, p)
            else:
                X_t = torch.zeros(B, n_all, 0, device=X_hist.device)
            
            coords_t = coords_all.unsqueeze(0).expand(B, -1, -1)  # (B, n_all, 2)
            t_t = torch.full((B, n_all, 1), t_hist[t_idx].item(), device=X_hist.device)
            
            # Kriging prediction
            z_pred = self.kriging(X_t, coords_t, t_t)  # (B, n_all, 1)
            z_list.append(z_pred)
        
        return torch.stack(z_list, dim=1)  # (B, L, n_all, 1)
    
    def forward(self, X_hist: torch.Tensor, coords_obs: torch.Tensor,
                y_hist_obs: torch.Tensor, coords_all: torch.Tensor,
                t_hist: torch.Tensor, H: int):
        """
        Full 2-stage forward pass
        
        X_hist: (B, L, n_obs, p)
        coords_obs: (n_obs, 2)
        y_hist_obs: (B, L, n_obs, 1)
        coords_all: (n_all, 2)
        t_hist: (L,)
        H: forecast horizon
        
        Returns: y_pred: (B, H, n_all, 1)
        """
        # Stage 1: Interpolation
        z_all = self.interpolate_sites(X_hist, coords_obs, y_hist_obs, coords_all, t_hist)
        
        # Stage 2: Forecasting
        y_pred = self.forecaster(z_all, H)
        
        return y_pred


def create_model(config: dict) -> STDKLSTM:
    """Create model from config"""
    return STDKLSTM(
        p=config.get('p_covariates', 0),
        k=config.get('basis_k', 250),
        t_min=config.get('t_min', 0.0),
        t_max=config.get('t_max', 1.0),
        n_t_centers=config.get('n_t_centers', [10, 15, 45]),
        kriging_hidden=config.get('kriging_hidden', [128, 128]),
        lstm_hidden=config.get('lstm_hidden', 64),
        lstm_layers=config.get('lstm_layers', 2),
        dropout=config.get('dropout', 0.1)
    )
