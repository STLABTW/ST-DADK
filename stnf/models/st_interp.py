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
    Spatial basis embedding with multi-resolution centers
    
    Supports multiple basis function types:
    - 'wendland': Wendland C^4 RBF (compact support)
    - 'gaussian': Gaussian RBF (infinite support, rapid decay)
    - 'triangular': Triangular/Linear basis (compact support)
    
    All basis functions are calibrated to have similar effective support
    (same standard deviation) for fair comparison.
    
    Four initialization methods:
    1. 'uniform': Regular grid (original method)
       - n_centers: list of integers (e.g., [25, 81, 121])
       - Each integer k -> sqrt(k) x sqrt(k) regular grid in [0,1]^2
       - Bandwidth: 2.5 × grid spacing for each resolution
    
    2. 'gmm': Gaussian Mixture Model (data-adaptive, density estimation)
       - Fit GMM with n_components from n_centers list
       - Use GMM means as centers
       - Use GMM std * 2.5 as bandwidths
       - Multi-resolution: smaller n_components -> larger bandwidth
    
    3. 'random_site': Random sampling from observation sites (density-weighted)
       - Randomly sample k sites from training coordinates
       - Bandwidth: 2.5 × average distance to 4 nearest neighbors
       - Data-driven but faster than GMM
    
    4. 'kmeans_balanced': Balanced K-means (density-adaptive via equal coverage)
       - Size-constrained K-means: each cluster has exactly n/k samples
       - Enforces balanced spatial coverage (density-adaptive)
       - Centers can be anywhere in data space
       - Bandwidth: 2.5 × average distance to 4 nearest cluster centers
    """
    
    # Calibration factors to match effective support across basis functions
    # Based on standard deviation matching (Wendland as reference)
    CALIBRATION_FACTORS = {
        'wendland': 1.000000,
        'gaussian': 0.223477,
        'triangular': 0.654714,
    }
    
    def __init__(self, n_centers: list = [25, 81, 121], learnable: bool = False,
                 init_method: str = 'uniform', train_coords: np.ndarray = None,
                 basis_function: str = 'wendland'):
        super().__init__()
        self.n_centers = n_centers
        self.learnable = learnable
        self.init_method = init_method
        self.basis_function = basis_function
        
        # Validate basis function
        if basis_function not in self.CALIBRATION_FACTORS:
            raise ValueError(f"Unknown basis function: {basis_function}. "
                           f"Choose from {list(self.CALIBRATION_FACTORS.keys())}")
        
        if init_method == 'uniform':
            centers, bandwidths = self._init_uniform()
        elif init_method == 'gmm':
            assert train_coords is not None, "train_coords required for GMM initialization"
            centers, bandwidths = self._init_gmm(train_coords)
        elif init_method == 'random_site':
            assert train_coords is not None, "train_coords required for random_site initialization"
            centers, bandwidths = self._init_random_site(train_coords)
        elif init_method == 'kmeans_balanced':
            assert train_coords is not None, "train_coords required for kmeans_balanced initialization"
            centers, bandwidths = self._init_kmeans_balanced(train_coords)
        else:
            raise ValueError(f"Unknown init_method: {init_method}")
        
        if learnable:
            # Learnable centers: no constraints, penalty applied in training
            self.centers = nn.Parameter(centers)
            # Store initial centers for movement penalty
            self.register_buffer('centers_init', centers.clone())
            # Store log(bandwidth) to ensure positivity: bandwidth = exp(log_bandwidth)
            self.log_bandwidths = nn.Parameter(torch.log(bandwidths))
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('_bandwidths', bandwidths)  # Use _ prefix to avoid property conflict
        
        self.k = centers.shape[0]
    
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
            
            # PyTorch < 1.10 compatibility (no indexing parameter)
            try:
                xx, yy = torch.meshgrid(x, y, indexing='ij')
            except TypeError:
                # Old PyTorch version - default behavior is 'ij'
                xx, yy = torch.meshgrid(x, y)
            
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
        
        Uses training samples (with temporal duplicates) to reflect spatio-temporal density.
        Subsamples to 50000 if data is too large for computational efficiency.
        
        Args:
            train_coords: (N, 2) numpy array of training coordinates in [0,1]^2
                         (includes temporal duplicates to reflect data density)
        
        Returns:
            centers: (sum(n_centers), 2) tensor
            bandwidths: (sum(n_centers),) tensor
        """
        centers_list = []
        bandwidths_list = []
        
        # Subsample if too many samples (for computational efficiency)
        max_samples_for_gmm = 10000
        if len(train_coords) > max_samples_for_gmm:
            print(f"  GMM initialization: Subsampling {max_samples_for_gmm}/{len(train_coords)} training samples")
            indices = np.random.choice(len(train_coords), max_samples_for_gmm, replace=False)
            train_coords_sub = train_coords[indices]
        else:
            print(f"  GMM initialization: {len(train_coords)} training samples (with temporal duplicates)")
            train_coords_sub = train_coords
        
        # Compute uniform bandwidth reference for each resolution (for clipping)
        uniform_bandwidths = []
        for k in self.n_centers:
            side = int(math.sqrt(k))
            spacing = 1.0 / (side - 1) if side > 1 else 1.0
            uniform_bw = 2.5 * spacing
            uniform_bandwidths.append(uniform_bw)
        
        # Convert to float64 for better numerical stability
        train_coords_64 = train_coords_sub.astype(np.float64)
        
        for i, n_components in enumerate(self.n_centers):
            # **SPEEDUP 2: Reduced iterations and initializations**
            # Fit GMM with spherical covariance (σ²I)
            gmm = GaussianMixture(
                n_components=n_components,
                covariance_type='spherical',  # σ²I form - simplest
                random_state=42,
                max_iter=100,
                n_init=3,
                init_params='k-means++',  # Better initialization
                reg_covar=1e-6,
                tol=1e-3,
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
            
            # Clip to [0.25 × uniform_bw, Inf]
            uniform_bw = uniform_bandwidths[i]
            bw_min = 0.25 * uniform_bw
            bw_max = float('inf')
            bandwidths_clipped = np.clip(bandwidths_raw, bw_min, bw_max)
            
            bandwidths = torch.from_numpy(bandwidths_clipped).float()  # (n_components,)
            
            centers_list.append(centers)
            bandwidths_list.append(bandwidths)
        
        # Concatenate all resolutions
        centers = torch.cat(centers_list, dim=0)  # (sum(n_centers), 2)
        bandwidths = torch.cat(bandwidths_list, dim=0)  # (sum(n_centers),)
        
        return centers, bandwidths
    
    def _init_random_site(self, train_coords: np.ndarray):
        """
        Random site sampling initialization
        
        For each resolution k:
        1. Randomly sample k sites from training coordinates (WITH temporal duplicates)
        2. Compute bandwidth as 2.5 × average distance to 4 nearest neighbors
        
        This gives a data-driven initialization that's faster than GMM
        and naturally adapts to spatio-temporal data density.
        Sites with more temporal observations have higher probability of being selected.
        
        Args:
            train_coords: (N, 2) numpy array of training coordinates in [0,1]^2
                         (includes temporal duplicates to reflect data density)
        
        Returns:
            centers: (sum(n_centers), 2) tensor
            bandwidths: (sum(n_centers),) tensor
        """
        from scipy.spatial.distance import cdist
        
        centers_list = []
        bandwidths_list = []
        
        # Do NOT use unique - keep temporal duplicates to reflect data density
        print(f"  Random site initialization: {len(train_coords)} training samples (with temporal duplicates)")
        
        for k in self.n_centers:
            # Random sampling from training coordinates (weighted by temporal frequency)
            if k > len(train_coords):
                print(f"  Warning: k={k} exceeds training samples ({len(train_coords)}), sampling with replacement")
                indices = np.random.choice(len(train_coords), k, replace=True)
            else:
                indices = np.random.choice(len(train_coords), k, replace=False)
            
            centers = train_coords[indices]  # (k, 2)
            
            # Compute pairwise distances
            distances = cdist(centers, centers)  # (k, k)
            
            # For each center, find distance to 4 nearest neighbors (excluding self)
            # Set diagonal to infinity to exclude self
            np.fill_diagonal(distances, np.inf)
            
            # Sort distances and take first 4 (or fewer if k < 5)
            n_neighbors = min(4, k - 1) if k > 1 else 1
            sorted_distances = np.sort(distances, axis=1)  # (k, k)
            nearest_distances = sorted_distances[:, :n_neighbors]  # (k, n_neighbors)
            
            # Average distance to n_neighbors nearest sites for each center
            avg_distances = nearest_distances.mean(axis=1)  # (k,)
            
            # Bandwidth = 2.5 × average distance to nearest neighbors
            bandwidth_scale = 2.5
            bandwidths = avg_distances * bandwidth_scale  # (k,)
            
            # Handle edge case: if all distances are inf (k=1), use default
            if k == 1:
                side = int(math.sqrt(self.n_centers[0]))
                spacing = 1.0 / (side - 1) if side > 1 else 1.0
                bandwidths = np.array([2.5 * spacing])
            
            centers_list.append(torch.from_numpy(centers).float())
            bandwidths_list.append(torch.from_numpy(bandwidths).float())
        
        # Concatenate all resolutions
        centers = torch.cat(centers_list, dim=0)  # (sum(n_centers), 2)
        bandwidths = torch.cat(bandwidths_list, dim=0)  # (sum(n_centers),)
        
        return centers, bandwidths
    
    def _init_kmeans_balanced(self, train_coords: np.ndarray):
        """
        Balanced K-means clustering initialization
        
        For each resolution k:
        1. Run size-constrained K-means: each cluster has exactly n/k samples
        2. Use cluster centers as basis centers (can be anywhere, not limited to obs sites)
        3. Compute bandwidth as 2.5 × average distance to 4 nearest cluster centers
        
        This enforces balanced spatial coverage (each center covers equal number of samples).
        Unlike standard K-means (which minimizes variance), this ensures equal cluster sizes.
        Subsamples to 50000 if data is too large for computational efficiency.
        
        Args:
            train_coords: (N, 2) numpy array of training coordinates in [0,1]^2
                         (includes temporal duplicates to reflect data density)
        
        Returns:
            centers: (sum(n_centers), 2) tensor
            bandwidths: (sum(n_centers),) tensor
        """
        from scipy.spatial.distance import cdist
        from k_means_constrained import KMeansConstrained
        
        centers_list = []
        bandwidths_list = []
        
        # Subsample if too many samples (for computational efficiency)
        max_samples_for_kmeans = 10000
        if len(train_coords) > max_samples_for_kmeans:
            print(f"  Balanced K-means initialization: Subsampling {max_samples_for_kmeans}/{len(train_coords)} training samples")
            indices = np.random.choice(len(train_coords), max_samples_for_kmeans, replace=False)
            train_coords_sub = train_coords[indices]
        else:
            print(f"  Balanced K-means initialization: {len(train_coords)} training samples (with temporal duplicates)")
            train_coords_sub = train_coords
        
        for k in self.n_centers:
            # Size-constrained K-means: each cluster has size_min = size_max = n/k
            n_samples = len(train_coords_sub)
            size_per_cluster = n_samples // k
            
            # Handle case where n is not perfectly divisible by k
            # Set size_min slightly smaller to allow flexibility
            size_min = max(1, size_per_cluster - 1)
            size_max = size_per_cluster + (n_samples % k)  # Last cluster can be slightly larger
            
            kmeans_balanced = KMeansConstrained(
                n_clusters=k,
                size_min=size_min,
                size_max=size_max,
                random_state=42,
                n_init=3,  # Match GMM (was 10)
                max_iter=100  # Match GMM (was 300)
            )
            kmeans_balanced.fit(train_coords_sub)
            
            # Extract cluster centers
            centers = kmeans_balanced.cluster_centers_  # (k, 2)
            
            # Compute pairwise distances between cluster centers
            distances = cdist(centers, centers)  # (k, k)
            
            # For each center, find distance to 4 nearest other centers (excluding self)
            np.fill_diagonal(distances, np.inf)
            
            # Sort distances and take first 4 (or fewer if k < 5)
            n_neighbors = min(4, k - 1) if k > 1 else 1
            sorted_distances = np.sort(distances, axis=1)  # (k, k)
            nearest_distances = sorted_distances[:, :n_neighbors]  # (k, n_neighbors)
            
            # Average distance to n_neighbors nearest centers
            avg_distances = nearest_distances.mean(axis=1)  # (k,)
            
            # Bandwidth = 2.5 × average distance to nearest centers
            bandwidth_scale = 2.5
            bandwidths = avg_distances * bandwidth_scale  # (k,)
            
            # Handle edge case: if k=1, use default bandwidth
            if k == 1:
                side = int(math.sqrt(self.n_centers[0]))
                spacing = 1.0 / (side - 1) if side > 1 else 1.0
                bandwidths = np.array([2.5 * spacing])
            
            centers_list.append(torch.from_numpy(centers).float())
            bandwidths_list.append(torch.from_numpy(bandwidths).float())
        
        # Concatenate all resolutions
        centers = torch.cat(centers_list, dim=0)  # (sum(n_centers), 2)
        bandwidths = torch.cat(bandwidths_list, dim=0)  # (sum(n_centers),)
        
        return centers, bandwidths
    
    def forward(self, coords: torch.Tensor):
        """
        coords: (B, 2) or (N, 2) - normalized coordinates in [0,1]^2
        Returns: (B, k) or (N, k) - basis function values
        """
        # Compute pairwise distances
        dist = torch.cdist(coords.unsqueeze(0) if coords.dim() == 2 else coords, 
                          self.centers.unsqueeze(0))  # (1, N, k) or (B, N, k)
        if coords.dim() == 2:
            dist = dist.squeeze(0)  # (N, k)
        
        # Normalize by bandwidth and apply calibration factor
        # Calibration factors < 1.0 make the basis NARROWER (more local)
        # by dividing bandwidth by a smaller value
        calibration = self.CALIBRATION_FACTORS[self.basis_function]
        r = dist / (self.bandwidths * calibration)
        
        # Apply basis function
        if self.basis_function == 'wendland':
            phi = self._wendland(r)
        elif self.basis_function == 'gaussian':
            phi = self._gaussian(r)
        elif self.basis_function == 'triangular':
            phi = self._triangular(r)
        else:
            raise ValueError(f"Unknown basis function: {self.basis_function}")
        
        return phi
    
    def _wendland(self, r: torch.Tensor) -> torch.Tensor:
        """
        Wendland C^4 RBF
        φ(r) = (1-r)^6_+ * (35*r^2 + 18*r + 3) / 3 for r in [0,1]
        
        Compact support: [0, 1]
        C^4 continuous (4 times continuously differentiable)
        """
        r = r.clamp(max=1.0)
        return torch.pow(1 - r, 6) * (35 * r**2 + 18 * r + 3) / 3
    
    def _gaussian(self, r: torch.Tensor) -> torch.Tensor:
        """
        Gaussian RBF
        φ(r) = exp(-0.5 * r^2)
        
        Infinite support but rapid decay
        C^∞ continuous (infinitely differentiable)
        """
        return torch.exp(-0.5 * r ** 2)
    
    def _triangular(self, r: torch.Tensor) -> torch.Tensor:
        """
        Triangular (Linear) basis
        φ(r) = (1-r)_+ for r in [0,1]
        
        Compact support: [0, 1]
        C^0 continuous (continuous but not differentiable at r=1)
        """
        return torch.clamp(1 - r, min=0.0)
    
    def compute_domain_penalty(self, domain_bounds=(0.0, 1.0)):
        """
        Compute L2 penalty for centers outside the domain [0,1]^2
        
        Only applies penalty to centers that violate boundaries:
        - penalty = sum of squared distances from boundary for out-of-bound centers
        - No penalty for centers inside domain
        
        Args:
            domain_bounds: (min, max) tuple for domain boundaries
            
        Returns:
            penalty: scalar tensor (0 if all centers are inside domain)
        """
        if not self.learnable:
            return torch.tensor(0.0, device=self.centers.device)
        
        min_bound, max_bound = domain_bounds
        
        # Compute violations for each dimension
        # Lower bound violations: max(0, min_bound - centers)
        lower_violations = torch.clamp(min_bound - self.centers, min=0.0)
        
        # Upper bound violations: max(0, centers - max_bound)
        upper_violations = torch.clamp(self.centers - max_bound, min=0.0)
        
        # Total violation per center (L2 distance from boundary)
        violations = lower_violations + upper_violations  # (k, 2)
        
        # L2 penalty: sum of squared violations
        penalty = torch.sum(violations ** 2)
        
        return penalty
    
    def compute_movement_penalty(self):
        """
        Compute L2 penalty for centers moving away from initial positions
        
        Penalty = sum of squared distances from initial positions
        This encourages centers to stay close to their initialization
        
        Returns:
            penalty: scalar tensor (0 if centers haven't moved or not learnable)
        """
        if not self.learnable:
            return torch.tensor(0.0, device=self.centers.device)
        
        # Compute L2 distance from initial positions
        movement = self.centers - self.centers_init  # (k, 2)
        
        # Sum of squared movements
        penalty = torch.sum(movement ** 2)
        
        return penalty


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
                 spatial_basis_function: str = 'wendland',
                 train_coords: np.ndarray = None):
        super().__init__()
        
        self.p = p
        self.k_spatial_centers = k_spatial_centers
        self.spatial_init_method = spatial_init_method
        self.spatial_basis_function = spatial_basis_function
        
        # Spatial basis embedding
        self.spatial_basis = SpatialBasisEmbedding(
            n_centers=k_spatial_centers,
            learnable=spatial_learnable,
            init_method=spatial_init_method,
            train_coords=train_coords,
            basis_function=spatial_basis_function
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
    
    def compute_domain_penalty(self):
        """
        Compute penalty for basis centers outside domain [0,1]^2
        
        Returns:
            penalty: scalar tensor
        """
        return self.spatial_basis.compute_domain_penalty()
    
    def compute_movement_penalty(self):
        """
        Compute penalty for basis centers moving away from initial positions
        
        Returns:
            penalty: scalar tensor
        """
        return self.spatial_basis.compute_movement_penalty()
    
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
        spatial_basis_function=config.get('spatial_basis_function', 'wendland'),
        train_coords=train_coords
    )
