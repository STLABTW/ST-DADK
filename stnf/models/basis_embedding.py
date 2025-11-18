"""
Basis Embedding Layer for spatial coordinates

Wendland radial basis function:
phi(r) = (1-r)^6_+ * (35r^2 + 18r + 3) / 3
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Literal


def wendland_phi(r: torch.Tensor) -> torch.Tensor:
    """
    Wendland radial basis function
    phi(r) = (1-r)^6_+ * (35r^2 + 18r + 3) / 3
    
    Args:
        r: distance values, any shape
        
    Returns:
        phi(r): same shape as r
    """
    # (1-r)_+ means max(0, 1-r)
    pos_part = torch.clamp(1 - r, min=0)
    return (pos_part ** 6) * (35 * r**2 + 18 * r + 3) / 3


class BasisEmbedding(nn.Module):
    """
    Basis embedding layer using Wendland radial basis functions
    
    Maps (x, y) coordinates to k-dimensional basis embedding vectors
    
    Args:
        k: number of basis functions (default: 250)
        basis_initialize: method for initializing centers ('regular' or custom)
        basis_learnable: whether centers and bandwidths are learnable
    """
    def __init__(
        self, 
        k: int = 250,
        basis_initialize: Literal['regular'] = 'regular',
        basis_learnable: bool = False
    ):
        super().__init__()
        self.k = k
        self.basis_initialize = basis_initialize
        self.basis_learnable = basis_learnable
        
        # Initialize centers (u_j) and bandwidths (theta_j)
        if basis_initialize == 'regular':
            centers, bandwidths = self._initialize_regular_grid()
        else:
            raise ValueError(f"Unknown basis_initialize method: {basis_initialize}")
        
        # Register as parameters or buffers
        if basis_learnable:
            self.centers = nn.Parameter(centers)
            self.bandwidths = nn.Parameter(bandwidths)
        else:
            self.register_buffer('centers', centers)
            self.register_buffer('bandwidths', bandwidths)
        
        print(f"[BasisEmbedding] k={k}, initialize={basis_initialize}, learnable={basis_learnable}")
        print(f"  Grid sizes: 5x5 (25), 9x9 (81), 12x12 (144) = {25+81+144} centers")
    
    def _initialize_regular_grid(self):
        """
        Initialize centers on regular grids with corresponding bandwidths
        
        Regular grid setup:
        - 5x5 grid (25 centers): theta = 0.10
        - 9x9 grid (81 centers): theta = 0.15
        - 12x12 grid (144 centers): theta = 0.45
        Total: 250 centers
        
        Returns:
            centers: (k, 2) tensor of center coordinates
            bandwidths: (k,) tensor of bandwidth values
        """
        all_centers = []
        all_bandwidths = []
        
        # Grid configurations: (grid_size, theta)
        grid_configs = [
            (5, 0.10),
            (9, 0.15),
            (12, 0.45)
        ]
        
        for grid_size, theta in grid_configs:
            # Create regular grid in [0, 1]^2
            x = np.linspace(0, 1, grid_size)
            y = np.linspace(0, 1, grid_size)
            xx, yy = np.meshgrid(x, y)
            
            # Flatten to get center coordinates
            centers_grid = np.stack([xx.flatten(), yy.flatten()], axis=1)  # (grid_size^2, 2)
            
            # Create bandwidths for this grid
            bandwidths_grid = np.full(grid_size**2, theta)
            
            all_centers.append(centers_grid)
            all_bandwidths.append(bandwidths_grid)
        
        # Concatenate all grids
        centers = np.concatenate(all_centers, axis=0)  # (k, 2)
        bandwidths = np.concatenate(all_bandwidths, axis=0)  # (k,)
        
        # Convert to tensors
        centers = torch.from_numpy(centers).float()
        bandwidths = torch.from_numpy(bandwidths).float()
        
        return centers, bandwidths
    
    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute basis embeddings for input coordinates
        
        Args:
            coords: (..., 2) tensor of (x, y) coordinates
            
        Returns:
            embeddings: (..., k) tensor of basis function values
        """
        # coords: (..., 2)
        # centers: (k, 2)
        # Need to compute distance from each coord to each center
        
        original_shape = coords.shape[:-1]  # Save original shape
        coords_flat = coords.reshape(-1, 2)  # (N, 2)
        
        # Compute distances: ||coords - center|| for all pairs
        # coords_flat: (N, 2), centers: (k, 2)
        # Expand: coords_flat: (N, 1, 2), centers: (1, k, 2)
        coords_expanded = coords_flat.unsqueeze(1)  # (N, 1, 2)
        centers_expanded = self.centers.unsqueeze(0)  # (1, k, 2)
        
        # Euclidean distance
        distances = torch.norm(coords_expanded - centers_expanded, dim=2)  # (N, k)
        
        # Normalize by bandwidth: r = distance / theta
        bandwidths_expanded = self.bandwidths.unsqueeze(0)  # (1, k)
        r = distances / bandwidths_expanded  # (N, k)
        
        # Apply Wendland basis function
        phi_values = wendland_phi(r)  # (N, k)
        
        # Reshape back to original shape
        embeddings = phi_values.reshape(*original_shape, self.k)
        
        return embeddings


if __name__ == '__main__':
    # Test
    basis_emb = BasisEmbedding(k=250, basis_initialize='regular', basis_learnable=False)
    
    # Test input
    coords = torch.rand(10, 100, 2)  # (B, S, 2)
    output = basis_emb(coords)
    
    print(f"Input shape: {coords.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test at center point
    test_coord = torch.tensor([[0.5, 0.5]])  # Center of [0,1]^2
    test_output = basis_emb(test_coord)
    print(f"\nTest at (0.5, 0.5): {test_output.shape}")
    print(f"Non-zero values: {(test_output > 1e-6).sum().item()} / {250}")
