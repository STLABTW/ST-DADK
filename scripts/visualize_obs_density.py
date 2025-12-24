"""
Visualize observation density maps for different sampling scenarios.
Shows how frequently each location is included in the training set.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_spatial_obs_prob_fn(pattern='uniform', intensity=1.0):
    """
    Create spatial observation probability function
    
    Args:
        pattern: 'uniform', 'corner', or custom function
        intensity: intensity parameter controlling the degree of non-uniformity
    
    Returns:
        obs_prob_fn: function(coord) -> probability
    """
    if pattern == 'uniform' or pattern is None:
        return None
    
    elif pattern == 'corner':
        # Heavy-tailed distribution with sharp peak at (0,0)
        def obs_prob_fn(coord):
            x, y = coord
            dist_sq = x**2 + y**2
            prob = 1.0 / (1.0 + intensity * dist_sq)**2
            return prob
        return obs_prob_fn
    
    else:
        raise ValueError(f"Unknown pattern: {pattern}")


def sample_observations(z_data, coords, obs_method='site-wise', obs_ratio=0.5, 
                       obs_prob_fn=None, seed=None):
    """
    Sample observations from full data
    
    Args:
        z_data: (T, S) - full spatio-temporal data
        coords: (S, 2) - spatial coordinates in [0,1]^2
        obs_method: 'site-wise' or 'random'
        obs_ratio: observation ratio (used as base rate or for site-wise)
        obs_prob_fn: function(coords) -> probs
        seed: random seed
    
    Returns:
        obs_mask: (T, S) boolean array indicating observed locations
        obs_sites: list of observed site indices (for site-wise method)
    """
    if seed is not None:
        np.random.seed(seed)
    
    T, S = z_data.shape
    
    # Compute observation probabilities per site
    if obs_prob_fn is not None:
        obs_weights = np.array([obs_prob_fn(coords[i]) for i in range(S)])
        obs_weights_normalized = obs_weights / obs_weights.mean()
        obs_probs = obs_weights_normalized * obs_ratio
        obs_probs = np.clip(obs_probs, 0, 1)
    else:
        obs_probs = np.ones(S) * obs_ratio
    
    if obs_method == 'site-wise':
        n_obs_sites = int(S * obs_ratio)
        obs_weights_normalized = obs_probs / obs_probs.sum()
        obs_sites = np.random.choice(S, size=n_obs_sites, replace=False, p=obs_weights_normalized)
        
        obs_mask = np.zeros((T, S), dtype=bool)
        obs_mask[:, obs_sites] = True
        
        return obs_mask, obs_sites
    
    elif obs_method == 'random':
        obs_probs_expanded = obs_probs[np.newaxis, :].repeat(T, axis=0)
        obs_mask = np.random.rand(T, S) < obs_probs_expanded
        obs_sites = None
        
        return obs_mask, obs_sites
    
    else:
        raise ValueError(f"Unknown obs_method: {obs_method}")


def compute_observation_density(z_data, coords, obs_method, spatial_pattern, 
                               obs_ratio=0.1, intensity=10.0, seed=42):
    """
    Compute observation density for a single experiment.
    
    Args:
        z_data: (T, S) array
        coords: (S, 2) array
        obs_method: 'site-wise' or 'random'
        spatial_pattern: 'uniform' or 'corner'
        obs_ratio: observation ratio
        intensity: intensity for corner pattern
        seed: random seed
    
    Returns:
        density: (S,) array of observation frequencies (0 to 1)
    """
    T, S = z_data.shape
    
    # Create probability function
    obs_prob_fn = create_spatial_obs_prob_fn(pattern=spatial_pattern, intensity=intensity)
    
    # Sample observations once
    obs_mask, _ = sample_observations(
        z_data, coords, 
        obs_method=obs_method, 
        obs_ratio=obs_ratio,
        obs_prob_fn=obs_prob_fn,
        seed=seed
    )
    
    if obs_method == 'site-wise':
        # For site-wise: selected sites have density=1.0, others=0.0
        density = obs_mask.any(axis=0).astype(float)  # (S,)
        
    elif obs_method == 'random':
        # For random: compute observation ratio per site
        density = obs_mask.sum(axis=0) / T  # (S,)
    
    return density


def plot_observation_density_maps(data_path, obs_ratio=0.1, intensity=10.0, 
                                  n_samples=100, seed=42, save_path=None):
    """
    Plot 2x2 observation density maps for 4 scenarios.
    
    Args:
        data_path: path to CSV data file (e.g., '2b_8_train.csv')
        obs_ratio: observation ratio
        intensity: intensity for corner pattern
        n_samples: number of sampling runs to compute density
        seed: base random seed
        save_path: path to save figure
    """
    # Load data
    df = pd.read_csv(data_path)
    
    # Get unique coordinates
    coords_df = df[['x', 'y']].drop_duplicates().sort_values(['x', 'y'])
    coords = coords_df.values
    S = len(coords)
    
    # Get temporal data
    T = df['t'].nunique()
    
    # Reshape z_data to (T, S)
    df_sorted = df.sort_values(['t', 'x', 'y'])
    z_data = df_sorted['z'].values.reshape(T, S)
    
    print(f"Data shape: T={T}, S={S}")
    print(f"Observation ratio: {obs_ratio}")
    print(f"Computing densities for single experiment...")
    
    # Define 4 scenarios
    scenarios = [
        {'obs_method': 'site-wise', 'spatial_pattern': 'uniform', 'title': 'Fixed + Uniform'},
        {'obs_method': 'site-wise', 'spatial_pattern': 'corner', 'title': 'Fixed + Clustered'},
        {'obs_method': 'random', 'spatial_pattern': 'uniform', 'title': 'Random + Uniform'},
        {'obs_method': 'random', 'spatial_pattern': 'corner', 'title': 'Random + Clustered'},
    ]
    
    # Create figure with 1x4 layout
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    
    for idx, (ax, scenario) in enumerate(zip(axes, scenarios)):
        print(f"  Scenario {idx+1}: {scenario['title']}")
        
        # Compute density
        density = compute_observation_density(
            z_data, coords,
            obs_method=scenario['obs_method'],
            spatial_pattern=scenario['spatial_pattern'],
            obs_ratio=obs_ratio,
            intensity=intensity,
            seed=seed + idx  # Different seed for each scenario
        )
        
        # Create scatter plot with fixed red color and alpha based on density
        # Alpha: 0 (transparent) to 1 (opaque) based on observation frequency
        scatter = ax.scatter(coords[:, 0], coords[:, 1], 
                           c='red', s=5, alpha=density, vmin=0, vmax=1)
        
        # Set title and labels with increased font sizes (1.5x)
        ax.set_title(scenario['title'], fontsize=27, fontweight='bold')
        ax.set_xlabel('x', fontsize=21)
        ax.set_ylabel('y', fontsize=21)
        ax.tick_params(axis='both', which='major', labelsize=18)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nDensity map saved to {save_path}")
    
    return fig


def main():
    """Main function to create observation density maps."""
    # Setup paths
    data_dir = Path(__file__).parent.parent / 'data' / '2b'
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Parameters
    data_file = '2b_8_train.csv'
    obs_ratio = 0.1
    intensity = 10.0
    seed = 42
    
    print("=" * 60)
    print("Observation Density Visualization")
    print("=" * 60)
    
    data_path = data_dir / data_file
    save_path = output_dir / f'obs_density_2b8_ratio{obs_ratio}.png'
    
    fig = plot_observation_density_maps(
        data_path=data_path,
        obs_ratio=obs_ratio,
        intensity=intensity,
        seed=seed,
        save_path=save_path
    )
    
    print("\nVisualization complete!")
    
    # Show plot
    plt.show()


if __name__ == '__main__':
    main()
