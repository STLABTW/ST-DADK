"""
Visualize 2b-7, 2b-8, 2b-9 datasets:
- Spatial maps at t=1 (1x3 subplots)
- Temporal line plots at a specific location (3x1 subplots)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def load_data(data_dir, file_names):
    """Load multiple CSV files."""
    data_dict = {}
    for name in file_names:
        file_path = Path(data_dir) / name
        df = pd.read_csv(file_path)
        data_dict[name] = df
    return data_dict


def plot_spatial_maps_t50(data_dict, file_names, save_path=None):
    """
    Plot spatial maps at t=50 for multiple files.
    Creates 1x3 subplot layout.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with file names as keys and dataframes as values
    file_names : list
        List of file names to plot
    save_path : str or Path, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, (ax, fname) in enumerate(zip(axes, file_names)):
        df = data_dict[fname]

        # Filter data at t=50
        df_t50 = df[df['t'] == 50].copy()

        # Create scatter plot
        scatter = ax.scatter(df_t50['x'], df_t50['y'], c=df_t50['z'], 
                           cmap='RdBu_r', s=10, alpha=0.8)
        
        # Set title and labels
        dataset_name = fname.replace('_train.csv', '').replace('_', '-')
        ax.set_title(f'{dataset_name} at t=50', fontsize=16, fontweight='bold')
        ax.set_xlabel('x', fontsize=14)
        ax.set_ylabel('y', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.set_aspect('equal')
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('z', fontsize=14)
        cbar.ax.tick_params(labelsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Spatial maps saved to {save_path}")
    
    return fig


def plot_temporal_lines(data_dict, file_names, x_coord=0.5, y_coord=0.5, 
                        tolerance=0.05, save_path=None):
    """
    Plot temporal line plots at a specific location for multiple files.
    Creates 3x1 subplot layout.
    
    Parameters:
    -----------
    data_dict : dict
        Dictionary with file names as keys and dataframes as values
    file_names : list
        List of file names to plot
    x_coord : float
        Target x coordinate
    y_coord : float
        Target y coordinate
    tolerance : float
        Tolerance for finding nearby points
    save_path : str or Path, optional
        Path to save the figure
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))
    
    for idx, (ax, fname) in enumerate(zip(axes, file_names)):
        df = data_dict[fname]
        
        # Find the closest location to target coordinates
        df_with_dist = df.copy()
        df_with_dist['dist'] = np.sqrt((df['x'] - x_coord)**2 + (df['y'] - y_coord)**2)
        
        # Get the closest unique location
        closest_point = df_with_dist.groupby(['x', 'y'])['dist'].first().idxmin()
        x_closest, y_closest = closest_point
        
        # Filter data at this location
        df_location = df[(df['x'] == x_closest) & (df['y'] == y_closest)].copy()
        df_location = df_location.sort_values('t')
        
        # Plot temporal evolution
        ax.plot(df_location['t'], df_location['z'], 
               linewidth=2, marker='o', markersize=4, alpha=0.7)
        
        # Set title and labels
        dataset_name = fname.replace('_train.csv', '').replace('_', '-')
        ax.set_title(f'{dataset_name} at (x={x_closest:.3f}, y={y_closest:.3f})', 
                    fontsize=16, fontweight='bold')
        ax.set_xlabel('Time (t)', fontsize=14)
        ax.set_ylabel('z', fontsize=14)
        ax.tick_params(axis='both', which='major', labelsize=12)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Temporal plots saved to {save_path}")
    
    return fig


def main():
    """Main function to create all visualizations."""
    # Setup paths
    data_dir = Path(__file__).parent.parent / 'data' / '2b'
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(exist_ok=True)
    
    # Files to visualize
    file_names = ['2b_7_train.csv', '2b_8_train.csv', '2b_9_train.csv']
    
    print("Loading data...")
    data_dict = load_data(data_dir, file_names)

    print("\nCreating spatial maps at t=50...")
    fig_spatial = plot_spatial_maps_t50(
        data_dict,
        file_names,
        save_path=output_dir / '2b_spatial_maps_t50.png'
    )
    
    print("\nCreating temporal line plots...")
    # You can adjust the coordinates here
    fig_temporal = plot_temporal_lines(
        data_dict,
        file_names,
        x_coord=0.5,  # Center of domain
        y_coord=0.5,
        save_path=output_dir / '2b_temporal_lines.png'
    )
    
    print("\nVisualization complete!")
    print(f"Results saved to {output_dir}")
    
    # Show plots
    plt.show()


if __name__ == '__main__':
    main()
