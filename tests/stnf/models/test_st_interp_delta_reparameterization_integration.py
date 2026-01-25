"""
Integration tests for δ reparameterization using real KAUST data

Tests the δ reparameterization functionality with actual data from
kaust_loader.py to ensure it works correctly in realistic scenarios.
"""
import torch
import pytest
import numpy as np
from pathlib import Path
from stnf.models.st_interp import STInterpMLP, create_model
from stnf.dataio.kaust_loader import load_kaust_csv_single


class TestDeltaReparameterizationIntegration:
    """Integration tests using real KAUST data"""

    @pytest.fixture
    def data_path(self):
        """Path to test data file"""
        # Use a small dataset for testing
        data_dir = Path(__file__).parent.parent.parent.parent / "data" / "2b"
        data_file = data_dir / "2b_8.csv"
        
        # Skip test if data file doesn't exist
        if not data_file.exists():
            pytest.skip(f"Test data file not found: {data_file}")
        
        return str(data_file)

    @pytest.fixture
    def sample_data(self, data_path):
        """Load sample data from KAUST CSV"""
        z_data, coords, metadata = load_kaust_csv_single(data_path, normalize=True)
        
        # Use a small subset for faster testing
        T, S = z_data.shape
        # Use first 10 time points and first 20 sites
        z_subset = z_data[:10, :20]
        coords_subset = coords[:20]
        
        return {
            'z_data': z_subset,
            'coords': coords_subset,
            'metadata': metadata,
            'T': z_subset.shape[0],
            'S': z_subset.shape[1]
        }

    def test_model_with_real_data_delta_enabled(self, sample_data):
        """Test model forward pass with real data when δ reparameterization is enabled"""
        T, S = sample_data['T'], sample_data['S']
        coords = sample_data['coords']
        
        # Create model with δ reparameterization
        model = STInterpMLP(
            p=0,
            k_spatial_centers=[9],  # 3x3 grid
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
            dropout=0.0,
            layernorm=False,
            spatial_learnable=False,
            spatial_init_method='uniform',
            spatial_basis_function='wendland',
            output_dim=5,  # 5 quantiles
            use_delta_reparameterization=True
        )
        model.eval()

        # Create sample inputs
        batch_size = 4
        coords_tensor = torch.from_numpy(coords[:batch_size]).float()
        t_tensor = torch.rand(batch_size, 1)
        X_tensor = torch.zeros(batch_size, 0)

        with torch.no_grad():
            output = model(X_tensor, coords_tensor, t_tensor)

        # Verify output shape
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_model_with_real_data_delta_disabled(self, sample_data):
        """Test model forward pass with real data when δ reparameterization is disabled"""
        T, S = sample_data['T'], sample_data['S']
        coords = sample_data['coords']
        
        # Create model without δ reparameterization
        model = STInterpMLP(
            p=0,
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
            dropout=0.0,
            layernorm=False,
            spatial_learnable=False,
            spatial_init_method='uniform',
            spatial_basis_function='wendland',
            output_dim=5,
            use_delta_reparameterization=False
        )
        model.eval()

        # Create sample inputs
        batch_size = 4
        coords_tensor = torch.from_numpy(coords[:batch_size]).float()
        t_tensor = torch.rand(batch_size, 1)
        X_tensor = torch.zeros(batch_size, 0)

        with torch.no_grad():
            output = model(X_tensor, coords_tensor, t_tensor)

        # Verify output shape
        assert output.shape == (batch_size, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_delta_vs_standard_output_difference(self, sample_data):
        """Test that δ reparameterization produces different outputs than standard method"""
        coords = sample_data['coords']
        
        # Create two models with same initialization seed
        torch.manual_seed(42)
        model_delta = STInterpMLP(
            p=0,
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
            dropout=0.0,
            layernorm=False,
            spatial_learnable=False,
            spatial_init_method='uniform',
            spatial_basis_function='wendland',
            output_dim=5,
            use_delta_reparameterization=True
        )
        
        torch.manual_seed(42)
        model_standard = STInterpMLP(
            p=0,
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
            dropout=0.0,
            layernorm=False,
            spatial_learnable=False,
            spatial_init_method='uniform',
            spatial_basis_function='wendland',
            output_dim=5,
            use_delta_reparameterization=False
        )
        
        model_delta.eval()
        model_standard.eval()

        # Same inputs
        batch_size = 4
        coords_tensor = torch.from_numpy(coords[:batch_size]).float()
        t_tensor = torch.rand(batch_size, 1)
        X_tensor = torch.zeros(batch_size, 0)

        with torch.no_grad():
            output_delta = model_delta(X_tensor, coords_tensor, t_tensor)
            output_standard = model_standard(X_tensor, coords_tensor, t_tensor)

        # Outputs should be different (different architectures)
        assert not torch.allclose(output_delta, output_standard, atol=1e-5)

    def test_delta_reparameterization_with_all_sites(self, sample_data):
        """Test δ reparameterization with all sites from real data"""
        coords = sample_data['coords']
        S = len(coords)
        
        model = STInterpMLP(
            p=0,
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
            dropout=0.0,
            layernorm=False,
            spatial_learnable=False,
            spatial_init_method='uniform',
            spatial_basis_function='wendland',
            output_dim=5,
            use_delta_reparameterization=True
        )
        model.eval()

        # Use all sites
        coords_tensor = torch.from_numpy(coords).float()
        t_tensor = torch.rand(S, 1)
        X_tensor = torch.zeros(S, 0)

        with torch.no_grad():
            output = model(X_tensor, coords_tensor, t_tensor)

        # Verify output shape: (S, 5)
        assert output.shape == (S, 5)
        assert not torch.isnan(output).any()
        assert not torch.isinf(output).any()

    def test_delta_parameters_gradient_with_real_data(self, sample_data):
        """Test that gradients flow correctly with real data"""
        coords = sample_data['coords']
        
        model = STInterpMLP(
            p=0,
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
            dropout=0.0,
            layernorm=False,
            spatial_learnable=False,
            spatial_init_method='uniform',
            spatial_basis_function='wendland',
            output_dim=5,
            use_delta_reparameterization=True
        )
        model.train()

        # Create inputs
        batch_size = 4
        coords_tensor = torch.from_numpy(coords[:batch_size]).float()
        t_tensor = torch.rand(batch_size, 1)
        X_tensor = torch.zeros(batch_size, 0)

        # Forward and backward
        output = model(X_tensor, coords_tensor, t_tensor)
        loss = output.sum()
        loss.backward()

        # Check gradients on δ parameters
        delta_params = model.get_delta_parameters()
        assert delta_params is not None
        
        for delta_k in delta_params:
            assert delta_k.grad is not None
            assert not torch.allclose(delta_k.grad, torch.zeros_like(delta_k.grad))

    def test_create_model_with_real_data_config(self, sample_data):
        """Test create_model() with real data and δ reparameterization config"""
        config = {
            'regression_type': 'multi-quantile',
            'quantile_levels': [0.05, 0.25, 0.5, 0.75, 0.95],
            'k_spatial_centers': [9],
            'k_temporal_centers': [5],
            'hidden_dims': [32, 16],
            'use_delta_reparameterization': True,
            'p_covariates': 0,
        }

        model = create_model(config, train_coords=sample_data['coords'])
        assert model.use_delta_reparameterization is True
        assert model.delta_params is not None
        assert len(model.delta_params) == 5

        # Test forward pass
        model.eval()
        batch_size = 4
        coords_tensor = torch.from_numpy(sample_data['coords'][:batch_size]).float()
        t_tensor = torch.rand(batch_size, 1)
        X_tensor = torch.zeros(batch_size, 0)

        with torch.no_grad():
            output = model(X_tensor, coords_tensor, t_tensor)

        assert output.shape == (batch_size, 5)

    def test_delta_reparameterization_output_range(self, sample_data):
        """Test that δ reparameterization outputs are in reasonable range"""
        coords = sample_data['coords']
        
        model = STInterpMLP(
            p=0,
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
            dropout=0.0,
            layernorm=False,
            spatial_learnable=False,
            spatial_init_method='uniform',
            spatial_basis_function='wendland',
            output_dim=5,
            use_delta_reparameterization=True
        )
        model.eval()

        batch_size = 4
        coords_tensor = torch.from_numpy(coords[:batch_size]).float()
        t_tensor = torch.rand(batch_size, 1)
        X_tensor = torch.zeros(batch_size, 0)

        with torch.no_grad():
            output = model(X_tensor, coords_tensor, t_tensor)

        # Outputs should be finite and not extremely large
        assert torch.isfinite(output).all()
        # With normalized data and small initialization, outputs should be reasonable
        assert output.abs().max() < 100.0  # Reasonable upper bound
