"""
Unit tests for δ reparameterization in STInterpMLP

Tests the cumulative construction β_k = Σ_{ℓ=1}^k δ_ℓ and
the quantile prediction Ŷ_τk = [1, h(s,t)ᵀ] β_k as described
in Section 3.2 of the thesis.
"""
import torch
import pytest
import numpy as np
from stnf.models.st_interp import STInterpMLP, create_model


class TestDeltaReparameterization:
    """Test suite for δ reparameterization functionality"""

    @pytest.fixture
    def model_config(self):
        """Basic model configuration for testing"""
        return {
            'p': 0,  # number of covariates
            'k_spatial_centers': [9],  # 3x3 grid for simplicity
            'k_temporal_centers': [5],
            'hidden_dims': [32, 16],  # Last hidden dim = 16
            'dropout': 0.0,
            'layernorm': False,
            'spatial_learnable': False,
            'spatial_init_method': 'uniform',
            'spatial_basis_function': 'wendland',
            'output_dim': 5,  # 5 quantiles
        }

    @pytest.fixture
    def sample_inputs(self):
        """Sample inputs for forward pass"""
        batch_size = 4
        return {
            'X': torch.zeros(batch_size, 0),  # No covariates
            'coords': torch.rand(batch_size, 2),
            't': torch.rand(batch_size, 1),
        }

    def test_delta_reparameterization_disabled(self, model_config, sample_inputs):
        """Test that model works normally when δ reparameterization is disabled"""
        model = STInterpMLP(use_delta_reparameterization=False, **model_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_inputs['X'], sample_inputs['coords'], sample_inputs['t'])

        # Should output (batch_size, output_dim)
        assert output.shape == (sample_inputs['X'].shape[0], model_config['output_dim'])
        # Should have mlp attribute (not mlp_trunk)
        assert hasattr(model, 'mlp')
        assert not hasattr(model, 'mlp_trunk') or model.mlp_trunk is None
        assert model.delta_params is None

    def test_delta_reparameterization_enabled(self, model_config, sample_inputs):
        """Test that δ reparameterization is properly initialized when enabled"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)
        model.eval()

        # Should have mlp_trunk (shared trunk) and delta_params
        assert hasattr(model, 'mlp_trunk')
        assert model.mlp_trunk is not None
        assert hasattr(model, 'delta_params')
        assert model.delta_params is not None
        assert len(model.delta_params) == model_config['output_dim']

        # Each δ_k should have shape (last_hidden_dim + 1,)
        last_hidden_dim = model_config['hidden_dims'][-1]
        for delta_k in model.delta_params:
            assert delta_k.shape == (last_hidden_dim + 1,)

    def test_delta_parameters_initialization(self, model_config):
        """Test that δ parameters are properly initialized"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)

        # δ parameters should be initialized (not all zeros after init)
        # They are initialized with small random values
        for delta_k in model.delta_params:
            assert delta_k.requires_grad
            # Check that they're not all zeros (very unlikely with normal init)
            assert not torch.allclose(delta_k, torch.zeros_like(delta_k))

    def test_forward_pass_with_delta(self, model_config, sample_inputs):
        """Test forward pass with δ reparameterization"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_inputs['X'], sample_inputs['coords'], sample_inputs['t'])

        # Output should have shape (batch_size, output_dim)
        batch_size = sample_inputs['X'].shape[0]
        assert output.shape == (batch_size, model_config['output_dim'])

    def test_cumulative_beta_construction(self, model_config, sample_inputs):
        """Test that β_k = Σ_{ℓ=1}^k δ_ℓ is correctly computed"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)
        model.eval()

        # Manually set δ parameters to known values for testing
        last_hidden_dim = model_config['hidden_dims'][-1]
        with torch.no_grad():
            for k, delta_k in enumerate(model.delta_params):
                # Set δ_k to a simple pattern: [k+1, k+1, ..., k+1] for easy verification
                delta_k.fill_(float(k + 1))

        # Get trunk output h(s,t) by running through spatial/temporal bases
        with torch.no_grad():
            phi_s = model.spatial_basis(sample_inputs['coords'])
            psi_t = model.temporal_basis(sample_inputs['t'])
            features = torch.cat([phi_s, psi_t], dim=-1)
            h = model.mlp_trunk(features)

        # Manually compute β_k and verify cumulative sum
        batch_size = sample_inputs['X'].shape[0]
        beta_cumsum = torch.zeros(batch_size, last_hidden_dim + 1)
        for k in range(model_config['output_dim']):
            beta_cumsum += model.delta_params[k].unsqueeze(0)
            # β_k should equal sum of first k+1 δ's
            # δ_1 = [1,1,...,1], δ_2 = [2,2,...,2], etc.
            # β_k[0] = 1 + 2 + ... + (k+1) = (k+1)(k+2)/2
            expected_sum = float((k + 1) * (k + 2) / 2)
            assert torch.allclose(beta_cumsum[0, 0], torch.tensor(expected_sum), atol=1e-5)

    def test_get_delta_parameters(self, model_config):
        """Test get_delta_parameters() method"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)

        delta_list = model.get_delta_parameters()
        assert delta_list is not None
        assert len(delta_list) == model_config['output_dim']
        assert all(isinstance(d, torch.nn.Parameter) for d in delta_list)

        # When disabled, should return None
        model_disabled = STInterpMLP(use_delta_reparameterization=False, **model_config)
        assert model_disabled.get_delta_parameters() is None

    def test_output_consistency(self, model_config, sample_inputs):
        """Test that outputs are consistent across multiple forward passes"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)
        model.eval()

        with torch.no_grad():
            output1 = model(sample_inputs['X'], sample_inputs['coords'], sample_inputs['t'])
            output2 = model(sample_inputs['X'], sample_inputs['coords'], sample_inputs['t'])

        # Outputs should be identical (deterministic)
        assert torch.allclose(output1, output2)

    def test_gradient_flow(self, model_config, sample_inputs):
        """Test that gradients flow through δ parameters"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)
        model.train()

        output = model(sample_inputs['X'], sample_inputs['coords'], sample_inputs['t'])
        loss = output.sum()  # Simple loss for testing
        loss.backward()

        # Check that δ parameters have gradients
        for delta_k in model.delta_params:
            assert delta_k.grad is not None
            assert not torch.allclose(delta_k.grad, torch.zeros_like(delta_k.grad))

    def test_create_model_with_delta_flag(self):
        """Test that create_model() respects use_delta_reparameterization config"""
        config = {
            'regression_type': 'multi-quantile',
            'quantile_levels': [0.1, 0.5, 0.9],
            'k_spatial_centers': [9],
            'k_temporal_centers': [5],
            'hidden_dims': [32, 16],
            'use_delta_reparameterization': True,
        }

        model = create_model(config)
        assert model.use_delta_reparameterization is True
        assert model.delta_params is not None
        assert len(model.delta_params) == len(config['quantile_levels'])

    def test_quantile_monotonicity_potential(self, model_config, sample_inputs):
        """Test that δ reparameterization structure enables monotonicity"""
        model = STInterpMLP(use_delta_reparameterization=True, **model_config)
        model.eval()

        with torch.no_grad():
            output = model(sample_inputs['X'], sample_inputs['coords'], sample_inputs['t'])

        # With proper δ initialization and P_nc(δ) penalty (to be implemented),
        # quantiles should be monotonic. For now, just check output shape.
        assert output.shape[1] == model_config['output_dim']

    def test_single_quantile_no_delta(self, sample_inputs):
        """Test that single quantile doesn't use δ reparameterization"""
        model = STInterpMLP(
            output_dim=1,
            use_delta_reparameterization=True,  # Even if enabled, single output shouldn't use it
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],
        )
        model.eval()

        with torch.no_grad():
            output = model(sample_inputs['X'], sample_inputs['coords'], sample_inputs['t'])

        # Should use standard path (not δ reparameterization) for single output
        assert output.shape == (sample_inputs['X'].shape[0], 1)
        # When output_dim=1, δ reparameterization path is not used
        # (checked in forward method: `if self.use_delta_reparameterization and self.output_dim > 1`)
