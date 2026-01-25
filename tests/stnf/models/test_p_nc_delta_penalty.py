"""
Unit tests for P_nc(δ) penalty implementation

Tests the parameter-level non-crossing penalty as defined in Section 3.2
(Equation 3.10) of the thesis.
"""
import torch
import pytest
import numpy as np
from scripts.train_st_interp import compute_p_nc_delta_penalty
from stnf.models.st_interp import STInterpMLP


class TestPNCDeltaPenalty:
    """Test suite for P_nc(δ) penalty functionality"""

    def test_compute_p_nc_delta_penalty_basic(self):
        """Test basic computation of P_nc(δ) penalty"""
        # Create 5 quantiles with d=16 (last hidden dim)
        d = 16
        Q = 5
        delta_params = [torch.nn.Parameter(torch.randn(d + 1)) for _ in range(Q)]
        
        penalty = compute_p_nc_delta_penalty(delta_params)
        
        # Should return a scalar tensor
        assert isinstance(penalty, torch.Tensor)
        assert penalty.dim() == 0  # Scalar
        assert torch.isfinite(penalty)

    def test_compute_p_nc_delta_penalty_single_quantile(self):
        """Test that penalty is 0 for single quantile (Q=1)"""
        d = 16
        delta_params = [torch.nn.Parameter(torch.randn(d + 1))]
        
        penalty = compute_p_nc_delta_penalty(delta_params)
        
        # Should be 0 (need at least 2 quantiles)
        assert penalty.item() == 0.0

    def test_compute_p_nc_delta_penalty_none(self):
        """Test that penalty is 0 when delta_params is None"""
        penalty = compute_p_nc_delta_penalty(None)
        assert penalty.item() == 0.0

    def test_compute_p_nc_delta_penalty_empty(self):
        """Test that penalty is 0 for empty list"""
        penalty = compute_p_nc_delta_penalty([])
        assert penalty.item() == 0.0

    def test_j_delta_k_formula(self):
        """Test J(δ_k) formula correctness"""
        d = 4
        # Create δ_k with known values for testing
        # δ_k = [intercept, feat1, feat2, feat3, feat4]
        # Set: intercept=2.0, features=[1.0, -0.5, 0.3, -0.2]
        delta_k = torch.nn.Parameter(torch.tensor([2.0, 1.0, -0.5, 0.3, -0.2]))
        
        delta_k_0 = delta_k[0]  # 2.0
        delta_k_features = delta_k[1:]  # [1.0, -0.5, 0.3, -0.2]
        
        # Compute Σ_{j=1}^d max(0, -δ_k,j)
        # max(0, -1.0) = 0, max(0, -(-0.5)) = max(0, 0.5) = 0.5
        # max(0, -0.3) = 0, max(0, -(-0.2)) = max(0, 0.2) = 0.2
        # Sum = 0 + 0.5 + 0 + 0.2 = 0.7
        negative_features = torch.clamp(-delta_k_features, min=0.0)
        sum_negative = negative_features.sum()
        
        expected_sum = 0.5 + 0.2  # Only negative features contribute
        assert torch.allclose(sum_negative, torch.tensor(expected_sum))
        
        # J(δ_k) = δ_k,0 - max(δ_k,0, Σ_{j=1}^d max(0, -δ_k,j))
        # = 2.0 - max(2.0, 0.7) = 2.0 - 2.0 = 0.0
        max_term = torch.max(delta_k_0, sum_negative)
        J_delta_k = delta_k_0 - max_term
        
        expected_J = 2.0 - max(2.0, 0.7)  # = 0.0
        assert torch.allclose(J_delta_k, torch.tensor(expected_J))

    def test_p_nc_delta_penalty_sum(self):
        """Test that P_nc(δ) = Σ_{k=2}^Q J(δ_k)"""
        d = 4
        Q = 5
        
        # Create δ parameters
        delta_params = [torch.nn.Parameter(torch.randn(d + 1)) for _ in range(Q)]
        
        # Compute penalty
        penalty = compute_p_nc_delta_penalty(delta_params)
        
        # Manually compute expected penalty
        expected_penalty = torch.tensor(0.0, device=delta_params[0].device)
        for k in range(1, Q):  # k=1 to Q-1 (indices 1 to 4)
            delta_k = delta_params[k]
            delta_k_0 = delta_k[0]
            delta_k_features = delta_k[1:]
            negative_features = torch.clamp(-delta_k_features, min=0.0)
            sum_negative = negative_features.sum()
            max_term = torch.max(delta_k_0, sum_negative)
            J_delta_k = delta_k_0 - max_term
            expected_penalty = expected_penalty + J_delta_k
        
        assert torch.allclose(penalty, expected_penalty)

    def test_p_nc_delta_penalty_gradient(self):
        """Test that P_nc(δ) penalty has gradients"""
        d = 8
        Q = 5
        delta_params = [torch.nn.Parameter(torch.randn(d + 1)) for _ in range(Q)]
        
        penalty = compute_p_nc_delta_penalty(delta_params)
        penalty.backward()
        
        # Check that gradients exist on δ parameters
        for k in range(1, Q):  # Only δ_2 to δ_Q should have gradients
            assert delta_params[k].grad is not None
            assert not torch.allclose(delta_params[k].grad, torch.zeros_like(delta_params[k].grad))
        
        # δ_1 should not contribute to penalty (k starts from 1, which is index 1 = δ_2)
        # But let's check if it has gradient (it shouldn't from this penalty, but might from other ops)

    def test_p_nc_delta_penalty_with_model(self):
        """Test P_nc(δ) penalty integration with STInterpMLP model"""
        model = STInterpMLP(
            p=0,
            k_spatial_centers=[9],
            k_temporal_centers=[5],
            hidden_dims=[32, 16],  # Last hidden dim = 16
            output_dim=5,  # 5 quantiles
            use_delta_reparameterization=True
        )
        
        # Get δ parameters
        delta_params = model.get_delta_parameters()
        assert delta_params is not None
        assert len(delta_params) == 5
        
        # Compute penalty
        penalty = compute_p_nc_delta_penalty(delta_params)
        
        # Should be a scalar tensor
        assert isinstance(penalty, torch.Tensor)
        assert penalty.dim() == 0
        assert torch.isfinite(penalty)

    def test_p_nc_delta_penalty_device_consistency(self):
        """Test that penalty is computed on correct device"""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        d = 8
        Q = 5
        delta_params = [torch.nn.Parameter(torch.randn(d + 1, device=device)) for _ in range(Q)]
        
        penalty = compute_p_nc_delta_penalty(delta_params)
        
        assert penalty.device == device

    def test_p_nc_delta_penalty_requires_grad(self):
        """Test that penalty computation preserves requires_grad"""
        d = 8
        Q = 5
        delta_params = [torch.nn.Parameter(torch.randn(d + 1), requires_grad=True) for _ in range(Q)]
        
        penalty = compute_p_nc_delta_penalty(delta_params)
        
        # Penalty should require grad if δ parameters require grad
        assert penalty.requires_grad
