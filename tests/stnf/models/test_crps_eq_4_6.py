"""
Unit tests for CRPS computation using Equation 4.6

Tests the CRPS implementation that matches Equation 4.6 from the thesis:
CRPS(F, y) = 2 * Σ_k w_k ρ_{τ_k}(y - Q_{τ_k})
"""
import numpy as np
import pytest
from scripts.train_st_interp import (
    compute_crps,
    compute_crps_multi_quantile,
    check_loss_numpy
)


class TestCRPSEquation46:
    """Test suite for CRPS computation using Equation 4.6"""

    def test_check_loss_numpy_basic(self):
        """Test basic check loss computation"""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.5, 2.5, 3.5])
        quantile = 0.5
        
        loss = check_loss_numpy(y_pred, y_true, quantile)
        
        # For quantile 0.5, check loss = mean(|y_true - y_pred|) / 2
        # errors = [0.5, 0.5, 0.5]
        # max((0.5-1)*0.5, 0.5*0.5) = max(-0.25, 0.25) = 0.25 for each
        expected = 0.25
        assert np.allclose(loss, expected)

    def test_check_loss_numpy_quantile_0_1(self):
        """Test check loss for quantile 0.1"""
        y_pred = np.array([1.0, 2.0, 3.0])
        y_true = np.array([1.5, 2.5, 3.5])
        quantile = 0.1
        
        loss = check_loss_numpy(y_pred, y_true, quantile)
        
        # errors = [0.5, 0.5, 0.5]
        # max((0.1-1)*0.5, 0.1*0.5) = max(-0.45, 0.05) = 0.05 for each
        expected = 0.05
        assert np.allclose(loss, expected)

    def test_compute_crps_single_quantile(self):
        """Test CRPS for single quantile"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        predictions_dict = {0.5: np.array([1.5, 2.5, 3.5, 4.5, 5.5])}
        
        crps = compute_crps(predictions_dict, y_true)
        
        # Should be 2 * check_loss
        expected_check_loss = check_loss_numpy(
            predictions_dict[0.5], y_true, 0.5
        )
        expected_crps = 2.0 * expected_check_loss
        
        assert np.allclose(crps, expected_crps)

    def test_compute_crps_multiple_quantiles_uniform_weights(self):
        """Test CRPS for multiple quantiles with uniform weights (default)"""
        y_true = np.array([2.0, 3.0, 4.0])
        predictions_dict = {
            0.05: np.array([1.0, 2.0, 3.0]),
            0.25: np.array([1.5, 2.5, 3.5]),
            0.5: np.array([2.0, 3.0, 4.0]),
            0.75: np.array([2.5, 3.5, 4.5]),
            0.95: np.array([3.0, 4.0, 5.0])
        }
        
        crps = compute_crps(predictions_dict, y_true)
        
        # Manual calculation using Equation 4.6
        K = len(predictions_dict)
        weights = np.ones(K) / K  # Uniform weights
        manual_sum = 0.0
        for i, (q, pred) in enumerate(sorted(predictions_dict.items())):
            check_loss_q = check_loss_numpy(pred, y_true, q)
            manual_sum += weights[i] * check_loss_q
        expected_crps = 2.0 * manual_sum
        
        assert np.allclose(crps, expected_crps)

    def test_compute_crps_custom_weights(self):
        """Test CRPS with custom quadrature weights"""
        y_true = np.array([2.0, 3.0])
        predictions_dict = {
            0.25: np.array([1.5, 2.5]),
            0.5: np.array([2.0, 3.0]),
            0.75: np.array([2.5, 3.5])
        }
        custom_weights = np.array([0.2, 0.5, 0.3])  # Non-uniform
        
        crps = compute_crps(predictions_dict, y_true, weights=custom_weights)
        
        # Manual calculation
        quantiles = sorted(predictions_dict.keys())
        normalized_weights = custom_weights / custom_weights.sum()
        manual_sum = 0.0
        for i, q in enumerate(quantiles):
            pred = predictions_dict[q]
            check_loss_q = check_loss_numpy(pred, y_true, q)
            manual_sum += normalized_weights[i] * check_loss_q
        expected_crps = 2.0 * manual_sum
        
        assert np.allclose(crps, expected_crps)

    def test_compute_crps_multi_quantile(self):
        """Test compute_crps_multi_quantile function"""
        quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        preds = np.array([
            [1.0, 1.5, 2.0, 2.5, 3.0],
            [2.0, 2.5, 3.0, 3.5, 4.0],
            [3.0, 3.5, 4.0, 4.5, 5.0]
        ])
        y_true = np.array([2.0, 3.0, 4.0])
        
        crps_multi = compute_crps_multi_quantile(preds, y_true, quantile_levels)
        
        # Convert to dict format and compute using compute_crps
        predictions_dict = {}
        for i, q in enumerate(quantile_levels):
            predictions_dict[q] = preds[:, i]
        crps_dict = compute_crps(predictions_dict, y_true)
        
        assert np.allclose(crps_multi, crps_dict)

    def test_compute_crps_multi_quantile_2d_y_true(self):
        """Test compute_crps_multi_quantile with 2D y_true"""
        quantile_levels = [0.5, 0.9]
        preds = np.array([
            [1.0, 1.5],
            [2.0, 2.5]
        ])
        y_true = np.array([[2.0], [3.0]])  # 2D
        
        crps = compute_crps_multi_quantile(preds, y_true, quantile_levels)
        
        # Should work the same as 1D
        y_true_1d = np.array([2.0, 3.0])
        crps_1d = compute_crps_multi_quantile(preds, y_true_1d, quantile_levels)
        
        assert np.allclose(crps, crps_1d)

    def test_compute_crps_empty_dict(self):
        """Test that empty predictions_dict raises error"""
        y_true = np.array([1.0, 2.0])
        predictions_dict = {}
        
        with pytest.raises(ValueError, match="cannot be empty"):
            compute_crps(predictions_dict, y_true)

    def test_compute_crps_weight_length_mismatch(self):
        """Test that weight length mismatch raises error"""
        y_true = np.array([1.0, 2.0])
        predictions_dict = {
            0.5: np.array([1.5, 2.5]),
            0.9: np.array([2.0, 3.0])
        }
        weights = np.array([0.5])  # Wrong length
        
        with pytest.raises(ValueError, match="weights length"):
            compute_crps(predictions_dict, y_true, weights=weights)

    def test_compute_crps_equation_4_6_verification(self):
        """Verify CRPS matches Equation 4.6 exactly"""
        # Use the 5 quantiles from thesis Section 4.2.2
        quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
        y_true = np.array([2.0, 3.0, 4.0, 5.0])
        
        # Create predictions
        predictions_dict = {
            0.05: np.array([1.0, 2.0, 3.0, 4.0]),
            0.25: np.array([1.5, 2.5, 3.5, 4.5]),
            0.5: np.array([2.0, 3.0, 4.0, 5.0]),
            0.75: np.array([2.5, 3.5, 4.5, 5.5]),
            0.95: np.array([3.0, 4.0, 5.0, 6.0])
        }
        
        crps = compute_crps(predictions_dict, y_true)
        
        # Verify using Equation 4.6: CRPS(F, y) = 2 * Σ_k w_k ρ_{τ_k}(y - Q_{τ_k})
        K = len(quantile_levels)
        w_k = 1.0 / K  # Uniform weights
        sum_term = 0.0
        for q in quantile_levels:
            pred = predictions_dict[q]
            rho_tau_k = check_loss_numpy(pred, y_true, q)
            sum_term += w_k * rho_tau_k
        
        expected_crps = 2.0 * sum_term
        
        assert np.allclose(crps, expected_crps)

    def test_compute_crps_scale_factor_2x(self):
        """Verify that CRPS includes the 2× scaling factor from Equation 4.6"""
        y_true = np.array([2.0])
        predictions_dict = {
            0.5: np.array([2.5])
        }
        
        crps = compute_crps(predictions_dict, y_true)
        
        # Without 2× factor, would be just check_loss
        check_loss_only = check_loss_numpy(
            predictions_dict[0.5], y_true, 0.5
        )
        
        # CRPS should be exactly 2× the check loss
        assert np.allclose(crps, 2.0 * check_loss_only)
