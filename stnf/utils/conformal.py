"""
Conformal calibration for quantile regression (CQR symmetric interval expansion).

Uses calibration set to compute expansion qhat; prediction interval becomes
[q_lo - qhat, q_hi + qhat] for nominal (1 - alpha) coverage.
"""
import math
import numpy as np


def compute_cqr_qhat(cal_preds, cal_y_true, quantile_levels, alpha=0.1):
    """
    Compute CQR expansion threshold from calibration set (symmetric expansion).

    Nonconformity score per sample (standard CQR, nonnegative):
        score = max(q_lo - y, y - q_hi, 0)
    so points inside [q_lo, q_hi] get score 0. qhat = k-th order statistic of
    scores with k = ceil((n+1)*(1-alpha)); qhat is forced >= 0.

    Args:
        cal_preds: (N, Q) calibration quantile predictions
        cal_y_true: (N,) or (N, 1) calibration true values
        quantile_levels: list of quantile levels (e.g. [0.05, 0.25, 0.5, 0.75, 0.95])
        alpha: nominal miscoverage (0.1 -> 90% nominal)

    Returns:
        qhat: float, expansion amount (same units as y)
        calibration_n: int, number of calibration samples used
    """
    if cal_y_true.ndim > 1:
        cal_y_true = cal_y_true.flatten()
    quantile_levels = np.asarray(quantile_levels)
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    q_lo_pred = cal_preds[:, idx_lo]
    q_hi_pred = cal_preds[:, idx_hi]
    # Standard CQR: score = max(q_lo - y, y - q_hi, 0) so in-interval points get 0
    scores = np.maximum.reduce([q_lo_pred - cal_y_true, cal_y_true - q_hi_pred, np.zeros_like(cal_y_true)])
    n = scores.size
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)  # if k > n use max score
    qhat = float(np.partition(scores, k - 1)[k - 1]) if k >= 1 else float(np.max(scores))
    qhat = max(qhat, 0.0)  # guarantee nonnegative expansion
    return qhat, n


def compute_conformal_coverage(preds, y_true, quantile_levels, qhat, alpha=0.1):
    """
    Empirical coverage of the conformalized interval [q_lo - qhat, q_hi + qhat].

    Args:
        preds: (N, Q) quantile predictions
        y_true: (N,) or (N, 1) true values
        quantile_levels: list of quantile levels
        qhat: expansion from compute_cqr_qhat
        alpha: nominal miscoverage (0.1 -> 90% nominal)

    Returns:
        coverage: fraction of y_true inside the conformalized interval
    """
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    quantile_levels = np.asarray(quantile_levels)
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    low = preds[:, idx_lo] - qhat
    high = preds[:, idx_hi] + qhat
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside))
