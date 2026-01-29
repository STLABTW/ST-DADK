"""
Compute empirical coverage of the 90% prediction interval from a run's predictions.npz.

Usage:
    python scripts/compute_coverage.py --result_dir results/scan_temporal_bandwidth_1.0
    python scripts/compute_coverage.py --result_dir results/scan_temporal_bandwidth_1.0 --alpha 0.1

If results.json has test_coverage_90, that value is shown; otherwise coverage is computed
from predictions.npz (requires quantile predictions).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def compute_coverage_from_quantiles(preds, y_true, quantile_levels, alpha=0.1):
    """Empirical coverage of (1-alpha)*100% interval [q_alpha/2, q_{1-alpha/2}]."""
    if y_true.ndim > 1:
        y_true = y_true.flatten()
    quantile_levels = np.asarray(quantile_levels)
    q_lo = alpha / 2
    q_hi = 1.0 - alpha / 2
    idx_lo = np.argmin(np.abs(quantile_levels - q_lo))
    idx_hi = np.argmin(np.abs(quantile_levels - q_hi))
    low = preds[:, idx_lo]
    high = preds[:, idx_hi]
    inside = (y_true >= low) & (y_true <= high)
    return float(np.mean(inside)), inside


def main():
    ap = argparse.ArgumentParser(description="Compute 90% PI coverage from result dir")
    ap.add_argument("--result_dir", type=str, default="results/scan_temporal_bandwidth_1.0", help="Result directory")
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage (0.1 -> 90% interval)")
    args = ap.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Directory not found: {result_dir}")
        return 1

    # 1) Prefer coverage from results.json if present
    results_file = result_dir / "results.json"
    if results_file.exists():
        with open(results_file) as f:
            data = json.load(f)
        cov = data.get("test_coverage_90")
        if cov is not None:
            print(f"Test coverage (90% PI) from results.json: {cov:.4f} (target 0.90)")
            return 0

    # 2) predictions.npz currently stores only median (T, S), not all quantiles; cannot compute coverage from it.
    #    Re-run training to get test_coverage_90 in results.json.
    pred_file = result_dir / "predictions.npz"
    if pred_file.exists():
        npz = np.load(pred_file, allow_pickle=True)
        predictions = npz["predictions"]
        if predictions.ndim >= 3:
            true_vals = npz["true"]
            test_mask = npz["test_mask"]
            quantile_levels = [0.05, 0.25, 0.5, 0.75, 0.95]
            if results_file.exists():
                with open(results_file) as f:
                    data = json.load(f)
                quantile_levels = data.get("config", {}).get("quantile_levels", quantile_levels)
            preds_flat = predictions[test_mask]
            true_flat = true_vals[test_mask]
            coverage, inside = compute_coverage_from_quantiles(
                preds_flat, true_flat, quantile_levels, alpha=args.alpha
            )
            nominal = 1.0 - args.alpha
            print(f"Test coverage ({nominal*100:.0f}% PI) from predictions.npz: {coverage:.4f} (target {nominal:.2f})")
            print(f"  Inside: {int(inside.sum())}, Total: {len(inside)}")
            return 0
    print("No test_coverage_90 in results.json and predictions.npz has no quantile dims.")
    print("Re-run training (multi-quantile) to get coverage in results and on console.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
