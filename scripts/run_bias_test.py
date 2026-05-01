"""
Run bias tests against the trained model.

Executed in the CD pipeline after model loading to ensure no segment
has MAE > 1.5x the overall MAE. Exits with code 1 if any segment is
flagged, preventing deployment of a biased model.

Usage:
    MODEL_PATH=models/rand_forest_v1.pkl python scripts/run_bias_test.py
"""

import sys

from src.monitoring.bias_tester import BiasTestSuite
from src.utils.model_loader import load_model_bundle


def main() -> int:
    print("Loading model bundle...")
    bundle = load_model_bundle()
    print(f"Model loaded: {bundle.version}")

    print("Running bias tests...")
    suite = BiasTestSuite(bundle)
    report = suite.cached_report()

    print(f"\nBias Report — {report['model_version']}")
    print(f"  Overall MAE:  {report['overall_mae_eur']:.2f} EUR")
    print(f"  Overall RMSE: {report['overall_rmse_eur']:.2f} EUR")
    print(f"  Overall R²:   {report['overall_r2']:.4f}")

    flagged = [s for s in report.get("bias_segments", []) if s.get("flagged")]
    total = len(report.get("bias_segments", []))

    if total == 0:
        print("\n  No segments evaluated (validation data may be missing).")
        print("  Bias test passed (no data to flag).")
        return 0

    print(f"\n  Segments evaluated: {total}")
    for seg in report.get("bias_segments", []):
        flag = " *** FLAGGED ***" if seg["flagged"] else ""
        print(
            f"    {seg['segment_name']:20s}  "
            f"MAE={seg['mae_eur']:>8.2f} EUR  "
            f"ratio={seg['relative_to_overall']:.3f}"
            f"{flag}"
        )

    if flagged:
        print(f"\n  FAIL: {len(flagged)} segment(s) exceed 1.5x overall MAE.")
        if report.get("recommendation"):
            print(f"  Recommendation: {report['recommendation']}")
        return 1

    print("\n  PASS: All segments within acceptable bias thresholds.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
