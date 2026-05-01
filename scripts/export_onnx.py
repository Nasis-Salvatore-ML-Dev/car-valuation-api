"""
Export the Random Forest model to ONNX format.

ONNX (Open Neural Network Exchange) enables cross-framework interoperability.
The RF model exported here can be served by ONNX Runtime, TorchScript,
TF-ONNX, or any ONNX-compatible runtime — without requiring scikit-learn.

Usage:
    python scripts/export_onnx.py

Outputs:
    models/rand_forest_v1.onnx
"""

import logging
import time
from pathlib import Path

import joblib
import numpy as np
import onnx
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MODEL_PKL_PATH = "models/rand_forest_v1.pkl"
ONNX_OUTPUT_PATH = "models/rand_forest_v1.onnx"
N_BENCHMARK_RUNS = 1000


def export_to_onnx(model_path: str, output_path: str) -> None:
    logger.info("Loading pkl model from %s", model_path)
    package = joblib.load(model_path)

    model = package["model"] if isinstance(package, dict) else package

    n_features = model.n_features_in_
    logger.info("Model: %s | n_features=%d", type(model).__name__, n_features)

    # Convert to ONNX
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model,
        initial_types=initial_type,
        options={type(model): {"zipmap": False}},  # Return array, not dict
    )

    # Validate ONNX model graph structure
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX graph validation passed.")

    output_path_obj = Path(output_path)
    output_path_obj.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path_obj, "wb") as f:
        f.write(onnx_model.SerializeToString())

    size_mb = output_path_obj.stat().st_size / 1024 / 1024
    logger.info("ONNX model saved: %s (%.1f MB)", output_path_obj, size_mb)


def verify_onnx(pkl_path: str, onnx_path: str) -> None:
    """Verify ONNX output matches pkl output within tolerance."""
    package = joblib.load(pkl_path)
    model = package["model"] if isinstance(package, dict) else package

    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    # Generate 5 test inputs
    rng = np.random.default_rng(42)
    n_features = model.n_features_in_
    test_inputs = rng.standard_normal((5, n_features)).astype(np.float32)

    for i, x in enumerate(test_inputs):
        pkl_pred = model.predict(x.reshape(1, -1))[0]
        onnx_pred = sess.run(None, {input_name: x.reshape(1, -1)})[0][0][0]
        rel_diff = abs(pkl_pred - onnx_pred) / (abs(pkl_pred) + 1e-9)
        assert (
            rel_diff < 0.001
        ), f"Sample {i}: pkl={pkl_pred:.6f} onnx={onnx_pred:.6f} rel_diff={rel_diff:.6f}"
        logger.info(
            "Sample %d: pkl=%.4f onnx=%.4f rel_diff=%.6f ✓", i, pkl_pred, onnx_pred, rel_diff
        )

    logger.info("ONNX verification passed: all predictions match pkl within 0.1%%.")


def benchmark(pkl_path: str, onnx_path: str, n_runs: int = N_BENCHMARK_RUNS) -> None:
    """Benchmark pkl vs ONNX inference latency."""
    package = joblib.load(pkl_path)
    model = package["model"] if isinstance(package, dict) else package
    sess = rt.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name

    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, model.n_features_in_)).astype(np.float32)

    # Warm-up
    for _ in range(10):
        model.predict(x)
        sess.run(None, {input_name: x})

    # pkl timing
    pkl_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        model.predict(x)
        pkl_times.append((time.perf_counter() - t0) * 1000)

    # ONNX timing
    onnx_times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        sess.run(None, {input_name: x})
        onnx_times.append((time.perf_counter() - t0) * 1000)

    pkl_arr = np.array(pkl_times)
    onnx_arr = np.array(onnx_times)

    pkl_size_mb = Path(pkl_path).stat().st_size / 1024 / 1024
    onnx_size_mb = Path(onnx_path).stat().st_size / 1024 / 1024

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│  Inference Benchmark: pkl vs ONNX                           │")
    print("├──────────────┬───────────┬───────────┬───────────┬──────────┤")
    print("│ Format       │ Size (MB) │  p50 (ms) │  p99 (ms) │  Speedup │")
    print("├──────────────┼───────────┼───────────┼───────────┼──────────┤")
    print(
        f"│ sklearn pkl  │  {pkl_size_mb:7.1f}  │  {np.percentile(pkl_arr, 50):7.3f}  │  {np.percentile(pkl_arr, 99):7.3f}  │    1.0x  │"
    )
    speedup = np.percentile(pkl_arr, 50) / np.percentile(onnx_arr, 50)
    print(
        f"│ ONNX         │  {onnx_size_mb:7.1f}  │  {np.percentile(onnx_arr, 50):7.3f}  │  {np.percentile(onnx_arr, 99):7.3f}  │  {speedup:5.1f}x  │"
    )
    print("└──────────────┴───────────┴───────────┴───────────┴──────────┘")
    print(
        f"\nONNX is {speedup:.1f}x {'faster' if speedup > 1 else 'slower'} than sklearn pkl at p50."
    )


def main() -> None:
    export_to_onnx(MODEL_PKL_PATH, ONNX_OUTPUT_PATH)
    verify_onnx(MODEL_PKL_PATH, ONNX_OUTPUT_PATH)
    benchmark(MODEL_PKL_PATH, ONNX_OUTPUT_PATH)
    logger.info("Export complete. ONNX model ready at %s", ONNX_OUTPUT_PATH)


if __name__ == "__main__":
    main()
