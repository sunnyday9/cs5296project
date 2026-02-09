#!/usr/bin/env python3
"""
EC2 inference server: HTTP API that runs model inference and returns result + timing.

This version supports two real model types on tabular data:
- XGBoost (tree-based model)
- MLP exported to ONNX (deep neural network)

You are expected to run scripts/train_models.py locally to produce the model
artifacts under models/ and test feature arrays under data/ first.
"""
import argparse
import time
import json
from pathlib import Path

import numpy as np
from flask import Flask, request, jsonify
import onnxruntime as ort
import xgboost as xgb

app = Flask(__name__)
MODEL_TYPE = None  # "xgb" or "mlp"
DATASET = None  # "adult" or "cancer"
MODEL_DIR: Path | None = None
XGB_MODEL: xgb.XGBClassifier | None = None
ONNX_SESSION: ort.InferenceSession | None = None


def load_models(model_dir: Path, model_type: str, dataset: str):
    """Load the appropriate model(s) into memory."""
    global XGB_MODEL, ONNX_SESSION
    if model_type == "xgb":
        if XGB_MODEL is not None:
            return
        if dataset not in {"adult", "cancer"}:
            raise ValueError(f"Unsupported dataset for XGBoost: {dataset}")
        model_path = model_dir / f"{dataset}_xgb.json"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run scripts/train_models.py locally to generate it."
            )
        XGB_MODEL = xgb.XGBClassifier()
        XGB_MODEL.load_model(str(model_path))
    elif model_type == "mlp":
        if ONNX_SESSION is not None:
            return
        if dataset not in {"adult", "cancer"}:
            raise ValueError(f"Unsupported dataset for MLP: {dataset}")
        model_path = model_dir / f"{dataset}_mlp.onnx"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run scripts/train_models.py locally to generate it."
            )
        ONNX_SESSION = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    """Accept JSON body: {"input": list of floats} or {"batch": list of lists}. Return prediction + latency_ms."""
    global MODEL_DIR, MODEL_TYPE, DATASET
    t0 = time.perf_counter()
    try:
        data = request.get_json(force=True, silent=True) or {}
        inp = data.get("input") or (data.get("batch") or [[]])[0]
        arr = np.array(inp, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # Run inference according to model type
        if MODEL_TYPE == "xgb":
            if XGB_MODEL is None:
                raise RuntimeError("XGBoost model not loaded.")
            # XGBoost expects 2D array
            proba = XGB_MODEL.predict_proba(arr)
            # Return probability of positive class
            result = proba[:, 1].tolist()
        elif MODEL_TYPE == "mlp":
            if ONNX_SESSION is None:
                raise RuntimeError("ONNX session not initialized.")
            input_name = ONNX_SESSION.get_inputs()[0].name
            logits = ONNX_SESSION.run(None, {input_name: arr.astype(np.float32)})[0]
            # Apply sigmoid to get probability
            prob = 1.0 / (1.0 + np.exp(-logits))
            result = prob.reshape(-1).tolist()
        else:
            raise RuntimeError(f"Unsupported MODEL_TYPE: {MODEL_TYPE}")

        latency_ms = (time.perf_counter() - t0) * 1000
        return jsonify({
            "result": result,
            "latency_ms": round(latency_ms, 2),
            "model_type": MODEL_TYPE,
            "dataset": DATASET,
        })
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        return jsonify({"error": str(e), "latency_ms": round(latency_ms, 2)}), 400


def main():
    global MODEL_DIR, MODEL_TYPE, DATASET
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--model-dir", type=Path, default=None, help="Directory containing trained model files.")
    parser.add_argument(
        "--model-type",
        choices=["xgb", "mlp"],
        default="xgb",
        help="Which model type to serve: XGBoost ('xgb') or ONNX MLP ('mlp').",
    )
    parser.add_argument(
        "--dataset",
        choices=["adult", "cancer"],
        default="adult",
        help="Which dataset-specific model to load.",
    )
    args = parser.parse_args()
    MODEL_TYPE = args.model_type
    DATASET = args.dataset
    MODEL_DIR = args.model_dir or (Path(__file__).resolve().parent.parent / "models")
    load_models(MODEL_DIR, MODEL_TYPE, DATASET)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
