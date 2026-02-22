"""
Lambda handler for ML/DL inference on tabular data.

Supports two model types (same artifacts as EC2 server):
- XGBoost: <dataset>_xgb.json
- MLP (ONNX): <dataset>_mlp.onnx

Environment variables control behaviour:
- MODEL_TYPE: "xgb" or "mlp"
- DATASET: "adult" or "cancer"
- MODEL_DIR (optional): directory containing model files (default: ../models)
"""
import json
import os
import time
from pathlib import Path

import numpy as np

# Lazy import so you can deploy XGB-only or MLP-only package (each under 250 MB)
_XGB_MODEL = None
_ONNX_SESSION = None
_MODEL_TYPE = os.getenv("MODEL_TYPE", "xgb")
_DATASET = os.getenv("DATASET", "adult")
_MODEL_DIR = Path(os.getenv("MODEL_DIR", str(Path(__file__).resolve().parent / "models")))


def _load_models():
    global _XGB_MODEL, _ONNX_SESSION
    if _MODEL_TYPE == "xgb":
        if _XGB_MODEL is not None:
            return
        import xgboost as xgb
        if _DATASET not in {"adult", "cancer"}:
            raise ValueError(f"Unsupported dataset for XGBoost: {_DATASET}")
        model_path = _MODEL_DIR / f"{_DATASET}_xgb.json"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found in Lambda package.")
        _XGB_MODEL = xgb.XGBClassifier()
        _XGB_MODEL.load_model(str(model_path))
    elif _MODEL_TYPE == "mlp":
        if _ONNX_SESSION is not None:
            return
        import onnxruntime as ort  # noqa: F401
        if _DATASET not in {"adult", "cancer"}:
            raise ValueError(f"Unsupported dataset for MLP: {_DATASET}")
        model_path = _MODEL_DIR / f"{_DATASET}_mlp.onnx"
        if not model_path.exists():
            raise FileNotFoundError(f"{model_path} not found in Lambda package.")
        _ONNX_SESSION = ort.InferenceSession(str(model_path))
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {_MODEL_TYPE}")


def lambda_handler(event, context):
    _load_models()
    t0 = time.perf_counter()
    try:
        body = event.get("body")
        if isinstance(body, str):
            body = json.loads(body)
        elif body is None:
            body = event
        inp = body.get("input") or body.get("body", {}).get("input") or [0.0] * 32
        arr = np.array(inp, dtype=np.float32)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)

        if _MODEL_TYPE == "xgb":
            proba = _XGB_MODEL.predict_proba(arr)
            result = proba[:, 1].tolist()
        else:  # mlp
            input_name = _ONNX_SESSION.get_inputs()[0].name
            logits = _ONNX_SESSION.run(None, {input_name: arr.astype(np.float32)})[0]
            prob = 1.0 / (1.0 + np.exp(-logits))
            result = prob.reshape(-1).tolist()

        latency_ms = (time.perf_counter() - t0) * 1000
        memory_mb = getattr(context, "memory_limit_in_mb", None)
        return {
            "statusCode": 200,
            "body": json.dumps({
                "result": result,
                "latency_ms": round(latency_ms, 2),
                "memory_mb": memory_mb,
                "model_type": _MODEL_TYPE,
                "dataset": _DATASET,
                "request_id": getattr(context, "aws_request_id", ""),
            }),
            "headers": {"Content-Type": "application/json"},
        }
    except Exception as e:
        latency_ms = (time.perf_counter() - t0) * 1000
        memory_mb = getattr(context, "memory_limit_in_mb", None)
        return {
            "statusCode": 500,
            "body": json.dumps({
                "error": str(e),
                "latency_ms": round(latency_ms, 2),
                "memory_mb": memory_mb,
            }),
            "headers": {"Content-Type": "application/json"},
        }
