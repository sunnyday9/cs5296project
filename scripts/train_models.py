#!/usr/bin/env python3
"""
Train tabular models (XGBoost + MLP) on two datasets:
- Adult Income (UCI)
- Breast Cancer Wisconsin

This script is intended to be run LOCALLY (not on EC2/Lambda) to produce
pretrained weights and test feature arrays that will be used solely for
inference in your experiments.

Outputs (under models/ and data/):
- models/adult_xgb.json
- models/adult_mlp.onnx
- data/adult_test_inputs.npy
- models/cancer_xgb.json
- models/cancer_mlp.onnx
- data/cancer_test_inputs.npy
"""

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from torch.utils.data import DataLoader, TensorDataset
import onnx
import onnxruntime as ort  # noqa: F401  # used at inference time; ensure installed


PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"


class MLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_mlp(X_train: np.ndarray, y_train: np.ndarray, input_dim: int, epochs: int = 5, batch_size: int = 256) -> MLP:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLP(input_dim=input_dim).to(device)
    X_tensor = torch.from_numpy(X_train.astype(np.float32))
    y_tensor = torch.from_numpy(y_train.astype(np.float32)).unsqueeze(1)
    ds = TensorDataset(X_tensor, y_tensor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for xb, yb in dl:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss += loss.item() * xb.size(0)
        epoch_loss /= len(ds)
        print(f"Epoch {epoch + 1}/{epochs}, loss={epoch_loss:.4f}")
    return model.cpu()


def export_mlp_to_onnx(model: MLP, input_dim: int, out_path: Path):
    """Export MLP to a single .onnx file (no separate .onnx.data)."""
    import io
    model.eval()
    dummy = torch.zeros(1, input_dim, dtype=torch.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.BytesIO()
    # dynamo=False: use legacy exporter so opset_version=9 is respected (Lambda onnxruntime supports up to IR 9)
    torch.onnx.export(
        model,
        dummy,
        buffer,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=9,
        dynamo=False,
    )
    with open(out_path, "wb") as f:
        f.write(buffer.getvalue())
    print(f"Saved ONNX model to {out_path}")


def train_adult():
    from sklearn.datasets import fetch_openml

    print("=== Training on Adult Income dataset ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    adult = fetch_openml("adult", version=2, as_frame=True)
    X = adult.data
    y = (adult.target == ">50K").astype(int).to_numpy()

    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    num_cols = X.select_dtypes(exclude=["category", "object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )
    reducer = TruncatedSVD(n_components=64, random_state=42)

    X_proc = preprocessor.fit_transform(X)
    X_reduced = reducer.fit_transform(X_proc)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y, test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_out = MODELS_DIR / "adult_xgb.json"
    xgb_model.save_model(str(xgb_out))
    print(f"Saved Adult XGBoost model to {xgb_out}")

    # MLP
    mlp = train_mlp(X_train, y_train, input_dim=X_reduced.shape[1])
    mlp_out = MODELS_DIR / "adult_mlp.onnx"
    export_mlp_to_onnx(mlp, input_dim=X_reduced.shape[1], out_path=mlp_out)

    # Save test inputs
    np.save(DATA_DIR / "adult_test_inputs.npy", X_test.astype(np.float32))
    print(f"Saved Adult test features to {DATA_DIR / 'adult_test_inputs.npy'}")


def train_cancer():
    print("=== Training on Breast Cancer dataset ===")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    X, y = datasets.load_breast_cancer(return_X_y=True, as_frame=True)
    num_cols = X.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X[num_cols])

    # Optionally reduce to 32 dims to mirror a smaller feature space
    reducer = TruncatedSVD(n_components=min(32, X_scaled.shape[1]), random_state=42)
    X_reduced = reducer.fit_transform(X_scaled)

    X_train, X_test, y_train, y_test = train_test_split(
        X_reduced, y.to_numpy(), test_size=0.2, random_state=42, stratify=y
    )

    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.1,
        subsample=0.9,
        colsample_bytree=0.9,
        objective="binary:logistic",
        n_jobs=-1,
        tree_method="hist",
        random_state=42,
    )
    xgb_model.fit(X_train, y_train)
    xgb_out = MODELS_DIR / "cancer_xgb.json"
    xgb_model.save_model(str(xgb_out))
    print(f"Saved Cancer XGBoost model to {xgb_out}")

    # MLP
    mlp = train_mlp(X_train, y_train, input_dim=X_reduced.shape[1])
    mlp_out = MODELS_DIR / "cancer_mlp.onnx"
    export_mlp_to_onnx(mlp, input_dim=X_reduced.shape[1], out_path=mlp_out)

    np.save(DATA_DIR / "cancer_test_inputs.npy", X_test.astype(np.float32))
    print(f"Saved Cancer test features to {DATA_DIR / 'cancer_test_inputs.npy'}")


def main():
    train_adult()
    train_cancer()


if __name__ == "__main__":
    main()

