#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Run inference with trained asteroid-family ANN weights."""

from __future__ import annotations

import ast
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from models import FlexiMLP


def _load_hyperparameters(path: Path) -> dict:
    params = pd.read_csv(path)
    params.columns = [c.strip() for c in params.columns]

    if "hidden_sizes" in params.columns:
        return {
            "hidden_sizes": ast.literal_eval(str(params["hidden_sizes"].iloc[0])),
            "dropouts": ast.literal_eval(str(params["dropouts"].iloc[0])),
            "activation": str(params["activation"].iloc[0]),
            "batch_size": int(params.get("batch_size", pd.Series([64])).iloc[0]),
            "lr": float(params.get("lr", pd.Series([1e-3])).iloc[0]),
            "optimizer": str(params.get("optimizer", pd.Series(["adam"])).iloc[0]),
            "weight_decay": float(params.get("weight_decay", pd.Series([1e-5])).iloc[0]),
            "val_split": float(params.get("val_split", pd.Series([0.2])).iloc[0]),
        }

    layer_cols = sorted(c for c in params.columns if "layer_" in c and "width" in c)
    dropout_cols = sorted(c for c in params.columns if "dropout_" in c)
    return {
        "hidden_sizes": [int(params[c].iloc[0]) for c in layer_cols],
        "dropouts": [float(params[c].iloc[0]) for c in dropout_cols],
        "activation": str(params.get("activation", pd.Series(["relu"])).iloc[0]),
        "batch_size": int(params.get("batch_size", pd.Series([64])).iloc[0]),
        "lr": float(params.get("lr", pd.Series([1e-3])).iloc[0]),
        "optimizer": str(params.get("optimizer", pd.Series(["adam"])).iloc[0]),
        "weight_decay": float(params.get("weight_decay", pd.Series([1e-5])).iloc[0]),
        "val_split": float(params.get("val_split", pd.Series([0.2])).iloc[0]),
    }


def test_model(
    csv_path: str | Path,
    model_path: str | Path,
    output_csv_path: str | Path | None,
    feature_list: list[str],
    artifact_root: str | Path = ".",
) -> pd.DataFrame:
    """Classify complete rows and preserve non-reviewable rows in the output."""
    artifact_root = Path(artifact_root)
    feature_tag = str(feature_list)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_full = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df_model = df_full[feature_list].dropna()

    scaler = joblib.load(artifact_root / "Scaler" / f"{feature_tag}_scaler.save")
    label_encoder = joblib.load(artifact_root / "LE" / f"{feature_tag}_label_encoder.save")
    params = _load_hyperparameters(artifact_root / "Tune" / f"{feature_tag}_best.dat")

    x = df_model.drop(columns=["family_id", "packed_mpc_name"])
    x_scaled = scaler.transform(x)
    x_tensor = torch.tensor(x_scaled, dtype=torch.float32, device=device)

    model = FlexiMLP(
        input_dim=x_tensor.shape[1],
        output_dim=len(label_encoder.classes_),
        hidden_sizes=params["hidden_sizes"],
        dropouts=params["dropouts"],
        activation=params["activation"],
    ).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    with torch.no_grad():
        probabilities = torch.softmax(model(x_tensor), dim=1).cpu().numpy()

    pred_idx = np.argmax(probabilities, axis=1)
    pred_label = label_encoder.inverse_transform(pred_idx)
    pred_confidence = probabilities[np.arange(len(probabilities)), pred_idx]

    df_final = df_full.copy()
    df_final["ANN_reviewed"] = False
    df_final.loc[df_model.index, "ANN_reviewed"] = True
    df_final["prediction_confidence"] = np.nan
    df_final["ANN_predicted_family_id"] = np.nan
    df_final.loc[df_model.index, "prediction_confidence"] = pred_confidence
    df_final.loc[df_model.index, "ANN_predicted_family_id"] = pred_label

    if output_csv_path is not None:
        Path(output_csv_path).parent.mkdir(parents=True, exist_ok=True)
        df_final.to_csv(output_csv_path, index=False)

    return df_final
