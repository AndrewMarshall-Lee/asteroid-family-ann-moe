#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Train family-stratified asteroid-family ANN classifiers."""

from __future__ import annotations

import ast
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset

from models import EarlyStopping, FlexiMLP


def _load_hyperparameters(path: Path) -> dict:
    params = pd.read_csv(path)
    params.columns = [c.strip() for c in params.columns]

    if "hidden_sizes" in params.columns:
        hidden_sizes = ast.literal_eval(str(params["hidden_sizes"].iloc[0]))
        dropouts = ast.literal_eval(str(params["dropouts"].iloc[0]))
    else:
        hidden_sizes = [int(params[c].iloc[0]) for c in sorted(c for c in params.columns if "layer_" in c and "width" in c)]
        dropouts = [float(params[c].iloc[0]) for c in sorted(c for c in params.columns if "dropout_" in c)]

    return {
        "hidden_sizes": hidden_sizes,
        "dropouts": dropouts,
        "activation": str(params.get("activation", pd.Series(["relu"])).iloc[0]),
        "batch_size": int(params.get("batch_size", pd.Series([64])).iloc[0]),
        "lr": float(params.get("lr", pd.Series([1e-3])).iloc[0]),
        "optimizer": str(params.get("optimizer", pd.Series(["adam"])).iloc[0]),
        "weight_decay": float(params.get("weight_decay", pd.Series([1e-5])).iloc[0]),
        "val_split": float(params.get("val_split", pd.Series([0.2])).iloc[0]),
    }


def train_model(
    data: torch.Tensor,
    target: torch.Tensor,
    num_epochs: int,
    early_stop: bool,
    patience: int,
    delta: float,
    feature_list: list[str],
    artifact_root: str | Path = ".",
) -> FlexiMLP:
    """Train one model and save weights/losses under artifact_root."""
    artifact_root = Path(artifact_root)
    for name in ["Train", "Loss"]:
        (artifact_root / name).mkdir(parents=True, exist_ok=True)

    input_dim = data.shape[1]
    output_dim = len(torch.unique(target))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_tag = str(feature_list)
    params = _load_hyperparameters(artifact_root / "Tune" / f"{feature_tag}_best.dat")

    model = FlexiMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=params["hidden_sizes"],
        dropouts=params["dropouts"],
        activation=params["activation"],
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer_name = params["optimizer"].lower()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=params["lr"],
            momentum=0.9,
            nesterov=True,
            weight_decay=params["weight_decay"],
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])

    dataset = TensorDataset(data, target)
    rng = np.random.default_rng()
    class_to_indices: dict[int, list[int]] = defaultdict(list)
    for idx, cls in enumerate(target.cpu().numpy()):
        class_to_indices[int(cls)].append(idx)

    train_idx = []
    val_idx = []
    for indices in class_to_indices.values():
        shuffled = np.array(indices)
        rng.shuffle(shuffled)
        n_val = max(1, int(len(shuffled) * params["val_split"]))
        val_idx.extend(shuffled[:n_val])
        train_idx.extend(shuffled[n_val:])

    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=params["batch_size"], shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=params["batch_size"], shuffle=False)
    early_stopping = EarlyStopping(patience, delta)
    val_loss_log = []

    for epoch in range(num_epochs):
        model.train()
        for batch_data, batch_target in train_loader:
            batch_data = batch_data.to(device).float()
            batch_target = batch_target.to(device).long()
            optimizer.zero_grad()
            loss = criterion(model(batch_data), batch_target)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data = val_data.to(device).float()
                val_target = val_target.to(device).long()
                val_loss += criterion(model(val_data), val_target).item()

        avg_val_loss = val_loss / max(1, len(val_loader))
        val_loss_log.append(avg_val_loss)
        print(f"Epoch {epoch + 1}/{num_epochs} | val_loss={avg_val_loss:.6f}")

        if early_stop and early_stopping(avg_val_loss, model):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    if early_stop:
        early_stopping.load_best_model(model)

    torch.save(model.state_dict(), artifact_root / "Train" / f"{feature_tag}_model.pth")
    np.savetxt(artifact_root / "Loss" / f"{feature_tag}_loss.dat", np.array(val_loss_log))
    return model
