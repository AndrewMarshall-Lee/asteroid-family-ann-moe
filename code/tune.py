#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Optuna tuning for one asteroid-family expert network feature set."""

from __future__ import annotations

import argparse
from pathlib import Path

import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split

from feature_sets import add_feature_args, resolve_feature_selection
import utils
from models import EarlyStopping, FlexiMLP
from reproducibility import set_global_seed


def objective(
    trial: optuna.Trial,
    data: torch.Tensor,
    target: torch.Tensor,
    epochs: int,
    feature_list: list[str],
    artifact_root: Path,
    artifact_tag: str,
    seed: int | None = None,
) -> float:
    """Train a trial model and return validation loss."""
    if seed is not None:
        torch.manual_seed(seed + trial.number)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed + trial.number)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = data.shape[1]
    output_dim = len(torch.unique(target))

    n_layers = trial.suggest_int("n_layers", 2, 6)
    hidden_sizes = [
        trial.suggest_int(
            f"layer_{idx}_width",
            max(16, input_dim // 2),
            max(64, input_dim * 8),
        )
        for idx in range(n_layers)
    ]
    dropouts = [
        trial.suggest_float(f"dropout_{idx}", 0.0, 0.6)
        for idx in range(n_layers)
    ]
    activation = trial.suggest_categorical(
        "activation",
        ["relu", "leaky_relu", "gelu", "tanh"],
    )
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    optimizer_name = trial.suggest_categorical("optimizer", ["adam", "adamw", "sgd"])
    weight_decay = trial.suggest_float("weight_decay", 1e-8, 1e-3, log=True)
    val_split = trial.suggest_float("val_split", 0.1, 0.3)
    patience = trial.suggest_int("early_stopping_patience", 8, 20)

    dataset = TensorDataset(data, target)
    val_len = max(1, int(len(dataset) * val_split))
    train_len = len(dataset) - val_len
    split_generator = torch.Generator()
    if seed is not None:
        split_generator.manual_seed(seed + trial.number)
    train_ds, val_ds = random_split(
        dataset,
        [train_len, val_len],
        generator=split_generator if seed is not None else None,
    )

    loader_generator = torch.Generator()
    if seed is not None:
        loader_generator.manual_seed(seed + trial.number)
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator if seed is not None else None,
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = FlexiMLP(
        input_dim=input_dim,
        output_dim=output_dim,
        hidden_sizes=hidden_sizes,
        dropouts=dropouts,
        activation=activation,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    if optimizer_name == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay,
        )
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    stopper = EarlyStopping(patience=patience, delta=1e-3)
    final_val_loss = float("inf")

    for epoch in range(epochs):
        model.train()
        for batch_data, batch_target in train_loader:
            batch_data = batch_data.to(device).float()
            batch_target = batch_target.to(device).long()

            optimizer.zero_grad(set_to_none=True)
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

        final_val_loss = val_loss / max(1, len(val_loader))
        trial.report(final_val_loss, epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
        if stopper(final_val_loss, model):
            break

    trial_params = {
        "Loss": final_val_loss,
        **trial.params,
    }
    utils.save_tuning_params(
        trial_params,
        artifact_root / "Trials" / artifact_tag,
        best=False,
    )
    return final_val_loss


def tune_hyperparameters(
    data: torch.Tensor,
    target: torch.Tensor,
    n_trials: int,
    epochs: int,
    feature_list: list[str],
    artifact_tag: str,
    artifact_root: str | Path = "artifacts",
    seed: int | None = None,
) -> dict:
    """Tune one feature set and save the best parameters for train_moe.py."""
    artifact_root = Path(artifact_root)
    (artifact_root / "Tune").mkdir(parents=True, exist_ok=True)
    (artifact_root / "Trials").mkdir(parents=True, exist_ok=True)

    sampler = optuna.samplers.TPESampler(seed=seed) if seed is not None else None
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: objective(
            trial,
            data=data,
            target=target,
            epochs=epochs,
            feature_list=feature_list,
            artifact_root=artifact_root,
            artifact_tag=artifact_tag,
            seed=seed,
        ),
        n_trials=n_trials,
    )

    best_params = {
        "Loss": study.best_value,
        **study.best_params,
    }
    utils.save_tuning_params(
        best_params,
        artifact_root / "Tune" / artifact_tag,
        best=True,
    )
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_data.csv")
    parser.add_argument("--artifact-root", default="artifacts")
    add_feature_args(parser)
    parser.add_argument("--epochs", type=int, default=50, help="Epochs per Optuna trial.")
    parser.add_argument("--trials", type=int, default=100, help="Number of Optuna trials.")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    set_global_seed(args.seed)
    feature_list, artifact_tag = resolve_feature_selection(args)
    train_data, _, target = utils.load_train_data(
        args.data,
        feature_list,
        args.artifact_root,
        artifact_tag,
    )
    best = tune_hyperparameters(
        train_data,
        target,
        n_trials=args.trials,
        epochs=args.epochs,
        feature_list=feature_list,
        artifact_tag=artifact_tag,
        artifact_root=args.artifact_root,
        seed=args.seed,
    )
    print("Best tuning parameters:")
    for key, value in best.items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
