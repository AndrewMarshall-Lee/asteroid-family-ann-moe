#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Neural-network model definitions for asteroid-family classification."""

from __future__ import annotations

import torch
import torch.nn as nn


class FlexiMLP(nn.Module):
    """Flexible fully connected classifier used for the ANN/MOE experiments."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_sizes: list[int],
        dropouts: list[float] | None = None,
        activation: str = "relu",
    ) -> None:
        super().__init__()

        if dropouts is None:
            dropouts = [0.2] * len(hidden_sizes)
        if len(dropouts) != len(hidden_sizes):
            raise ValueError("dropouts must have the same length as hidden_sizes")

        activation = activation.lower()
        if activation == "relu":
            activation_factory = nn.ReLU
        elif activation in {"leaky_relu", "lrelu"}:
            activation_factory = lambda: nn.LeakyReLU(negative_slope=0.01)
        elif activation == "gelu":
            activation_factory = nn.GELU
        elif activation == "tanh":
            activation_factory = nn.Tanh
        else:
            raise ValueError(f"Unknown activation function: {activation}")

        layers: list[nn.Module] = []
        prev_dim = input_dim
        for width, dropout in zip(hidden_sizes, dropouts):
            layers.append(nn.Linear(prev_dim, int(width)))
            layers.append(activation_factory())
            layers.append(nn.Dropout(float(dropout)))
            prev_dim = int(width)

        layers.append(nn.Linear(prev_dim, output_dim))
        self.MLP = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.MLP(x)


class EarlyStopping:
    """Small early-stopping helper for optional retraining."""

    def __init__(self, patience: int, delta: float) -> None:
        self.patience = patience
        self.delta = delta
        self.best_score: float | None = None
        self.early_stop = False
        self.counter = 0
        self.best_model_state = None

    def __call__(self, val_loss: float, model: nn.Module) -> bool:
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.best_model_state = model.state_dict()
            return False

        if score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_model_state = model.state_dict()
            self.counter = 0

        return self.early_stop

    def load_best_model(self, model: nn.Module) -> None:
        if self.best_model_state is not None:
            model.load_state_dict(self.best_model_state)
