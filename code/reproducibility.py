#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Small helpers for reproducible ANN/MOE runs."""

from __future__ import annotations

import os
import random

import numpy as np
import torch


def set_global_seed(seed: int | None) -> None:
    """Seed Python, NumPy, and PyTorch when a seed is provided."""
    if seed is None:
        return

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
