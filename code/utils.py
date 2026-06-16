#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Data loading helpers used by the reproducibility scripts."""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import torch
from astropy.io import fits
from astropy.table import Table
from sklearn.preprocessing import LabelEncoder, StandardScaler


BASE_FEATURES = [
    "packed_mpc_name",
    "PROPER_SEMIMAJOR_AXIS",
    "PROPER_ECCENTRICITY",
    "SINE_OF_PROPER_INCLINATION",
    "family_id",
]


def save_tuning_params(
    params: dict,
    path_prefix: str | Path,
    best: bool,
) -> Path:
    """Persist Optuna parameters in the filename convention used by training."""
    path_prefix = Path(path_prefix)
    path_prefix.parent.mkdir(parents=True, exist_ok=True)

    if best:
        out_path = path_prefix.with_name(path_prefix.name + "_best.dat")
        pd.DataFrame([params]).to_csv(out_path, index=False)
        return out_path

    out_path = path_prefix.with_name(path_prefix.name + "_trials.dat")
    row = pd.DataFrame([params])
    if out_path.exists() and out_path.stat().st_size > 0:
        row.to_csv(out_path, mode="a", header=False, index=False)
    else:
        row.to_csv(out_path, index=False)
    return out_path


def preprocess_data(fits_path: str | Path, output_csv: str | Path = "processed_data.csv") -> pd.DataFrame:
    """Create the inner-belt processed CSV from the unified FITS table."""
    dat_tab = Table(fits.open(fits_path)[1].data)
    df = dat_tab.to_pandas()

    family_columns = ["family_2015", "family_2015_2", "family_2024", "family_2024_2"]
    df[family_columns] = df[family_columns].replace("<NA>", 0)
    df[family_columns] = df[family_columns].replace(999999, 0)

    df = df[(df["PROPER_SEMIMAJOR_AXIS"] <= 2.5) & (2.1 <= df["PROPER_SEMIMAJOR_AXIS"])]

    def unique_families(values):
        return sorted({str(v).strip() for v in values if pd.notna(v) and str(v).strip() != "0"})

    expanded_rows = []
    for _, row in df.iterrows():
        families_2024 = unique_families([row["family_2024"], row["family_2024_2"]])
        families_2015 = unique_families([row["family_2015"], row["family_2015_2"]])
        families = families_2024 or families_2015 or ["0"]
        for family in families:
            new_row = row.copy()
            new_row["family_id"] = family
            expanded_rows.append(new_row)

    processed = pd.DataFrame(expanded_rows).reset_index(drop=True)
    processed.to_csv(output_csv)
    return processed


def load_train_data(
    csv_path: str | Path,
    feature_list: list[str],
    artifact_root: str | Path = ".",
) -> tuple[torch.Tensor, StandardScaler, torch.Tensor]:
    """Load complete rows for a feature set and persist scaler/label encoder."""
    artifact_root = Path(artifact_root)
    (artifact_root / "Scaler").mkdir(parents=True, exist_ok=True)
    (artifact_root / "LE").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path, index_col=0, low_memory=False)
    df = df[feature_list].dropna()

    labels = df["family_id"].astype(str)
    label_encoder = LabelEncoder()
    label_indices = label_encoder.fit_transform(labels)

    features = df.drop(columns=["family_id", "packed_mpc_name"])
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    feature_tag = str(feature_list)
    joblib.dump(scaler, artifact_root / "Scaler" / f"{feature_tag}_scaler.save")
    joblib.dump(label_encoder, artifact_root / "LE" / f"{feature_tag}_label_encoder.save")

    train = torch.tensor(features_scaled, dtype=torch.float32)
    target = torch.tensor(label_indices, dtype=torch.long)
    return train, scaler, target


if __name__ == "__main__":
    preprocess_data("data/unified_asteroid_data_with_gaia.fits", "data/processed_data.csv")
