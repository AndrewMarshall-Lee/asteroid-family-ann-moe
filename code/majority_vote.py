#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Aggregate repeated ANN prediction CSVs by majority vote."""

from __future__ import annotations

import argparse
import glob
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from feature_sets import add_feature_args, artifact_tag_for, resolve_feature_selection


PRED_COL = "ANN_predicted_family_id"
CONF_COL = "prediction_confidence"


def vote_entropy(votes: list[str]) -> float:
    if not votes:
        return float("nan")
    counts = Counter(votes)
    total = sum(counts.values())
    p = np.array([count / total for count in counts.values()], dtype=float)
    return float(-(p * np.log(p)).sum())


def load_multirun_majority(
    prediction_dir: Path,
    feature_list: list[str],
    artifact_tag: str | None = None,
) -> pd.DataFrame:
    artifact_tag = artifact_tag or artifact_tag_for(feature_list)
    pattern = str(prediction_dir / f"{glob.escape(artifact_tag)}*run*.csv")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No prediction files found for pattern: {pattern}")

    big = pd.concat((pd.read_csv(path, low_memory=False) for path in files), ignore_index=True)
    rows = []
    for name, group in big.groupby("packed_mpc_name"):
        row = {col: group[col].iloc[0] for col in group.columns if col not in {PRED_COL, CONF_COL}}
        votes = group[PRED_COL].dropna().tolist()
        if votes:
            counts = Counter(votes)
            top_label, top_count = counts.most_common(1)[0]
            row[PRED_COL] = top_label
            row["n_votes"] = len(votes)
            row["vote_top_frac"] = top_count / len(votes)
            row["vote_entropy"] = vote_entropy(votes)
            row["n_unique_votes"] = len(counts)
        else:
            row[PRED_COL] = np.nan
            row["n_votes"] = 0
            row["vote_top_frac"] = np.nan
            row["vote_entropy"] = np.nan
            row["n_unique_votes"] = 0
        if CONF_COL in group.columns:
            row[f"{CONF_COL}_mean"] = pd.to_numeric(group[CONF_COL], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-dir", default="Predictions")
    parser.add_argument("--output-dir", default="Majority Predictions")
    add_feature_args(parser)
    args = parser.parse_args()

    feature_list, artifact_tag = resolve_feature_selection(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    df = load_multirun_majority(Path(args.prediction_dir), feature_list, artifact_tag)
    df.to_csv(output_dir / f"{artifact_tag}.csv", index=False)


if __name__ == "__main__":
    main()
