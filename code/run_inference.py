#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""CLI wrapper for reproducing prediction CSVs from saved weights."""

from __future__ import annotations

import argparse
from pathlib import Path

from classify import test_model
from feature_sets import add_feature_args, resolve_feature_selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_data.csv")
    parser.add_argument("--artifact-root", default="artifacts")
    add_feature_args(parser)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    feature_list, artifact_tag = resolve_feature_selection(args)
    artifact_root = Path(args.artifact_root)
    model_path = artifact_root / "Train" / f"{artifact_tag}_model.pth"
    test_model(args.data, model_path, args.output, feature_list, artifact_root, artifact_tag)


if __name__ == "__main__":
    main()
