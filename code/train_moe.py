#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""CLI wrapper for retraining one feature-set model from processed_data.csv."""

from __future__ import annotations

import argparse

from feature_sets import add_feature_args, resolve_feature_selection
from reproducibility import set_global_seed
import utils
from train_family_stratified import train_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_data.csv")
    parser.add_argument("--artifact-root", default="artifacts")
    add_feature_args(parser)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--early-stop", action="store_true")
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
    train_model(
        train_data,
        target,
        num_epochs=args.epochs,
        early_stop=args.early_stop,
        patience=20,
        delta=0.005,
        feature_list=feature_list,
        artifact_root=args.artifact_root,
        artifact_tag=artifact_tag,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
