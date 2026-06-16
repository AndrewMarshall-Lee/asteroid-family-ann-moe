#!/usr/bin/env python3
# Author: Andrew Marshall-Lee
"""Run the full tune/train/test/vote workflow for one expert feature set."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from classify import test_model
from feature_sets import add_feature_args, resolve_feature_list
from majority_vote import load_multirun_majority
from train_family_stratified import train_model
from tune import tune_hyperparameters
import utils


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data/processed_data.csv")
    parser.add_argument("--fits", default="data/unified_asteroid_data_with_gaia.fits")
    parser.add_argument("--artifact-root", default="artifacts")
    parser.add_argument("--prediction-dir", default="outputs/Predictions")
    parser.add_argument("--majority-dir", default="outputs/Majority_Predictions")
    add_feature_args(parser)
    parser.add_argument("--build-fits", action="store_true")
    parser.add_argument("--rebuild-data", action="store_true")
    parser.add_argument("--tune", action="store_true")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--tune-epochs", type=int, default=50)
    parser.add_argument("--train-epochs", type=int, default=300)
    parser.add_argument("--runs", type=int, default=20)
    parser.add_argument("--early-stop", action="store_true")
    args = parser.parse_args()

    feature_list = resolve_feature_list(args)
    feature_tag = str(feature_list)
    data_path = Path(args.data)
    artifact_root = Path(args.artifact_root)
    prediction_dir = Path(args.prediction_dir)
    majority_dir = Path(args.majority_dir)

    if args.build_fits or not Path(args.fits).exists():
        print(f"Building source FITS table: {args.fits}")
        script_path = Path(__file__).resolve().parent / "generate_asteroid_database_with_gaia.py"
        subprocess.run([sys.executable, str(script_path)], check=True)

    if args.rebuild_data or not data_path.exists():
        print(f"Building processed data: {data_path}")
        utils.preprocess_data(args.fits, data_path)

    train_data, _, target = utils.load_train_data(
        data_path,
        feature_list,
        artifact_root,
    )

    if args.tune or not (artifact_root / "Tune" / f"{feature_tag}_best.dat").exists():
        print("Tuning expert hyperparameters")
        tune_hyperparameters(
            train_data,
            target,
            n_trials=args.trials,
            epochs=args.tune_epochs,
            feature_list=feature_list,
            artifact_root=artifact_root,
        )

    prediction_dir.mkdir(parents=True, exist_ok=True)
    for run_id in range(1, args.runs + 1):
        print(f"\nRun {run_id}/{args.runs}")
        train_model(
            train_data,
            target,
            num_epochs=args.train_epochs,
            early_stop=args.early_stop,
            patience=20,
            delta=0.005,
            feature_list=feature_list,
            artifact_root=artifact_root,
        )
        output_path = prediction_dir / f"{feature_tag}_run{run_id}.csv"
        test_model(
            data_path,
            artifact_root / "Train" / f"{feature_tag}_model.pth",
            output_path,
            feature_list,
            artifact_root,
        )

    majority_dir.mkdir(parents=True, exist_ok=True)
    majority = load_multirun_majority(prediction_dir, feature_list)
    majority_path = majority_dir / f"{feature_tag}.csv"
    majority.to_csv(majority_path, index=False)
    print(f"\nSaved majority-vote output: {majority_path}")


if __name__ == "__main__":
    main()
