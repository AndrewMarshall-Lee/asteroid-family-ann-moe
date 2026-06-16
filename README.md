# asteroid-family-ann-moe

Code and data for training asteroid-family ANN expert networks and running
predictions with a mixture-of-experts style workflow.

The repository is organised so you can either:

- use the included trained expert-network artifacts, or
- rebuild the source table and tune, train, and test your own expert networks.

## Repository Layout

```text
code/                 Python scripts for data prep, tuning, training, inference, and voting
data/                 Local generated/downloaded data files; not tracked
artifacts/            Saved model weights, scalers, label encoders, and tuning results
outputs/              Generated prediction and majority-vote CSVs
requirements.txt      Python package requirements
```

## Citation And License

If you use this repository, the generated data products, or the trained model
artifacts in academic work, please cite the associated paper when it is
available.

The Python source code is licensed under the MIT License; see `LICENSE`.

Data products, trained model artifacts, fitted preprocessing artifacts, and
generated catalogues are licensed under Creative Commons Attribution 4.0
International (CC BY 4.0); see `DATA_LICENSE.md`.

## Setup

From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

If you are uploading this repository to GitHub, install and enable Git LFS first
for the saved model and preprocessing artifacts:

```bash
git lfs install
git lfs track "*.pth" "*.save"
git add .gitattributes
```

## Available Expert Networks

List the built-in expert feature sets:

```bash
python code/feature_sets.py
```

The named experts are:

```text
base
H
pV
H_pV
A_iz
gaia_color
A_iz_gaia_color
A_iz_pV_gaia_color
H_A_slope
```

All commands below accept either `--expert <name>` for these built-in feature
sets or `--features "[...]"` for a custom Python-list column specification.

## Build The Source FITS Table

The large source FITS table is not tracked in the repository. Build it locally
by downloading the upstream asteroid catalogues and Gaia reflectance data:

```bash
python code/generate_asteroid_database_with_gaia.py
```

This writes:

```text
data/unified_asteroid_data_with_gaia.fits
```

The script downloads source catalogues into `data/`, so this step needs network
access and can take a while.

## Rebuild The Processed Dataset

Create the training table from the generated FITS table:

```bash
python code/utils.py
```

This writes `data/processed_data.csv`. The file is intentionally not tracked in
the repository because it is generated.

## Tune An Expert

Tune hyperparameters with Optuna:

```bash
python code/tune.py \
  --expert H_A_slope \
  --trials 100 \
  --epochs 50
```

This writes:

```text
artifacts/Tune/<feature_list>_best.dat
artifacts/Trials/<feature_list>_trials.dat
```

## Train An Expert

Train using the matching tuned hyperparameters:

```bash
python code/train_moe.py \
  --expert H_A_slope \
  --epochs 300
```

This writes model weights to `artifacts/Train/` and updates the corresponding
scaler and label encoder in `artifacts/Scaler/` and `artifacts/LE/`.

## Run Inference

Use a trained expert network to classify all catalogue rows that have complete
features for that expert:

```bash
python code/run_inference.py \
  --expert H_A_slope \
  --output outputs/H_A_slope_predictions.csv
```

Rows with missing required features are kept in the output and marked
`ANN_reviewed = False`.

## Majority Vote

If you have repeated prediction files from multiple stochastic runs, combine
them with:

```bash
python code/majority_vote.py \
  --expert H_A_slope \
  --prediction-dir outputs/Predictions \
  --output-dir outputs/Majority_Predictions
```

This writes majority-vote CSVs under `outputs/Majority_Predictions/`.

## Full Expert Pipeline

To produce the generated CSVs from scratch for one expert, run:

```bash
python code/run_expert_pipeline.py \
  --expert H_A_slope \
  --build-fits \
  --rebuild-data \
  --tune \
  --trials 100 \
  --tune-epochs 50 \
  --train-epochs 300 \
  --runs 20
```

This runs the full sequence:

```text
downloaded catalogues -> FITS -> data/processed_data.csv -> tuning -> repeated training/inference -> majority vote
```

For a faster rerun using the included tuned parameters, omit `--tune`. If
`data/processed_data.csv` already exists, omit `--rebuild-data`.

## Quick Pipeline Test

For a fast end-to-end pipeline check without overwriting the included artifacts:

```bash
python code/tune.py \
  --expert base \
  --artifact-root /tmp/asteroid-ann-smoke-artifacts \
  --trials 1 \
  --epochs 1

python code/train_moe.py \
  --expert base \
  --artifact-root /tmp/asteroid-ann-smoke-artifacts \
  --epochs 1
```

## Notes

Training and tuning are stochastic. Exact reproduction of the included
predictions should use the saved artifacts in `artifacts/`; new tuning/training
runs may produce slightly different models.
