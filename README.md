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

`requirements.txt` records the package versions used for this release. Other
nearby versions may work, but the pinned file is the recommended starting point.

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

| Expert | Additional features beyond proper elements |
| --- | --- |
| `base` | none |
| `H` | absolute magnitude |
| `pV` | visible albedo |
| `H_pV` | absolute magnitude, visible albedo |
| `A_iz` | SDSS `a*`, SDSS `i-z` colour |
| `gaia_color` | Gaia spectral slope, Gaia `z-i` colour |
| `A_iz_gaia_color` | SDSS colour and Gaia colour features |
| `A_iz_pV_gaia_color` | visible albedo, SDSS colour, and Gaia colour features |
| `H_A_slope` | absolute magnitude, SDSS `a*`, Gaia spectral slope |

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

## Data Provenance

The source table is assembled from public asteroid catalogues, including MPCORB
identifiers/orbital elements, Nesvorny family/proper-element catalogues, SDSS
MOC photometry, NEOWISE diameters/albedos, PRIMASS spectral parameters, and Gaia
DR3 asteroid reflectance spectra. The builder downloads these sources into the
local `data/` directory and writes the merged FITS table.

## Rebuild The Processed Dataset

Create the training table from the generated FITS table:

```bash
python code/utils.py
```

This writes `data/processed_data.csv`. The file is intentionally not tracked in
the repository because it is generated.

For the release dataset used while preparing this repository, the processed
training table contained approximately:

```text
321705 rows
31 columns
53 family labels
```

Because the source-builder downloads live upstream catalogues, exact counts may
change slightly if those catalogues are updated.

## Tune An Expert

Tune hyperparameters with Optuna:

```bash
python code/tune.py \
  --expert H_A_slope \
  --trials 100 \
  --epochs 50 \
  --seed 42
```

This writes:

```text
artifacts/Tune/<expert>_best.dat
artifacts/Trials/<expert>_trials.dat
```

## Train An Expert

Train using the matching tuned hyperparameters:

```bash
python code/train_moe.py \
  --expert H_A_slope \
  --epochs 300 \
  --seed 42
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

Inference output keeps the catalogue columns and appends:

```text
ANN_reviewed
prediction_confidence
ANN_predicted_family_id
```

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

Majority-vote output adds vote diagnostics including:

```text
n_votes
vote_top_frac
vote_entropy
n_unique_votes
prediction_confidence_mean
```

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
  --runs 20 \
  --seed 42
```

This runs the full sequence:

```text
downloaded catalogues -> FITS -> data/processed_data.csv -> tuning -> repeated training/inference -> majority vote
```

For a faster rerun using the included tuned parameters, omit `--tune`. If
`data/processed_data.csv` already exists, omit `--rebuild-data`.

## Quick Pipeline Test

After `data/processed_data.csv` exists, this gives a fast training-pipeline
check without overwriting the included artifacts:

```bash
python code/tune.py \
  --expert base \
  --artifact-root /tmp/asteroid-ann-pipeline-artifacts \
  --trials 1 \
  --epochs 1 \
  --seed 42

python code/train_moe.py \
  --expert base \
  --artifact-root /tmp/asteroid-ann-pipeline-artifacts \
  --epochs 1 \
  --seed 42
```

## Notes

Training and tuning are stochastic unless `--seed` is provided. Exact
reproduction of the included predictions should use the saved artifacts in
`artifacts/`; new tuning/training runs may produce slightly different models.
