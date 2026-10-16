# Early Prediction of Long-Running Administrative Cases

A Data Science proof of concept for SPAR using **BPI Challenge 2015 Municipality 1**.

The main goal is to predict whether an active administrative case is at risk of becoming long-running.

BPI data is external public process-mining data. It is not SPAR production data.

## Project structure

- `data/raw/`: local original dataset files (excluded from Git).
- `data/processed/`: generated datasets.
- `notebooks/`: data loading, EDA, feature engineering, modeling, and evaluation stages.
- `src/`: reusable Python modules.
- `outputs/figures/` and `outputs/tables/`: generated results.

## Environment

On Windows PowerShell, activate the existing environment with:

```powershell
.\.venv\Scripts\Activate.ps1
```

To recreate the environment with Python installed:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Dataset setup

1. Download BPI Challenge 2015 Municipality 1 from the official BPI Challenge 2015 / 4TU Research Data source.
2. Place `BPIC15_1.xes` in `data/raw/`.
3. The raw dataset is intentionally excluded from Git and is not SPAR production data.
4. Run `notebooks/01_data_loading.ipynb` first using the project `.venv`.

Notebook 01 performs dataset intake and structural validation only. The remaining notebooks contain placeholders. No full EDA, target definition, feature engineering, or model training has been performed.
