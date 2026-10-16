# Early Prediction of Long-Running Administrative Cases

A Data Science proof of concept for SPAR using **BPI Challenge 2015 Municipality 1**.

The main goal is to predict whether an active administrative case is at risk of becoming long-running.

BPI data is external public process-mining data. It is not SPAR production data.

## Project structure

- `data/raw/`: original dataset files (no dataset included yet).
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

The notebooks currently contain placeholders only. No EDA, feature engineering, model training, or evaluation has been performed. Add the BPI XES file to `data/raw/` before implementing data loading. Raw data is not ignored by Git; decide separately whether to version the dataset.
