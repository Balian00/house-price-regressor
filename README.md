# Ames Housing — Price Regression

Personal project applying classical machine learning (scikit-learn only,
no deep learning) to the Ames Housing dataset.

## Objective

Predict house sale prices (`SalePrice`) from property features (size,
quality, location, etc.). The main goal is to practice a complete and
correct tabular regression workflow — preprocessing, pipelines, cross-
validation, model comparison, error analysis — rather than to chase
leaderboard performance.

## Dataset

Full Ames Housing dataset (De Cock, 2011), 2930 rows / 82 columns, from
[wblakecannon/ames](https://github.com/wblakecannon/ames) (CSV + column
documentation in the same repo). `data/raw/`, not versioned.

## Methodology

- Train/test split performed once, before any statistic is computed
  (imputation, scaling, feature engineering), to avoid data leakage.
- Preprocessing encapsulated in a scikit-learn `Pipeline` /
  `ColumnTransformer`, fitted only inside cross-validation folds.
- Multiple model families compared (linear/regularized, tree-based,
  boosting) via cross-validation before final evaluation on a held-out
  test set.

## Project structure

```
ames-housing-regression/
├── data/raw/                       # raw dataset, never modified
├── notebooks/
│   ├── 01_exploration.ipynb        # EDA (train only, post-split)
│   └── 02_results_summary.ipynb    # results and model comparison
├── src/
│   ├── data_loading.py             # loading + train/test split
│   ├── preprocessing.py            # ColumnTransformer definition
│   ├── models.py                   # pipelines per model family
│   └── evaluation.py               # cross-validation, metrics, selection
├── results/
│   ├── figures/
│   └── metrics/
├── main.py
├── requirements.txt
└── README.md
```

## Installation

**Windows (PowerShell)**
```powershell
python -m venv venv
venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Data license

Educational use, per De Cock (Truman State University). Full documentation
in the [source repository](https://github.com/wblakecannon/ames).

## AI disclosure

This README was written by Claude (Anthropic) at the author's direction,
as part of an AI-assisted learning project on classical ML workflows.