"""
main.py

Entry point of the project. Orchestrates the full pipeline end to end:
load the raw data, split into train/test, define preprocessing and
candidate models, run cross-validation to select the best configuration,
refit on the full training set, and evaluate once on the held-out test
set.

This file only calls functions defined in src/ — it contains no data
transformation logic of its own.
"""

from src.data_loading import load_raw_data, split_features_target, train_test_split_data
from src.preprocessing import build_preprocessor, apply_structural_na_handling
"""
from src.models import get_candidate_models
from src.evaluation import run_cross_validation, evaluate_final_model
import pandas as pd """


def main():
    """
    Runs the full pipeline: load, split, preprocess, select, evaluate.
    """
    print("Reading CSV file...")
    try:
        df = load_raw_data("data/raw/housing.csv")
    except FileNotFoundError:
        print("Error: file not found at data/raw/housing.csv")
        return
    print("Data read.")
    print("Splitting features...")
    X, y = split_features_target(df)
    print(X.shape)
    print("Feature splitted.")
    print("Splitting dataset...")
    X_train, X_test, y_train, y_test = train_test_split_data(X, y)
    print("Dataset splitted.")
    print("Applying structural NaN handling...")
    X_train = apply_structural_na_handling(X_train)
    X_test = apply_structural_na_handling(X_test)
    print("Structural NaN handled.")
    print(X_train.shape)
    print([c for c in X_train.columns if c.startswith("Has_")])

    print("Building preprocessor...")
    preprocessor = build_preprocessor(X_train)
    print("Preprocessor built.")
    print(preprocessor)
    
    

if __name__ == "__main__":
    main()