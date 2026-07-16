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
from src.models import get_all_pipelines
from src.evaluation import tune_all_pipelines, select_best_model, evaluate_on_test, error, save_results, load_best_model, load_tuning_results
import click

@click.command()
@click.option(
    "--mode",
    type=click.Choice(["train", "test"]),
    default="test",
    help="train: tune tous les modèles et sauvegarde le meilleur. test: recharge le meilleur modèle sauvegardé.",
)

def main(mode : str) -> None:
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
    if mode == "train" :
        print("Building preprocessor...")
        preprocessor = build_preprocessor(X_train)
        print("Preprocessor built.")
        print(preprocessor)

        print("Building pipelines...")
        pipelines = get_all_pipelines(preprocessor)
        print(f"{len(pipelines)} pipelines built.")

        print("Tuning all pipelines (this may take a while)...")
        results = tune_all_pipelines(pipelines, X_train, y_train)
        print("Tuning complete.")
        for name, search in results.items():
            print(f"{name}: best_score_ = {search.best_score_}, best_params_ = {search.best_params_}")
        print("Selecting best model...")
        best_model = select_best_model(results)
        print(f"Best model: {best_model}")

        best_name, best_model = select_best_model(results)
        print(f"Best model: {best_name}")
        print("Best model saved.")
        
    elif mode == "test" :
        best_model = load_best_model()

    print("Evaluating on test set...")
    raw_score = evaluate_on_test(best_model, X_test, y_test)
    relative_score = raw_score / y_test.mean() * 100
    metric_name = "RMSE" if error == 2 else "MAE"
    print(f"Final test score ({metric_name}): {raw_score:.2f}$ ({relative_score:.2f}% of mean price)")
        
    

if __name__ == "__main__":
    main()