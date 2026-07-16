from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
import time
from sklearn.compose import TransformedTargetRegressor
import joblib
from pathlib import Path

error = 2
ERROR = "neg_root_mean_squared_error" if error == 2 else "neg_mean_absolute_error"


hyperparameters = {
    "ols" : {},
    "ridge" : {"regressor__model__alpha" : [0.001, 0.01, 0.1, 1, 10, 100]},
    "lasso" : {"regressor__model__alpha" : [0.001, 0.01, 0.1, 1, 10, 100],
               "regressor__model__max_iter" : [10000]},
    "random_forest": {
        "regressor__model__n_estimators": [100, 300, 500],
        "regressor__model__max_features": ["sqrt", "log2", None],
        "regressor__model__max_depth": [None, 10, 20, 30],
        "regressor__model__min_samples_leaf": [1, 3, 5, 10]
    },
    "gradient_boosting": {
        "regressor__model__n_estimators": [100, 300, 500],
        "regressor__model__learning_rate": [0.01, 0.05, 0.1],
        "regressor__model__max_depth": [2, 3, 4]
    }
}

def tune_all_pipelines(pipelines : dict[str, TransformedTargetRegressor], X_train : pd.DataFrame, y_train : pd.DataFrame) -> dict[str, GridSearchCV] :
    results = {}

    for name, pipeline in pipelines.items():
        grid = hyperparameters[name]
        print(f"Tuning {name}...")
        start = time.time()

        search = GridSearchCV(estimator=pipeline, param_grid=grid, cv=5, scoring=ERROR, verbose=1)
        search.fit(X_train, y_train)

        elapsed = time.time() - start
        print(f"{name} done in {elapsed:.1f}s")

        results[name] = search

    return results

def select_best_model(results: dict[str, GridSearchCV]) -> tuple[str, TransformedTargetRegressor]:
    best_score = float("-inf")
    best_name = None

    for name, search in results.items():
        score = search.best_score_
        if score > best_score:
            best_score = score
            best_name = name

    return best_name, results[best_name].best_estimator_
    
def evaluate_on_test(best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.DataFrame) -> float:
    y_predict = best_model.predict(X_test)

    if error == 2:
        return root_mean_squared_error(y_test, y_predict)
    else:
        return mean_absolute_error(y_test, y_predict)

MODEL_DIR = "results/models"

def save_results(results: dict[str, GridSearchCV], best_model: Pipeline) -> None:
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, f"{MODEL_DIR}/best_model.joblib")
    joblib.dump(results, f"{MODEL_DIR}/tuning_results.joblib")

def load_best_model() -> Pipeline:
    return joblib.load(f"{MODEL_DIR}/best_model.joblib")

def load_tuning_results() -> dict[str, GridSearchCV]:
    return joblib.load(f"{MODEL_DIR}/tuning_results.joblib")
