"""Model pipeline definitions for the Ames Housing regression project.

Defines one scikit-learn Pipeline per model family, each combining a
shared preprocessor (ColumnTransformer) with a distinct estimator.
Hyperparameter tuning is not handled here; it is performed downstream
in evaluation.py via GridSearchCV, using the `model__<param>` syntax
to reach into each pipeline's estimator step.
"""

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer


def get_all_pipelines(preprocessor: ColumnTransformer) -> dict[str, Pipeline]:
    """Build one pipeline per model family, sharing the same preprocessor.

    Parameters
    ----------
    preprocessor :
        Fitted-on-call ColumnTransformer (built via build_preprocessor)
        that handles numerical, nominal, and ordinal feature branches.
        The same instance is reused across all pipelines below; each
        pipeline fits its own copy internally during cross-validation,
        so sharing the instance here is safe and avoids redundant
        reconstruction.

    Returns
    -------
    dict[str, Pipeline] :
        Mapping from model family name to its corresponding Pipeline,
        each with a "preprocessor" step and a "model" step holding the
        estimator at default hyperparameters.
    """
    return {
            "ols" : Pipeline([
                ("preprocessor", preprocessor),
                ("model", LinearRegression())
            ]),
            "ridge" : Pipeline([
                ("preprocessor", preprocessor),
                ("model", Ridge())
            ]),
            "lasso" : Pipeline([
                ("preprocessor", preprocessor),
                ("model", Lasso())
            ]),
            "random_forest" : Pipeline([
                ("preprocessor", preprocessor),
                ("model", RandomForestRegressor())
            ]),
            "gradient_boosting" : Pipeline([
                ("preprocessor", preprocessor),
                ("model", GradientBoostingRegressor())
            ]),
        }