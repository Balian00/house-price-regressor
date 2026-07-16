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
from sklearn.compose import TransformedTargetRegressor
from sklearn.preprocessing import StandardScaler

def get_all_pipelines(preprocessor: ColumnTransformer) -> dict[str, TransformedTargetRegressor]:
    """Build one pipeline per model family, sharing the same preprocessor.

    Each pipeline is wrapped in a TransformedTargetRegressor that scales
    y (StandardScaler) before fitting and inverse-transforms predictions
    back to the original dollar scale. This avoids convergence issues in
    penalized linear models (e.g. Lasso) caused by the large, unscaled
    magnitude of SalePrice relative to standardized features.

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
    dict[str, TransformedTargetRegressor] :
        Mapping from model family name to its corresponding
        TransformedTargetRegressor, each wrapping a Pipeline with a
        "preprocessor" step and a "model" step holding the estimator
        at default hyperparameters.
    """
    return {
            "ols" : TransformedTargetRegressor(
                regressor=Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", LinearRegression())
                ]),
                transformer=StandardScaler()
            ),
            "ridge" : TransformedTargetRegressor(
                regressor=Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", Ridge())
                ]),
                transformer=StandardScaler()
            ),
            "lasso" : TransformedTargetRegressor(
                regressor=Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", Lasso())
                ]),
                transformer=StandardScaler()
            ),
            "random_forest" : TransformedTargetRegressor(
                regressor=Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", RandomForestRegressor())
                ]),
                transformer=StandardScaler()
            ),
            "gradient_boosting" : TransformedTargetRegressor(
                regressor=Pipeline([
                    ("preprocessor", preprocessor),
                    ("model", GradientBoostingRegressor())
                ]),
                transformer=StandardScaler()
            ),
        }