"""
data_loading.py

Sole responsibility of this file: load the raw dataset and produce a
train/test split, without any statistic computed on the data (no
imputation, no encoding). Any further transformation must be defined
elsewhere (preprocessing.py) and applied only inside cross-validation,
never here.
"""

import pandas as pd
from sklearn.model_selection import train_test_split

def load_raw_data(filepath : str) -> pd.DataFrame :
    """
    Loads the raw CSV file of the Ames Housing dataset.

    Parameters
    ----------
    filepath : str or Path
        Path to the raw CSV file (expected in data/raw/).

    Returns
    -------
    pd.DataFrame
        The full dataset, as is, without any transformation.
        Still contains SalePrice and all missing values.
    """
    return pd.read_csv(filepath, sep=',', na_values=None, dtype=None)


def split_features_target(df: pd.DataFrame, target_column: str = "SalePrice") -> tuple[pd.DataFrame, pd.Series]:
    """
    Splits the DataFrame into features (X) and target (y).

    Parameters
    ----------
    df : pd.DataFrame
        The full dataset, as returned by load_raw_data.
    target_column : str
        Name of the target column to predict.

    Returns
    -------
    X : pd.DataFrame
        All columns except the target.
    y : pd.Series
        The target column alone.
    """
    X = df.drop(columns=target_column)
    y = df[target_column]
    return X, y


def train_test_split_data(X : pd.DataFrame, y : pd.Series, test_size : float = 0.2, random_state : int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series ,pd.Series]:
    """
    Performs the single, definitive train/test split for the project.

    This split must be done only once, before any statistic is computed
    (imputation, scaling, etc.). The resulting test set must not be
    inspected or used before the final evaluation of the selected model.

    Parameters
    ----------
    X : pd.DataFrame
        Full features.
    y : pd.Series
        Full target.
    test_size : float
        Proportion of the dataset reserved for testing (0.2 = 20%).
    random_state : int
        Random seed for split reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : pd.DataFrame / pd.Series
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)