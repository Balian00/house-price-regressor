"""
Preprocessing pipeline for the Ames Housing dataset.

Handles structural missing values (NaN encoding the absence of a feature,
e.g. no basement, rather than unknown data) via fixed, hand-written rules,
then builds a scikit-learn ColumnTransformer that encodes/scales each
column group (ordinal, nominal, continuous, discrete) appropriately.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

ORDINAL_STRUCTURAL_NA = {
    "Bsmt Qual": "Po",
    "Bsmt Cond": "Po",
    "Bsmt Exposure": "No",
    "BsmtFin Type 1": "Unf",
    "BsmtFin Type 2": "Unf",
    "Garage Finish": "Unf",
    "Garage Qual": "Po",
    "Garage Cond": "Po",
    "Fireplace Qu": "Po",
    "Pool QC": "Fa",
    "Fence": "MnWw",
}

NOMINAL_STRUCTURAL_NA = {
    "Garage Type": "None",
    "Alley": "None",
    "Misc Feature": "None",
}

HAS_INDICATOR_GROUPS = {
    "Has_Basement": ["Bsmt Qual", "Bsmt Cond", "Bsmt Exposure", "BsmtFin Type 1", "BsmtFin Type 2"],
    "Has_Garage": ["Garage Type", "Garage Finish", "Garage Qual", "Garage Cond"],
    "Has_Fireplace": ["Fireplace Qu"],
    "Has_Pool": ["Pool QC"],
    "Has_Fence": ["Fence"],
    "Has_Alley": ["Alley"],
    "Has_MiscFeature": ["Misc Feature"],
}


def add_structural_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a binary Has_X indicator for each group in HAS_INDICATOR_GROUPS.

    Parameters
    ----------
    df
        Input data containing the columns referenced in HAS_INDICATOR_GROUPS.

    Returns
    -------
        Copy of `df` with one Has_X column added per group, derived from
        the NaN pattern of the group's first column (before that NaN is
        filled). Assumes all columns within a group share the same NaN rows.
    """
    df = df.copy()
    for indicator_name, columns in HAS_INDICATOR_GROUPS.items():
        reference_column = columns[0]
        df[indicator_name] = df[reference_column].notna().astype(int)
    return df


def fill_structural_na(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace structural NaN with a fixed neutral placeholder.

    Parameters
    ----------
    df
        Input data containing the columns listed in ORDINAL_STRUCTURAL_NA
        and NOMINAL_STRUCTURAL_NA.

    Returns
    -------
        Copy of `df` with structural NaN filled by the worst ordinal level
        (ordinal columns) or "None" (nominal columns). Must run after
        `add_structural_indicators`, since the Has_X columns depend on the
        NaN pattern this function removes.
    """
    df = df.copy()
    for column, fill_value in {**ORDINAL_STRUCTURAL_NA, **NOMINAL_STRUCTURAL_NA}.items():
        df[column] = df[column].fillna(fill_value)
    return df


def apply_structural_na_handling(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full structural NaN handling: add Has_X indicators, then fill
    the NaN they were derived from.

    Parameters
    ----------
    df
        Input data (X_train or X_test).

    Returns
    -------
        Copy of `df` with Has_X indicators added and structural NaN filled.
        Call once per split independently, never on the two combined.
    """
    df = add_structural_indicators(df)
    df = fill_structural_na(df)
    return df

ORDINAL_CATEGORIES = {
    "Overall Qual": list(range(1, 11)),         
    "Overall Cond": list(range(1, 11)),
    "Exter Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Exter Cond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Kitchen Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Heating QC": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Functional": ["Sal", "Sev", "Maj2", "Maj1", "Mod", "Min2", "Min1", "Typ"],
    "Paved Drive": ["N", "P", "Y"],
    "Lot Shape": ["IR3", "IR2", "IR1", "Reg"],
    "Utilities": ["ELO", "NoSeWa", "NoSewr", "AllPub"],
    "Land Slope": ["Sev", "Mod", "Gtl"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Bsmt Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Cond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Bsmt Exposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFin Type 1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFin Type 2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Garage Finish": ["Unf", "RFn", "Fin"],
    "Garage Qual": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Garage Cond": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Fireplace Qu": ["Po", "Fa", "TA", "Gd", "Ex"],
    "Pool QC": ["Fa", "TA", "Gd", "Ex"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

ORDINAL_COLUMNS = list(ORDINAL_CATEGORIES.keys())

CONTINUOUS_COLUMNS = [
    "Lot Area", "Lot Frontage", "Gr Liv Area",
    "Garage Area", "Total Bsmt SF",
]

NUMERIC_BUT_NOMINAL = ["MS SubClass"]


def get_column_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str], list[str]]:
    """
    Split the DataFrame columns into ordinal, continuous, nominal, and
    discrete groups based on the module-level column lists.

    Parameters
    ----------
    df
        Input data whose columns are to be classified.

    Returns
    -------
    ordinal_columns
        Columns from ORDINAL_COLUMNS present in `df`.
    continuous_columns
        Columns from CONTINUOUS_COLUMNS present in `df`.
    nominal_columns
        Object-dtype columns not in `ordinal_columns`, plus NUMERIC_BUT_NOMINAL.
    discrete_columns
        Remaining numeric columns not classified as ordinal or continuous.
    """
    ordinal_columns = []
    for c in ORDINAL_COLUMNS:
        if c in df.columns:
            ordinal_columns.append(c)

    continuous_columns = []
    for c in CONTINUOUS_COLUMNS:
        if c in df.columns:
            continuous_columns.append(c)

    nominal_columns = []
    for col in df.select_dtypes(include="object").columns:
        if col not in ordinal_columns:
            nominal_columns.append(col)

    for col in NUMERIC_BUT_NOMINAL: # juste one here
        if col in df.columns:
            nominal_columns.append(col)

    discrete_columns = [] # all remaining colmuns a priori
    for col in df.select_dtypes(include="number").columns:
        is_ordinal = col in ordinal_columns
        is_continuous = col in continuous_columns
        is_numeric_but_nominal = col in NUMERIC_BUT_NOMINAL

        if not is_ordinal and not is_continuous and not is_numeric_but_nominal:
            discrete_columns.append(col)

    return ordinal_columns, continuous_columns, nominal_columns, discrete_columns

def build_nominal_pipeline() -> Pipeline:
    """
    Returns
    -------
        Imputes missing values with the most frequent category, then
        one-hot encodes.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])


def build_continuous_pipeline() -> Pipeline:
    """
    Returns
    -------
        Imputes missing values with the median, then scales to zero mean
        and unit variance.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_discrete_pipeline() -> Pipeline:
    """
    Returns
    -------
        Imputes missing values with the median, then scales to zero mean
        and unit variance.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])


def build_preprocessor(df: pd.DataFrame) -> ColumnTransformer:
    """
    Build the full preprocessing ColumnTransformer for the dataset.

    Parameters
    ----------
    df
        Input data used to determine which columns fall into each group.

    Returns
    -------
        Ordinal-encodes ordinal columns, one-hot encodes nominal columns,
        and scales continuous/discrete columns.
    """
    ordinal_columns, continuous_columns, nominal_columns, discrete_columns = get_column_groups(df)

    ordinal_categories = [ORDINAL_CATEGORIES[col] for col in ordinal_columns]

    ordinal_pipeline = Pipeline([
        ("encoder", OrdinalEncoder(categories=ordinal_categories)),
    ])

    preprocessor = ColumnTransformer([
        ("ordinal", ordinal_pipeline, ordinal_columns),
        ("nominal", build_nominal_pipeline(), nominal_columns),
        ("continuous", build_continuous_pipeline(), continuous_columns),
        ("discrete", build_discrete_pipeline(), discrete_columns),
    ])

    return preprocessor