# train_pipeline.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from typing import Tuple, List, Dict

BASE_FEATURES = [
    # Planetary properties
    'koi_period', 'koi_duration', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_impact', 'koi_model_snr',
    
    # Stellar properties  
    'koi_steff', 'koi_slogg', 'koi_srad', 'feh', 'mass', 'dens',
    
    # Observational
    'koi_kepmag',
    
    # Key uncertainties
    'koi_period_err1', 'koi_prad_err1', 'koi_steff_err1'
]

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans a dataframe by:
    - Dropping duplicates
    - Filling numeric NaNs with mean
    - Filling categorical NaNs with mode
    """
    df = df.drop_duplicates()
    numeric_columns = df.select_dtypes(include=["int64", "float64"]).columns
    df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].mean())
    categorical_columns = df.select_dtypes(include=["object"]).columns
    for col in categorical_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df

def merge_datasets(
    stellar: pd.DataFrame,
    toi: pd.DataFrame,
    fpp: pd.DataFrame,
    tce: pd.DataFrame,
    koi: pd.DataFrame,
    fpp_threshold: float = 0.5
) -> pd.DataFrame:
    """
    Merges multiple exoplanet datasets into a master dataframe.
    Filters FPP below threshold and encodes target.
    """
    # Remove unnamed columns
    for df in [stellar, toi, fpp, tce, koi]:
        df = df.loc[:, ~df.columns.str.contains('^Unnamed')]

    # Filter FPP
    fpp_filtered = fpp[fpp['fpp_prob'] <= fpp_threshold]

    # Merge KOI with FPP
    koi = koi.merge(fpp_filtered[['kepid','fpp_prob']], on='kepid', how='left')
    koi = koi.dropna(subset=['fpp_prob'])

    # Merge with stellar
    master = koi.merge(stellar, on='kepid', how='left')

    # Merge TCE
    master = master.merge(
        tce[['kepid', 'tce_period', 'tce_duration', 'tce_depth', 'tce_model_snr']],
        on='kepid', how='left'
    )

    # Drop rows with essential missing values
    master = master.dropna(subset=['teff', 'radius', 'koi_period', 'koi_prad'])
    master = master.drop_duplicates()
    master.reset_index(drop=True, inplace=True)

    # Encode target
    le = LabelEncoder()
    master['koi_disposition_encoded'] = le.fit_transform(master['koi_disposition'])

    return master

def train_model(
    df: pd.DataFrame,
    target_col: str = "koi_disposition_encoded",
    id_col: str = "kepoi_name",
    n_estimators: int = 500,
    random_state: int = 42,
    test_size: float = 0.2
) -> Tuple[RandomForestClassifier, List[str], dict]:
    """
    Trains RandomForestClassifier on the cleaned dataframe
    using only the fixed 58 base features.
    Returns model, feature columns, and metrics.
    """
    # Encode target if not numeric
    if df[target_col].dtype == "object":
        le = LabelEncoder()
        df[target_col] = le.fit_transform(df[target_col])
    # Drop 'kepid' column if present
    if 'kepid' in df.columns:
        df = df.drop(columns=['kepid'])
    # Keep only the base features (drop if some missing)
    available_features = [f for f in BASE_FEATURES if f in df.columns]
    X = df[available_features]
    y = df[target_col]

    # Stratified split by unique IDs
    unique_ids = df[id_col].unique()
    id_to_label = df.groupby(id_col)[target_col].first()
    train_ids, test_ids = train_test_split(
        unique_ids,
        test_size=test_size,
        random_state=random_state,
        stratify=id_to_label.loc[unique_ids]
    )
    train_mask = df[id_col].isin(train_ids)
    test_mask = df[id_col].isin(test_ids)
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    # Train RandomForest
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        class_weight="balanced",
        random_state=random_state
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred).tolist()  # make JSON serializable
    cr = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "confusion_matrix": cm,
        "classification_report": cr
    }

    return model, available_features, metrics