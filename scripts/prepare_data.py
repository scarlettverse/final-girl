# scripts/data_prep.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from scripts.config import RANDOM_SEED, TARGET, TEST_SIZE

def prepare_data(df):
    """
    Splits the dataset into train/test sets and scales features.
    
    Parameters:
        df (pd.DataFrame): Input dataframe with target column.
    
    Returns:
        X_train_scaled, X_test_scaled, y_train, y_test, scaler
    """
    # Separate features and target
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        index=X_train.index,
        columns=X_train.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        index=X_test.index,
        columns=X_test.columns
    )

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
