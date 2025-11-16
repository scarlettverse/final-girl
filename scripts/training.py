# scripts/training.py
# Logistic Regression survival model
# Imports engineered dataset from prepare_data.py
# Restores guardrails (flattening, safe feature restriction, numeric filtering)
# Notes included for clarity

from prepare_data import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

RANDOM_SEED = 42

def train_logistic():
    # === Load engineered dataset ===
    df = get_dataset()
    
    # --- Survival diagnostics ---
    print("Unique values in status column:")
    print(df["status"].value_counts())

    if "in_production" in df.columns:
        print("\nUnique values in in_production column:")
        print(df["in_production"].value_counts())

    print("\nSurvival split (Final Girl vs Slashed):")
    print(df["is_final_girl"].value_counts())

    print("\nAverage values for survivors vs slashed:")
    print(df.groupby("is_final_girl")[["number_of_seasons", "vote_average", "popularity"]].mean())

    # === Defensive flattening ===
    # If any list-type columns slipped through (e.g. genre_list, network_list),
    # flatten them into strings to prevent scikit-learn errors.
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: ", ".join(x) if isinstance(x, list) else x)

    # === Feature selection ===
    # Safe numeric + categorical features
    base_features = ["type", "status", "runtime", "popularity", "vote_average", "vote_count"]

    # Engineered flags (genres, networks, companies, audience signals, tiers, composite scores)
    engineered_flags = [
        col for col in df.columns
        if col.startswith(("has_", "is_", "scream_queen_score", "final_girl_index"))
    ]

    # Restrict to safe + engineered features
    X = df[[c for c in base_features + engineered_flags if c in df.columns]]
    y = df["is_final_girl"]

    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y if y.sum() > 0 else None
    )

    # === Encode categorical features ===
    # Only low-cardinality features ("type", "status") are one-hot encoded.
    cat_cols = [c for c in ["type", "status"] if c in X.columns]
    X_encoded = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # === Numeric filter ===
    # Keep only numeric columns to prevent "could not convert string to float" errors.
    X_encoded_clean = X_encoded.select_dtypes(include=["number"])

    # Align train/test indices
    X_train_enc = X_encoded_clean.loc[X_train.index]
    X_test_enc = X_encoded_clean.loc[X_test.index]

    # === Train model ===
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train_enc, y_train)

    # === Predictions ===
    y_pred = model.predict(X_test_enc)
    y_prob = model.predict_proba(X_test_enc)[:, 1]

    # === Metrics ===
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else float("nan")
    }

    return model, metrics

if __name__ == "__main__":
    model, metrics = train_logistic()
    print("=== Logistic Regression Baseline ===")
    print(f"Accuracy: {metrics['accuracy']:.3f}")
    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall: {metrics['recall']:.3f}")
    print(f"F1 Score: {metrics['f1']:.3f}")
    print(f"AUC: {metrics['auc']:.3f}")
    print("🩸 Commentary: This baseline is stable and notebook-faithful. "
          "It restores guardrails to prevent error loops while honoring the full Final Girl feature set.")
