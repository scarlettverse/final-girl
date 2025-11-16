# scripts/training.py
# Logistic Regression & Decision Tree survival models
# Imports engineered dataset from prepare_data.py
# Restores guardrails (flattening, safe feature restriction, numeric filtering)
# Notes included for clarity

from prepare_data import get_dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import pandas as pd

RANDOM_SEED = 42

def prep_data():
    # === Load engineered dataset ===
    df = get_dataset()
    
    # --- Survival diagnostics ---
    print("Unique values in status column:")
    print(df["status"].value_counts())

    if "in_production" in df.columns:
        print("\nUnique values in in_production column:")
        print(df["in_production"].value_counts())

    print("\nSurvival split (Final Girl vs Scream Queen):")
    print(df["is_final_girl"].value_counts())

    print("\nAverage values for Final Girl vs Scream Queen:")
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
        if col.startswith(("has_", "is_", "killer_karma", "last_laughs"))
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
    
    return X_train_enc, X_test_enc, y_train, y_test

    # === Train models ===
def train_logistic(X_train, y_train, X_test, y_test):
    model = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else float("nan")
    }
    return model, metrics

def train_decision_tree(X_train, y_train, X_test, y_test):
    model = DecisionTreeClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob) if len(set(y_test)) > 1 else float("nan")
    }
    return model, metrics

def train_random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob)
    }
    return model, metrics

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    model = GradientBoostingClassifier(random_state=RANDOM_SEED)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "auc": roc_auc_score(y_test, y_prob)
    }
    return model, metrics

if __name__ == "__main__":
    X_train_enc, X_test_enc, y_train, y_test = prep_data()
    
    # Logistic Regression
    log_model, log_metrics = train_logistic(X_train_enc, y_train, X_test_enc, y_test)
    print("\n=== Logistic Regression Baseline ===")
    for k, v in log_metrics.items():
        print(f"{k.capitalize()}: {v:.3f}")
        
    # Decision Tree
    dt_model, dt_metrics = train_decision_tree(X_train_enc, y_train, X_test_enc, y_test)
    print("\n=== Decision Tree Baseline ===")
    for k, v in dt_metrics.items():
        print(f"{k.capitalize()}: {v:.3f}")
        
    # Random Forest
    rf_model, rf_metrics = train_random_forest(X_train_enc, y_train, X_test_enc, y_test)
    print("\n=== Random Forest Baseline ===")
    for k, v in rf_metrics.items():
        print(f"{k.capitalize()}: {v:.3f}")
        
    # Gradient Boosting
    gb_model, gb_metrics = train_gradient_boosting(X_train_enc, y_train, X_test_enc, y_test)
    print("\n=== Gradient Boosting Baseline ===")
    for k, v in gb_metrics.items():
        print(f"{k.capitalize()}: {v:.3f}")
        
    print("🩸 Commentary: This baseline is stable and notebook-faithful. "
          "It restores guardrails to prevent error loops while honoring the full Final Girl feature set.")
