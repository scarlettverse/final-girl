# scripts/training.py
# Runs with: python scripts/training.py
# Trains survival models on engineered dataset
# Imports preprocessed train/test splits from prepare_data.py
# Notes included for clarity

from prepare_data import prep_data
from config import RANDOM_SEED
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os


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
        
    # === Save best model (Gradient Boosting) ===
    os.makedirs("models", exist_ok=True)
    joblib.dump(gb_model, "models/gradient_boosting.pkl")
    print("\nModel saved to models/gradient_boosting.pkl")

        
    print("🩸 Commentary: I've trained the survivor, now I'm locking her fate into a .pkl artifact that predict.py can read later.")
