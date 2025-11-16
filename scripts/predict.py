# scripts/predict.py
# Loads the saved Gradient Boosting model and makes a single prediction
# Runs with: python scripts/predict.py

import sys
import joblib
import pandas as pd
from prepare_data import prep_data
from config import MODEL_PATH, SHOWS

def prediction(tv_show_path=SHOWS):
    # === Load trained model ===
    model = joblib.load(MODEL_PATH)

    # === Get training columns === (so we know the exact feature set)
    X_train_enc, _, _, _ = prep_data()
    expected_features = X_train_enc.columns

    # === Load Data CSV ===
    df = pd.read_csv(tv_show_path)

    # === Apply same preprocessing pipeline ===
    cat_cols = [c for c in ["type", "status"] if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_enc = df_enc.select_dtypes(include=["number"])
    df_enc = df_enc.reindex(columns=expected_features, fill_value=0)

    # === Predict ===
    predictions = model.predict(df_enc)
    probabilities = model.predict_proba(df_enc)[:, 1]
    
    # === Output results ===
    results = pd.DataFrame({
        "prediction": predictions,
        "probability_final_girl": probabilities
    }, index=df.index)
    
    # === Summary metrics ===
    avg_prob = probabilities.mean()
    final_girl_count = (predictions == 1).sum()
    scream_queen_count = (predictions == 0).sum()

    print("\n=== Summary Metrics ===")
    print(f"Average survival probability: {avg_prob:.3f}")
    print(f"Final Girl predictions: {final_girl_count}")
    print(f"Scream Queen predictions: {scream_queen_count}")
    
    # === Save predictions to CSV ===
    results.to_csv("predictions.csv", index=False)
    print("\nPredictions saved to predictions.csv")
    print(results.head())
    print("\n🩸 Commentary: Each row’s fate is revealed with probability of survival.")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        prediction(SHOWS)
    else:
        prediction(sys.argv[1])