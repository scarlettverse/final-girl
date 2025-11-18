# scripts/predict.py
# Loads the saved Gradient Boosting model and makes a single prediction
# Runs with: python scripts/predict.py

import os
import sys
import joblib
import pandas as pd
from prepare_data import prep_data, get_dataset
from config import MODEL_PATH, SHOWS

# === Title → features lookup ===
df_lookup = get_dataset()

def get_features_by_title(show_name):
    row = df_lookup[df_lookup["name"] == show_name]
    if row.empty:
        return None
    return row.drop(columns=["name"]).iloc[0].to_dict()

def prediction(show_name=SHOWS):
    # === Load trained model ===
    model = joblib.load(MODEL_PATH)

    # === Get training columns === (so we know the exact feature set)
    X_train_enc, _, _, _ = prep_data()
    expected_features = X_train_enc.columns

    # === Lookup features by show title ===
    features = get_features_by_title(show_name)
    if features is None:
        print(f"[!] Show '{show_name}' not found in dataset.")
        return
    df = pd.DataFrame([features])

    # === Encode categorical features ===
    cat_cols = [c for c in ["type", "status"] if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_enc = df_enc.select_dtypes(include=["number"])
    df_enc = df_enc.reindex(columns=expected_features, fill_value=0)

    # === Predict ===
    pred = int(model.predict(df_enc)[0])
    prob_final = float(model.predict_proba(df_enc)[0, 1])
    
    # === Lore-coded output ===
    fate = "Final Girl" if pred == 1 else "Scream Queen"
    randy_rule = (
        "She learned the rules and lived to tell the tale"
        if pred == 1
        else "Never say I'll be right back"
    )
    
    # === Summary metrics ===
    print("\n=== Fate Revealed ===")
    print(f"Victim: {show_name}")
    print(f"Fate: {fate}")
    print(f"Final Girl Grit: {prob_final:.2f}")
    print(f"Killer Karma: {features.get('killer_karma', 'N/A')}")
    print(f"Last Laughs: {features.get('last_laughs', 'N/A')}")
    print(f"Randy Rule: {randy_rule}")
    
    # === Save predictions to CSV ===
    out_row = {
        "victim": show_name,
        "fate": fate,
        "final_girl_grit": round(prob_final, 4),
        "killer_karma": features.get("killer_karma"),
        "last_laughs": features.get("last_laughs"),
        "randy_rule": randy_rule,
        "prediction_raw": pred
    }
    out_df = pd.DataFrame([out_row])

    out_path = "predictions.csv"
    write_header = not os.path.exists(out_path)
    out_df.to_csv(out_path, mode="w", header=True, index=False)
    
    print(f"\nPredictions appended to {out_path}")
    print(out_df)
    
if __name__ == "__main__":
    if len(sys.argv) < 2:
        prediction(SHOWS)
    else:
        prediction(sys.argv[1])