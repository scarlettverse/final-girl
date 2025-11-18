# scripts/serve.py
# To run: python scripts/serve.py
# Minimal Flask app to serve predictions as a web service

from flask import Flask, request, jsonify, Response
import json
import joblib
import pandas as pd
from prepare_data import prep_data, get_dataset
from config import MODEL_PATH

app = Flask(__name__)

# Load model once at startup
model = joblib.load(MODEL_PATH)

# Get expected features from training
X_train_enc, _, _, _ = prep_data()
expected_features = X_train_enc.columns

# Load lookup dataset once
df_lookup = get_dataset()

def get_features_by_title(show_name):
    row = df_lookup[df_lookup["name"] == show_name]
    if row.empty:
        return None
    return row.drop(columns=["name"]).iloc[0].to_dict()

@app.route("/predict", methods=["POST"])
def predict():
    """
    Accepts JSON payload with show data and returns prediction + probability.
    Example input:
    {
        "number_of_seasons": 2,
        "vote_average": 7.5,
        "popularity": 12.3,
        "status": "Ended"
    }
    """
    data = request.get_json()
    # === Title Mode ===
    if "title" in data:
        features = get_features_by_title(data["title"])
        if features is None:
            return jsonify({"error": f"Show '{data['title']}' not found in dataset"}), 404
        df = pd.DataFrame([features])
    else:
        # === Raw Features Mode ===
        df = pd.DataFrame([data])

    # Conditional encoding
    cat_cols = [c for c in ["type", "status"] if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_enc = df_enc.select_dtypes(include=["number"])
    df_enc = df_enc.reindex(columns=expected_features, fill_value=0)

    # Predict
    prediction = int(model.predict(df_enc)[0])
    probability = float(model.predict_proba(df_enc)[0, 1])
    
    # Lore-coded classification and narrator line
    fate = "Final Girl" if prediction == 1 else "Scream Queen"
    randy_rule = (
        "She learned the rules and lived to tell the tale"
        if prediction == 1
        else "Never say I\'ll be right back"
    )
    
    # Pull engineered features if this is Title Mode
    killer_karma = None
    last_laughs = None
    victim = data.get("title", "")
    if "title" in data:
        # Use composite scores from engineered features
        killer_karma = features.get("killer_karma")
        last_laughs = features.get("last_laughs")

    return Response(
        json.dumps({
        "victim": victim,
        "fate": fate,
        "final_girl_grit": round(probability, 2),
        "killer_karma": killer_karma,
        "last_laughs": last_laughs,
        "randy_rule": randy_rule
    }, indent=2), # pretty-print with 2 spaces
    mimetype="application/json"
    )

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
