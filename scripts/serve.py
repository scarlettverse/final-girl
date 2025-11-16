# scripts/serve.py
# Minimal Flask app to serve predictions as a web service

from flask import Flask, request, jsonify
import joblib
import pandas as pd
from prepare_data import prep_data
from config import MODEL_PATH

app = Flask(__name__)

# Load model once at startup
model = joblib.load(MODEL_PATH)

# Get expected features from training
X_train_enc, _, _, _ = prep_data()
expected_features = X_train_enc.columns

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

    # Convert to DataFrame
    df = pd.DataFrame([data])

    # Conditional encoding
    cat_cols = [c for c in ["type", "status"] if c in df.columns]
    df_enc = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    df_enc = df_enc.select_dtypes(include=["number"])
    df_enc = df_enc.reindex(columns=expected_features, fill_value=0)

    # Predict
    prediction = int(model.predict(df_enc)[0])
    probability = float(model.predict_proba(df_enc)[0, 1])

    return jsonify({
        "prediction": prediction,
        "probability_final_girl": probability
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
