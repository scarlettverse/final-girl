# testing.py
# Here's where I test the code as I refactor

import pandas as pd
from scripts.prepare_data import prepare_data
from scripts.features import create_features

if __name__ == "__main__":
    # Load your actual dataset
    df = pd.read_csv("data/TMDB.csv")

    # Run only the genre features
    df = create_features(df)

# Final summary snapshot
flag_columns = [
    # Genres
    "has_drama", "has_comedy", "has_documentary", "has_animation", "has_reality",
    "has_crime", "has_family", "has_sci-fi_and_fantasy",
    # Networks
    "is_netflix", "is_youtube", "is_bbc_one", "is_abc", "is_prime_video",
    "is_nbc", "is_cbs", "is_fox",
    # Companies
    "has_tvb", "has_bbc", "has_warner_bros_television",
    "has_universal_television", "has_amazon_studios",
    # Popularity
    "is_highly_rated", "is_popular", "is_high_quality",
    "is_obscure", "is_moderately_popular", "is_hyped",
    # Episodes
    "is_miniseries", "is_mid_length", "is_long_running"
]



print("\n=== Final Feature Snapshot ===")
print(df[["name", "number_of_episodes", "vote_average", "vote_count", "popularity"] + flag_columns].sample(5, random_state=42))

# --- Survival Rule Snapshot ---
print("\n=== Survival Rule Snapshot ===")
print(df[["name", "killer_karma", "last_laughs"]].sample(5, random_state=42))

# --- Correlation Analysis ---
correlation_features = df.select_dtypes(include=["bool", "int64", "float64"])
print("\n=== Correlation Features Included ===")
print(correlation_features.columns.tolist())

if "is_final_girl" in df.columns:
    correlation_matrix = df[correlation_features.columns].corr()
    print("\n=== Correlation with Survival ===")
    print(correlation_matrix["is_final_girl"].sort_values(ascending=False))
else:
    print("\n[!] 'is_final_girl' column not found in dataset. Skipping survival correlation.")
    correlation_matrix = df[correlation_features.columns].corr()
    print("\n=== General Correlation Matrix (no survival flag) ===")
    print(correlation_matrix.head())
