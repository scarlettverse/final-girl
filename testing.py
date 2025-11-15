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

