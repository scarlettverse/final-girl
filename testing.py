# testing.py
# Here's where I test the code as I refactor

import pandas as pd
from scripts.prepare_data import prepare_data
from scripts.features import create_features

print("Hello Chaos, this file is running!")

if __name__ == "__main__":
    # Load your actual dataset
    df = pd.read_csv("data/TMDB.csv")

    # Run only the genre features
    df = create_features(df)

    # Checkpoint outputs
    print("Dataset shape:", df.shape)
    print("Columns:", df.columns.tolist())
    print(df[["networks", "network_list"]].sample(5, random_state=42))
    print(df[["is_abc", "is_netflix", "is_bbc_one"]].head(10))
    print(df[["is_nbc", "is_cbs", "is_fox"]].head(10))
    print(df[["production_companies", "company_list"]].sample(5, random_state=42))
    print(df[["vote_average", "vote_count", "popularity",
          "is_highly_rated", "is_popular", "is_high_quality",
          "is_obscure", "is_moderately_popular", "is_hyped"]].sample(5, random_state=42))
    print(df[["number_of_episodes", "is_miniseries", "is_mid_length", "is_long_running"]].sample(5, random_state=42))
    print(df[[c for c in df.columns if c.startswith("is_")]].head())
