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
    print(df[["genres", "genre_list"]].head())
    print(df[[c for c in df.columns if c.startswith("has_")]].head())
