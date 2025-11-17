# testing.py
# Runs with: python testing.py
# Here's where I test the code as I refactor

import pandas as pd
# Load your dataset directly
df = pd.read_csv("data/TMDB.csv")

def get_features_by_title(show_name):
    row = df[df["name"] == show_name]  # assumes 'name' column in TMDB.csv
    if row.empty:
        return None
    return row.drop(columns=["name"]).iloc[0].to_dict()

print("\n=== Lookup Test ===")
print(get_features_by_title("Buffy the Vampire Slayer"))
print(get_features_by_title("Nonexistent Show"))