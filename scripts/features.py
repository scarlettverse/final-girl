# features.py
# Contains funtions for engineered features

import ast
import pandas as pd

print(">>> importing features.py")

def extract_genres(genre_str):
    
    #Safely parse the 'genres' field (stored as a stringified list of dicts).
    #Returns a list of genre names.
   
    try:
        genres = ast.literal_eval(genre_str)
        return [g["name"] for g in genres if "name" in g]
    except Exception:
        return []

def split_genres(genre_str):
    
    #Handle comma-separated genre strings (alternative format).
    #Returns a list of genre names.
    
    if pd.isnull(genre_str):
        return []
    return [g.strip() for g in genre_str.split(",")]

def add_genre_list(df):
    
    #Adds a 'genre_list' column to the dataframe.
    #Chooses the appropriate parsing function depending on format.
    
    # If your dataset uses dict-like strings:
    #df["genre_list"] = df["genres"].apply(extract_genres)

    # If your dataset uses comma-separated strings:
    df["genre_list"] = df["genres"].apply(split_genres)

    return df

def add_genre_flags(df):
    
    #Adds binary flags for top and additional genres.
    
    top_genres = ["Drama", "Comedy", "Documentary", "Animation", "Reality"]
    more_genres = ["Crime", "Family", "Sci-Fi & Fantasy"]
    all_genres_to_check = top_genres + more_genres

    for genre in all_genres_to_check:
        col_name = f"has_{genre.lower().replace(' & ', '_and_').replace(' ', '_')}"
        df[col_name] = df["genre_list"].apply(lambda x: genre in x)

    return df

def create_features(df):
    df = add_genre_list(df)
    df = add_genre_flags(df)
    return df

print("features.py executed")
print("Functions:", dir())
