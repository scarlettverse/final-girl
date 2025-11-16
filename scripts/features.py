# features.py
# Contains funtions for engineered features

import ast
import pandas as pd

# --- GENRES ---
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

# --- NETWORKS ---
def split_networks(network_str):
    if pd.isnull(network_str):
        return []
    return [n.strip() for n in network_str.split(",")]

def add_network_list(df):
    df["network_list"] = df["networks"].apply(split_networks)
    return df

def add_network_flags(df):
    # Ensure network_list exists
    if "network_list" not in df.columns:
        df["network_list"] = df["networks"].apply(lambda s: [] if pd.isnull(s) else [n.strip() for n in str(s).split(",")])

    # Normalize network names to lowercase
    df["network_list"] = df["network_list"].apply(lambda nets: [n.strip().lower() for n in nets])

    # Top networks (lowercase for matching)
    top_networks = ["netflix", "youtube", "bbc one", "abc", "prime video"]
    for network in top_networks:
        col_name = f"is_{network.replace(' ', '_')}"
        df[col_name] = df["network_list"].apply(
            lambda nets: any(network in n for n in nets)  # substring match, all lowercase
        )

    # Legacy networks (lowercase columns and matching)
    legacy_networks = ["nbc", "cbs", "fox"]
    for network in legacy_networks:
        col_name = f"is_{network}"
        df[col_name] = df["networks"].apply(
            lambda x: (network in str(x).lower()) if pd.notnull(x) else False
        )
    return df

# --- PRODUCTION COMPANIES ---
import re

def split_companies(company_str):
    if pd.isnull(company_str):
        return []
    return [c.strip() for c in company_str.split(",")]

def clean_companies(company_str):
    if pd.isnull(company_str):
        return []
    # Split on commas not inside parentheses
    return [c.strip() for c in re.split(r',(?![^()]*\))', company_str)]

def add_company_list(df):
    # Use the more reliable cleaner
    df["company_list"] = df["production_companies"].apply(clean_companies)
    return df

def add_company_flags(df):
    # Top studios to analyze
    top_companies = [
        "TVB",
        "BBC",
        "Warner Bros. Television",
        "Universal Television",
        "Amazon Studios",
    ]
    for company in top_companies:
        col_name = (
            f"has_{company.lower().replace(' ', '_').replace('.', '').replace('&', 'and')}"
        )
        df[col_name] = df["company_list"].apply(lambda x: company in x)
    return df

# --- POPULARITY & RATINGS ---
def add_popularity_flags(df):
    # Vote-based features
    df["is_highly_rated"] = df["vote_average"] > 7
    df["is_popular"] = df["vote_count"] > 100
    df["is_high_quality"] = df["is_highly_rated"] & df["is_popular"]

    # Popularity tiers
    df["is_obscure"] = df["popularity"] <= 1.0
    df["is_moderately_popular"] = (df["popularity"] > 1.0) & (df["popularity"] <= 10.0)
    df["is_hyped"] = df["popularity"] > 10.0

    return df

# --- EPISODES / TYPE ---
def add_episode_flags(df):
    # Episode count tiers
    df["is_miniseries"] = df["number_of_episodes"] <= 6
    df["is_mid_length"] = (df["number_of_episodes"] > 6) & (df["number_of_episodes"] <= 20)
    df["is_long_running"] = df["number_of_episodes"] > 20
    return df

# --- SURVIVAL RULES ---
def add_survival_rules(df):
    # Scream Queen composite
    df["killer_karma"] = (
        df["is_highly_rated"].astype(int) +
        df["is_popular"].astype(int) +
        df["is_hyped"].astype(int) +
        df["has_drama"].astype(int) +
        df["is_abc"].astype(int) +
        df["has_tvb"].astype(int)
    )

    # Final Girl composite
    df["last_laughs"] = (
        df["is_miniseries"].astype(int) +
        df["is_obscure"].astype(int) +
        df["has_reality"].astype(int) +
        df["is_youtube"].astype(int) +
        df["has_documentary"].astype(int) +
        df["has_amazon_studios"].astype(int)
    )
    return df



# --- MASTER FEATURE BUILDER ---
def create_features(df):
    df = add_genre_list(df)
    df = add_genre_flags(df)
    df = add_network_list(df)
    df = add_network_flags(df)
    df = add_company_list(df)
    df = add_company_flags(df)
    df = add_popularity_flags(df)
    df = add_episode_flags(df)
    df = add_survival_rules(df)
    return df

print("features.py executed")
print("Functions:", dir())
