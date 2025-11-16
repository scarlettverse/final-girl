# scripts/prepare_data.py
# This module prepares the dataset for survival modeling.

import pandas as pd
import re

DATA_PATH = "data/TMDB.csv"

def get_dataset():
    # === Load raw data ===
    df = pd.read_csv(DATA_PATH)

    # === Drop sparse or misleading columns ===
    # These features are often missing or reflect post-cancellation artifacts (e.g. overview more common in slashed shows).
    drop_cols = ["overview", "backdrop_path", "homepage", "tagline", "created_by", "production_companies"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # === Define survival label ===
    # Option B: Use status for nuanced control.
    # Shows marked as "Returning Series" or "In Production" are considered survivors (Final Girls).
    df["is_final_girl"] = df["status"].isin(["Returning Series", "In Production"])

    # === Genre flags ===
    # Expanded set of genres, capturing both renewal-prone and slasher-coded categories.
    genres = ["Drama", "Comedy", "Documentary", "Animation", "Reality", "Crime", "Family", "Sci-Fi & Fantasy"]
    for genre in genres:
        col_name = f"has_{genre.lower().replace(' & ', '_and_').replace(' ', '_')}"
        df[col_name] = df["genres"].apply(lambda x: genre in x if isinstance(x, list) else False)

    # === Modern network flags ===
    # Streaming and broadcast platforms with distinct survival patterns.
    # Prestige broadcast networks with high cancellation rates.
    modern_networks = ["Netflix", "YouTube", "BBC One", "ABC", "Prime Video"]
    legacy_networks = ["NBC", "CBS", "FOX"]

    for network in modern_networks + legacy_networks:
        col_name = f"is_{network.lower().replace(' ', '_')}"
        df[col_name] = df["networks"].apply(
            lambda x: network in x if isinstance(x, list) else (network in str(x) if pd.notnull(x) else False)
    )

    # === Production company flags ===
    # Studios with distinct renewal/cancellation tendencies.
    # --- Production company parsing ---
    def clean_companies(company_str):
        if pd.isnull(company_str):
            return []
        # Split on commas not inside parentheses
        return [c.strip() for c in re.split(r',(?![^()]*\))', company_str)]

    # Apply parser to create a clean list column
    if "production_companies" in df.columns:
        df["company_list"] = df["production_companies"].apply(clean_companies)
    else:
        df["company_list"] = [[] for _ in range(len(df))]

    # --- Flags for specific studios ---
    companies = ["TVB", "BBC", "Warner Bros. Television", "Universal Television", "Amazon Studios"]

    for company in companies:
        col_name = f"has_{company.lower().replace(' ', '_').replace('.', '').replace('&', 'and')}"
        df[col_name] = df["company_list"].apply(lambda x: company in x if isinstance(x, list) else False)
    
    # === Audience signals (vote-based features) ===
    df["is_highly_rated"] = df["vote_average"] > 7
    df["is_popular"] = df["vote_count"] > 100
    df["is_high_quality"] = df["is_highly_rated"] & df["is_popular"]

    # === Popularity tiers ===
    df["is_obscure"] = df["popularity"] <= 1.0
    df["is_moderately_popular"] = (df["popularity"] > 1.0) & (df["popularity"] <= 10.0)
    df["is_hyped"] = df["popularity"] > 10.0

    # === Episode count tiers ===
    df["is_miniseries"] = df["number_of_episodes"] <= 6
    df["is_mid_length"] = (df["number_of_episodes"] > 6) & (df["number_of_episodes"] <= 20)
    df["is_long_running"] = df["number_of_episodes"] > 20

    # === Composite scores ===
    # Killer karma: prestige-coded, hype-driven, often slashed mid-arc.
    df["killer_karma"] = (
        df["is_highly_rated"].astype(int)
        + df["is_popular"].astype(int)
        + df["is_hyped"].astype(int)
        + df["has_drama"].astype(int)
        + df["is_abc"].astype(int)
        + df["has_tvb"].astype(int)
    )

    # Last Laughs: quiet survivors, niche formats, renewal-prone signals.
    df["last_laughs"] = (
        df["is_miniseries"].astype(int)
        + df["is_obscure"].astype(int)
        + df["has_reality"].astype(int)
        + df["is_youtube"].astype(int)
        + df["has_documentary"].astype(int)
        + df["has_amazon_studios"].astype(int)
    )

    # === Return engineered dataset ===
    # At this point, df contains:
    # - Target: is_final_girl
    # - Numeric features: vote_average, vote_count, popularity
    # - Binary flags: genres, networks, companies, audience signals, popularity tiers, episode tiers
    # - Composite scores: killer_karma, last_laughs
    return df
