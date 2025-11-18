# scripts/config.py
# Centralized constants for reproducibility and easy toggling between Final Girl and Scream Queens
# Activate virtual env: venv\Scripts\Activate.ps1

RANDOM_SEED = 42
DATA_PATH = "data/TMDB.csv" # Training Data
SHOWS = "data/tv.csv" # Serving Data
MODEL_PATH = "models/gradient_boosting.pkl"
TEST_SIZE = 0.2
