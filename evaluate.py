import ast
import numpy as np
import pandas as pd
# Split arrays or matrices into random train n test subsets
from sklearn.model_selection import train_test_split

from collaborative import SVDRecommender
from content_based import ContentBasedRecommender

# Produce consistent n reproducible results
RANDOM_STATE = 42
DATA_DIR = "data"

# ---------------------------------------------------------------------------
# 1. Collaborative filtering
# ---------------------------------------------------------------------------
def evaluate_collaborative(ratings: pd.DataFrame, n_components: int = 20, test_size: float = 0.2):
    # Split 80% for train 20% for test
    train, test = train_test_split(ratings, test_size=test_size, random_state=RANDOM_STATE)

    model = SVDRecommender(train, n_components=n_components)

    # Training dt user id n movie id -> set
    known_users = set(model.user_ids)
    known_movies = set(model.movie_ids)
    # Find the both user's ID n movie inside 80% of training dt
    test_known = test[test["userId"].isin(known_users) & test["tmdbId"].isin(known_movies)]
    # Find num of 20% test rating can actually evaluated
    coverage = len(test_known) / len(test)

    # preds store predicted ratings
    # actuals store actual ratings given by users
    preds, actuals = [], []
    for row in test_known.itertuples(index=False):
        # Find index / position of user n movie in SVD model
        u_idx = model.user_ids.get_loc(row.userId)
        m_idx = model.movie_ids.get_loc(row.tmdbId)
        # Get model's predicted rating n actual rating put in list
        preds.append(model.predicted[u_idx, m_idx])
        actuals.append(row.rating)

    # Ratings only ever range 0.5 - 5
    preds = np.clip(np.array(preds), 0.5, 5.0)
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    mae = np.mean(np.abs(preds - actuals))

    return {
        "Num of train": len(train),
        "Num of test": len(test),
        # Fraction of test rows the model could even score
        "Coverage": coverage,
        "Root Mean Squared Error (RMSE)": rmse,
        "Mean Absolute Error (MAE)": mae,
    }


# ---------------------------------------------------------------------------
# 2. Content-based
# ---------------------------------------------------------------------------
def _parse_genres(cell):
    if isinstance(cell, str):
        try:
            return set(ast.literal_eval(cell))
        except (ValueError, SyntaxError):
            return set()
    return set()

# k: top 10 recommended movie
# n_seeds: randomly select 200 movies
def evaluate_content_based(movies: pd.DataFrame, k: int = 10, n_seeds: int = 200):
    model = ContentBasedRecommender(movies)
    # Convert each movie's genre information into a set for comparison
    genres = movies.set_index("tmdbId")["genre_names"].apply(_parse_genres)

    # Set the random generator to ensure reproducible sampling
    rng = np.random.default_rng(RANDOM_STATE)
    # Randomly select movies to use as test seed movies
    seed_titles = movies["title"].dropna().sample(n=n_seeds, random_state=RANDOM_STATE).tolist()

    precisions = []
    for title in seed_titles:
        # Generate the top-k recommendations for the selected movie
        recs = model.recommend(title, top_n=k)
        # Skip the movie if no recommendations are returned
        if recs.empty:
            continue
        # Extract the genres of the selected seed movie
        seed_genres = _parse_genres(movies.loc[movies["title"] == title, "genre_names"].iloc[0])
        # Skip the movie if it has no valid genre information
        if not seed_genres:
            continue
        # Count recommended movies that share at least one genre with the seed movie
        hits = sum(
            1 for tmdb_id in recs["tmdbId"]
            if genres.get(tmdb_id, set()) & seed_genres
        )
        precisions.append(hits / len(recs))

    return {
        "k": k,
        "Num of seeds used": len(precisions),
        "Precision": float(np.mean(precisions)),
    }


if __name__ == "__main__":
    movies = pd.read_csv(f"{DATA_DIR}/movies_clean.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings_clean.csv")

    print("=== Collaborative filtering - RMSE / MAE ===")
    cf_results = evaluate_collaborative(ratings)
    for k, v in cf_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n======== Content-based - Precision ========")
    cb_results = evaluate_content_based(movies)
    for k, v in cb_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")