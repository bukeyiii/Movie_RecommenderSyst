"""
evaluate.py
-----------
Offline evaluation for the two recommenders in this project.

  1. Collaborative filtering (SVDRecommender)
     -> RMSE / MAE on held-out ratings (a regression-style metric,
        since this recommender's job is to predict a numeric rating).

  2. Content-based recommender (ContentBasedRecommender)
     -> Precision@K, using "shares at least one genre with the seed
        movie" as a stand-in for "relevant", since we have no explicit
        ground-truth of "movies the user would like" to test against.

Run with:  python evaluate.py
(place this file next to app.py / collaborative.py / content_based.py —
it expects a data/ subfolder containing movies_clean.csv and
ratings_clean.csv, same as app.py does)
"""

import ast
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from collaborative import SVDRecommender
from content_based import ContentBasedRecommender

RANDOM_STATE = 42
DATA_DIR = "data"


# ---------------------------------------------------------------------------
# 1. Collaborative filtering: RMSE / MAE
# ---------------------------------------------------------------------------
def evaluate_collaborative(ratings: pd.DataFrame, n_components: int = 20, test_size: float = 0.2):
    train, test = train_test_split(ratings, test_size=test_size, random_state=RANDOM_STATE)

    model = SVDRecommender(train, n_components=n_components)

    # Only score rows whose user AND movie were seen in training —
    # the model has no way to predict for a brand-new user/movie
    # (the classic "cold start" problem), so those rows are reported
    # separately rather than silently dropped.
    known_users = set(model.user_ids)
    known_movies = set(model.movie_ids)
    test_known = test[test["userId"].isin(known_users) & test["tmdbId"].isin(known_movies)]
    coverage = len(test_known) / len(test)

    preds, actuals = [], []
    for row in test_known.itertuples(index=False):
        u_idx = model.user_ids.get_loc(row.userId)
        m_idx = model.movie_ids.get_loc(row.tmdbId)
        preds.append(model.predicted[u_idx, m_idx])
        actuals.append(row.rating)

    preds = np.clip(np.array(preds), 0.5, 5.0)  # ratings only ever range 0.5-5
    actuals = np.array(actuals)

    rmse = np.sqrt(np.mean((preds - actuals) ** 2))
    mae = np.mean(np.abs(preds - actuals))

    return {
        "n_train": len(train),
        "n_test": len(test),
        "coverage": coverage,   # fraction of test rows the model could even score
        "rmse": rmse,
        "mae": mae,
    }


# ---------------------------------------------------------------------------
# 2. Content-based: Precision@K (genre-overlap proxy for "relevant")
# ---------------------------------------------------------------------------
def _parse_genres(cell):
    if isinstance(cell, str):
        try:
            return set(ast.literal_eval(cell))
        except (ValueError, SyntaxError):
            return set()
    return set()


def evaluate_content_based(movies: pd.DataFrame, k: int = 10, n_seeds: int = 200):
    model = ContentBasedRecommender(movies)
    genres = movies.set_index("tmdbId")["genre_names"].apply(_parse_genres)

    rng = np.random.default_rng(RANDOM_STATE)
    seed_titles = movies["title"].dropna().sample(n=n_seeds, random_state=RANDOM_STATE).tolist()

    precisions = []
    for title in seed_titles:
        recs = model.recommend(title, top_n=k)
        if recs.empty:
            continue
        seed_genres = _parse_genres(movies.loc[movies["title"] == title, "genre_names"].iloc[0])
        if not seed_genres:
            continue
        hits = sum(
            1 for tmdb_id in recs["tmdbId"]
            if genres.get(tmdb_id, set()) & seed_genres
        )
        precisions.append(hits / len(recs))

    return {
        "k": k,
        "n_seeds_used": len(precisions),
        "precision_at_k": float(np.mean(precisions)),
    }


if __name__ == "__main__":
    movies = pd.read_csv(f"{DATA_DIR}/movies_clean.csv")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings_clean.csv")

    print("=== Collaborative filtering (SVD) — RMSE / MAE ===")
    cf_results = evaluate_collaborative(ratings)
    for k, v in cf_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    print("\n=== Content-based — Precision@K (genre-overlap proxy) ===")
    cb_results = evaluate_content_based(movies)
    for k, v in cb_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")