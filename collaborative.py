"""
collaborative.py
-----------------
Collaborative filtering recommender using matrix factorization (SVD via
sklearn's TruncatedSVD) on the user-item rating matrix.

This avoids extra dependencies (no `surprise` install headaches). For your
report you can compare this against a simple item-based kNN/cosine approach
as a second method — see ItemBasedRecommender below.
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


class SVDRecommender:
    def __init__(self, ratings_df: pd.DataFrame, n_components: int = 20):
        self.pivot = ratings_df.pivot_table(
            index="userId", columns="tmdbId", values="rating"
        ).fillna(0)
        self.user_ids = self.pivot.index
        self.movie_ids = self.pivot.columns

        svd = TruncatedSVD(n_components=n_components, random_state=42)
        matrix_reduced = svd.fit_transform(self.pivot.values)
        self.predicted = np.dot(matrix_reduced, svd.components_)

    def recommend_for_user(self, user_id: int, movies_df: pd.DataFrame, top_n: int = 10):
        if user_id not in self.user_ids:
            return pd.DataFrame()

        user_row_idx = self.user_ids.get_loc(user_id)
        predicted_ratings = pd.Series(self.predicted[user_row_idx], index=self.movie_ids)

        already_rated = self.pivot.loc[user_id]
        already_rated_ids = already_rated[already_rated > 0].index
        predicted_ratings = predicted_ratings.drop(index=already_rated_ids, errors="ignore")

        top_ids = predicted_ratings.sort_values(ascending=False).head(top_n).index
        result = movies_df[movies_df["tmdbId"].isin(top_ids)].copy()
        result["predicted_rating"] = result["tmdbId"].map(predicted_ratings)
        return result.sort_values("predicted_rating", ascending=False)


class ItemBasedRecommender:
    """Item-item collaborative filtering: 'users who liked this also liked...'"""

    def __init__(self, ratings_df: pd.DataFrame):
        self.pivot = ratings_df.pivot_table(
            index="tmdbId", columns="userId", values="rating"
        ).fillna(0)
        self.similarity = cosine_similarity(self.pivot.values)
        self.movie_ids = self.pivot.index

    def recommend(self, tmdb_id: int, movies_df: pd.DataFrame, top_n: int = 10):
        if tmdb_id not in self.movie_ids:
            return pd.DataFrame()

        idx = self.movie_ids.get_loc(tmdb_id)
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if self.movie_ids[s[0]] != tmdb_id][:top_n]

        rec_ids = [self.movie_ids[i] for i, _ in scores]
        result = movies_df[movies_df["tmdbId"].isin(rec_ids)].copy()
        return result
