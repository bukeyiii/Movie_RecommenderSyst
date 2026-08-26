import numpy as np
import pandas as pd
# Large user–movie rating matrix -> smaller num of latent factors
from sklearn.decomposition import TruncatedSVD
# Item-item similarity from co-rating patterns
from sklearn.metrics.pairwise import cosine_similarity


class SVDRecommender:
    # n_components: Reduce the rating matrix to 20 latent factors
    def __init__(self, ratings_df: pd.DataFrame, n_components: int = 20):
        # Make original dt transform to pivot table
        self.pivot = ratings_df.pivot_table(
            index="userId", columns="tmdbId", values="rating"
        # Put 0 when user does not rated a movie
        ).fillna(0)
        self.user_ids = self.pivot.index
        self.movie_ids = self.pivot.columns
        # Determine num of SVD components, prevent invalid SVD configuration
        n_comp = min(n_components, min(self.pivot.shape) - 1)
        # Create TruncatedSVD
        # svd: latent factors x movies
        # random_state: Ensure result reproducible
        svd = TruncatedSVD(n_components=n_comp, random_state=42)
        # matrix_reduced: user x latent factors
        matrix_reduced = svd.fit_transform(self.pivot.values)
        # Get predicted rating by multiple matrix_reduced n svd.components_
        self.predicted = np.dot(matrix_reduced, svd.components_)

    def recommend_for_user(self, user_id: int, movies_df: pd.DataFrame, top_n: int = 10):
        if user_id not in self.user_ids:
            return pd.DataFrame()
        # Find user's row
        user_row_idx = self.user_ids.get_loc(user_id)
        # Get user's predicted ratings
        predicted_ratings = pd.Series(self.predicted[user_row_idx], index=self.movie_ids)
        # Find movies user alr rated
        already_rated = self.pivot.loc[user_id]
        # Get only movies that actually rated
        already_rated_ids = already_rated[already_rated > 0].index
        # Remove alr-rated movies
        predicted_ratings = predicted_ratings.drop(index=already_rated_ids, errors="ignore")
        # Select the top N predictions, sort highest -> lowest
        top_ids = predicted_ratings.sort_values(ascending=False).head(top_n).index
        # Get actual movie information
        result = movies_df[movies_df["tmdbId"].isin(top_ids)].copy()
        result["predicted_rating"] = result["tmdbId"].map(predicted_ratings)
        return result.sort_values("predicted_rating", ascending=False)


class ItemBasedCFRecommender:
    # min_common_raters: ignore item pairs propped up by just 1-2 shared raters
    def __init__(self, ratings_df: pd.DataFrame, min_common_raters: int = 3):
        # Same pivot shape as SVDRecommender: rows = users, columns = movies
        self.pivot = ratings_df.pivot_table(
            index="userId", columns="tmdbId", values="rating"
        ).fillna(0)
        self.movie_ids = self.pivot.columns

        # Item vectors = columns of the pivot -> transpose to movies x users
        item_matrix = self.pivot.values.T

        # Cosine similarity between every pair of movie rating-vectors
        self.similarity = cosine_similarity(item_matrix)

        # Cosine similarity is inflated for pairs with very few shared raters
        rated_mask = (self.pivot.values.T > 0).astype(int)  # movies x users
        co_rated_counts = rated_mask @ rated_mask.T  # movies x movies
        self.similarity[co_rated_counts < min_common_raters] = 0.0

        self._movie_pos = {m: i for i, m in enumerate(self.movie_ids)}

    def recommend_from_liked(self, liked_ids, movies_df: pd.DataFrame, top_n: int = 10):
        liked_idxs = [self._movie_pos[m] for m in liked_ids if m in self._movie_pos]
        if not liked_idxs:
            return pd.DataFrame()

        # Average similarity of every movie to the set of liked movies
        agg_scores = self.similarity[liked_idxs].mean(axis=0)
        scores = pd.Series(agg_scores, index=self.movie_ids)
        scores = scores.drop(index=liked_ids, errors="ignore")
        scores = scores[scores > 0]

        if scores.empty:
            return pd.DataFrame()

        top_ids = scores.sort_values(ascending=False).head(top_n).index
        result = movies_df[movies_df["tmdbId"].isin(top_ids)].copy()
        result["similarity_score"] = result["tmdbId"].map(scores)
        return result.sort_values("similarity_score", ascending=False)