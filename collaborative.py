import numpy as np
import pandas as pd
# Large user–movie rating matrix -> smaller num of latent factors
from sklearn.decomposition import TruncatedSVD


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
