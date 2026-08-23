import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
 
 
class ContentBasedRecommender:
    def __init__(self, movies_df: pd.DataFrame, popularity_pool: int = 30, m_quantile: float = 0.90):
        self.movies = movies_df.reset_index(drop=True)
        self.title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        )
        self._build_similarity_matrix()
 
        # ---- IMDB weighted-rating params (Q2: popularity filter) ----
        # C = mean vote_average across the whole movie set
        # m = minimum vote_count required to be considered (quantile-based
        #     threshold, same idea as the tutorial's "top X% by votes")
        self.popularity_pool = popularity_pool
        self.C = self.movies["vote_average"].mean()
        self.m = self.movies["vote_count"].quantile(m_quantile)
 
    def _build_similarity_matrix(self):
        vectorizer = CountVectorizer(stop_words="english", max_features=20000)
        soup_matrix = vectorizer.fit_transform(self.movies["soup"].fillna(""))
        self.similarity = cosine_similarity(soup_matrix, soup_matrix)
 
    def _weighted_rating(self, row):
        v = row["vote_count"]
        r = row["vote_average"]
        return (v / (v + self.m)) * r + (self.m / (v + self.m)) * self.C
 
    def recommend(self, title: str, top_n: int = 10, use_popularity_filter: bool = False) -> pd.DataFrame:
        title_key = title.lower()
        if title_key not in self.title_to_index:
            return pd.DataFrame()
        idx = self.title_to_index[title_key]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx]
 
        # Q2: pull a larger similarity pool, re-rank it by IMDB weighted
        # rating, then cut down to top_n — instead of just taking the
        # top_n most-similar movies outright.
        pool_size = self.popularity_pool if use_popularity_filter else top_n
        scores = scores[:pool_size]
 
        movie_indices = [i for i, _ in scores]
        result = self.movies.iloc[movie_indices].copy()
        result["similarity_score"] = [s for _, s in scores]
 
        if use_popularity_filter:
            result["weighted_rating"] = result.apply(self._weighted_rating, axis=1)
            result = result.sort_values("weighted_rating", ascending=False).head(top_n)
 
        return result
