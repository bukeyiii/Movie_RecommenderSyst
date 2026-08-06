import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedRecommender:
    def __init__(self, movies_df: pd.DataFrame):
        self.movies = movies_df.reset_index(drop=True)
        self.title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        )
        self._build_similarity_matrix()

    def _build_similarity_matrix(self):
        vectorizer = CountVectorizer(stop_words="english", max_features=20000)
        soup_matrix = vectorizer.fit_transform(self.movies["soup"].fillna(""))
        self.similarity = cosine_similarity(soup_matrix, soup_matrix)

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        title_key = title.lower()
        if title_key not in self.title_to_index:
            return pd.DataFrame()
        idx = self.title_to_index[title_key]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]
        scores = list(enumerate(self.similarity[idx]))
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        scores = [s for s in scores if s[0] != idx][:top_n]
        movie_indices = [i for i, _ in scores]
        result = self.movies.iloc[movie_indices].copy()
        result["similarity_score"] = [s for _, s in scores]
        return result
