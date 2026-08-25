import pandas as pd
# CountVectorizer = Text -> Num
from sklearn.feature_extraction.text import CountVectorizer
# Calc how similar movies r based on numerical vectors
from sklearn.metrics.pairwise import cosine_similarity
 
 
class ContentBasedRecommender:
    def __init__(self, movies_df: pd.DataFrame, popularity_pool: int = 30, m_quantile: float = 0.90):
        # Reset the DtFrame index
        self.movies = movies_df.reset_index(drop=True)
        # Create title -> index loopup
        self.title_to_index = pd.Series(
            self.movies.index, index=self.movies["title"].str.lower()
        )
        self._build_similarity_matrix()

        # IMDB weighted-rating
        self.popularity_pool = popularity_pool
        # C = Average movie rating
        self.C = self.movies["vote_average"].mean()
        # m = Min num of votes
        self.m = self.movies["vote_count"].quantile(m_quantile)
    
    # Function: Build Similarity Matrix
    def _build_similarity_matrix(self):
        vectorizer = CountVectorizer(stop_words="english", max_features=20000)
        # Fit_transform = Learn word n convert to num
        soup_matrix = vectorizer.fit_transform(self.movies["soup"].fillna(""))
        self.similarity = cosine_similarity(soup_matrix, soup_matrix)
 
    # Function: Cal Weighted Rating
    def _weighted_rating(self, row):
        # Get movie num of votes
        v = row["vote_count"]
        # Get movie average rating
        r = row["vote_average"]
        return (v / (v + self.m)) * r + (self.m / (v + self.m)) * self.C
 
    def recommend(self, title: str, top_n: int = 10, use_popularity_filter: bool = False) -> pd.DataFrame:
        title_key = title.lower()
        # If movie not exist return empty
        if title_key not in self.title_to_index:
            return pd.DataFrame()
        # Get movie index
        idx = self.title_to_index[title_key]
        # Handle duplicate titles
        if isinstance(idx, pd.Series):
            # Take first matching index
            idx = idx.iloc[0]
        # x = (Movie index, similarity score)
        scores = list(enumerate(self.similarity[idx]))
        # Sort higher scores first
        scores = sorted(scores, key=lambda x: x[1], reverse=True)
        # Remove the user inputed movie itself
        scores = [s for s in scores if s[0] != idx]
 
        # If weighted rating ON -> use 30, if OFF use 10
        pool_size = self.popularity_pool if use_popularity_filter else top_n
        scores = scores[:pool_size]
 
        # Get movie index
        movie_indices = [i for i, _ in scores]
        # Retrieve actual movie rows
        result = self.movies.iloc[movie_indices].copy()
        # Add similarity scores
        result["similarity_score"] = [s for _, s in scores]
    
        # If weighted rating ON
        if use_popularity_filter:
            # Cal weighted rating for each of those 30 movies
            result["weighted_rating"] = result.apply(self._weighted_rating, axis=1)
            # Sort by highest weighted rating first, then take top_n
            result = result.sort_values("weighted_rating", ascending=False).head(top_n)
 
        return result
