"""
app.py
------
Streamlit UI for the Movie Recommendation System.
Run with:  streamlit run app.py
"""

import ast
import pandas as pd
import streamlit as st

from content_based import ContentBasedRecommender
from collaborative import SVDRecommender

POSTER_BASE = "https://image.tmdb.org/t/p/w300"


@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies_clean.csv")
    # list-like columns come back as strings from CSV — parse them
    for col in ["genre_names", "cast_top5", "keyword_names"]:
        movies[col] = movies[col].apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) else []
        )
    ratings = pd.read_csv("data/ratings_clean.csv")
    return movies, ratings


@st.cache_resource
def load_content_model(movies):
    return ContentBasedRecommender(movies)


@st.cache_resource
def load_cf_model(ratings):
    return SVDRecommender(ratings)


def poster_url(path):
    if isinstance(path, str) and path.strip():
        return POSTER_BASE + path
    return "https://via.placeholder.com/300x450?text=No+Poster"


def release_year(value):
    if isinstance(value, str) and len(value) >= 4:
        return value[:4]
    return "—"


def show_movie_grid(df, score_col=None):
    cols = st.columns(5)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % 5]:
            st.image(poster_url(row.get("poster_path")), use_container_width=True)
            st.caption(f"**{row['title']}** ({release_year(row.get('release_date'))})")
            if score_col and score_col in row and pd.notna(row[score_col]):
                st.caption(f"{score_col.replace('_', ' ')}: {row[score_col]:.2f}")


st.set_page_config(page_title="Movie Recommender", layout="wide")
st.title("🎬 Movie Recommendation System")
st.caption("Content-Based Filtering vs Collaborative Filtering — AI Assignment Prototype")

movies, ratings = load_data()

tab1, tab2 = st.tabs(["🔎 Content-Based (by movie)", "👤 Collaborative (by user)"])

with tab1:
    st.subheader("Find movies similar to one you like")
    content_model = load_content_model(movies)
    movie_choice = st.selectbox("Pick a movie", sorted(movies["title"].dropna().unique()))
    top_n = st.slider("Number of recommendations", 5, 20, 10, key="cb_slider")

    if st.button("Recommend", key="cb_button"):
        recs = content_model.recommend(movie_choice, top_n=top_n)
        if recs.empty:
            st.warning("No recommendations found for that title.")
        else:
            show_movie_grid(recs, score_col="similarity_score")

with tab2:
    st.subheader("Recommend movies for a specific user based on rating history")
    cf_model = load_cf_model(ratings)
    valid_users = sorted(ratings["userId"].unique())
    user_choice = st.selectbox("Pick a user ID", valid_users)
    top_n_cf = st.slider("Number of recommendations", 5, 20, 10, key="cf_slider")

    if st.button("Recommend", key="cf_button"):
        recs = cf_model.recommend_for_user(user_choice, movies, top_n=top_n_cf)
        if recs.empty:
            st.warning("No recommendations found for that user.")
        else:
            show_movie_grid(recs, score_col="predicted_rating")

        with st.expander("This user's rating history"):
            history = ratings[ratings["userId"] == user_choice].merge(
                movies[["tmdbId", "title"]], on="tmdbId"
            )
            st.dataframe(history[["title", "rating"]].sort_values("rating", ascending=False))
