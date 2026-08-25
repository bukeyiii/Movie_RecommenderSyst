# ast = Abstract Syntax Trees: Convert string -> list
import ast
import html as html_escape
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from content_based import ContentBasedRecommender
from collaborative import SVDRecommender

# For display list
POSTER_BASE = "https://image.tmdb.org/t/p/w300"
# For display movie details
POSTER_BASE_LARGE = "https://image.tmdb.org/t/p/w500"

# Storage box for values
_STICKY = "_sticky_values"

# Get previously search values
def sticky_get(name, default=None):
    return st.session_state.get(_STICKY, {}).get(name, default)

# Save previously search values
def sticky_set(name, value):
    st.session_state.setdefault(_STICKY, {})
    st.session_state[_STICKY][name] = value

# Able store one or more widget key
def sticky_save(*names):
    # Create callback get then saves
    def _cb():
        for n in names:
            sticky_set(n, st.session_state[n])
    return _cb

# Modify Streamlit webpage's HTML elements using JS
# Make selected Streamlit dropdown inputs read-only
def lock_dropdown_typing(*container_keys):
    selectors = ", ".join(f'div[class*="st-key-{k}"] input' for k in container_keys)
    components.html(
        f"""
        <script>
        function lockInputs() {{
            const doc = window.parent.document;
            doc.querySelectorAll('{selectors}').forEach(function(el) {{
                el.readOnly = true;
                el.style.caretColor = 'transparent';
                el.style.cursor = 'pointer';
            }});
        }}
        lockInputs();
        new MutationObserver(lockInputs).observe(window.parent.document.body, {{childList: true, subtree: true}});
        </script>
        """,
        height=0,
    )

SEARCH_MODES = ["Movie Title", "Keyword", "Genre", "Overview", "Rating"]

SORT_OPTIONS = ["Best Match", "A-Z", "Popularity", "User Rating", "Release Year"]
SORT_FIELD = {"A-Z": "title", "Popularity": "popularity", "User Rating": "vote_average", "Release Year": "release_date"}

SORT_DEFAULT_ASCENDING = {"A-Z": True, "Popularity": False, "User Rating": False, "Release Year": False}

# Only able select when search by Movie Title
CB_SORT_OPTIONS = SORT_OPTIONS + ["Weighted Rating"]
CB_SORT_FIELD = {**SORT_FIELD, "Weighted Rating": "weighted_rating"}
CB_SORT_DEFAULT_ASCENDING = {**SORT_DEFAULT_ASCENDING, "Weighted Rating": False}

CARD_CSS = """
<style>
.poster-wrap {
    position: relative;
    width: 100%;
    padding-top: 150%; /* fixed 2:3 poster aspect ratio for every card */
    border-radius: 10px;
    overflow: hidden;
    background: #1e1e1e;
}
.poster-wrap img {
    position: absolute;
    top: 0; left: 0;
    width: 100%; height: 100%;
    object-fit: cover;
}
.movie-card {
    margin-bottom: 8px;
}
.movie-card-title {
    font-weight: 600;
    font-size: 0.92rem;
    margin-top: 8px;
    height: 2.5em;
    line-height: 1.25em;
    overflow: hidden;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
}
.movie-card-sub {
    font-size: 0.8rem;
    color: #9a9a9a;
    height: 1.2em;
    overflow: hidden;
}
.genre-badge {
    display: inline-block;
    background: #333;
    color: #fff;
    padding: 3px 12px;
    border-radius: 12px;
    font-size: 0.78rem;
    margin: 0 6px 6px 0;
}
.stars {
    font-size: 1.1rem;
    color: #f5c518;
    letter-spacing: 1px;
}

div[class*="st-key-card_"] div[data-testid="stButton"] button {
    width: 100%;
    border: 1px solid rgba(250, 250, 250, 0.2);
    background: transparent;
    color: #ccc;
    font-size: 0.8rem;
    padding: 4px 0;
    margin-top: 2px;
}
div[class*="st-key-card_"] div[data-testid="stButton"] button:hover {
    border-color: #ff4b4b;
    color: #fff;
}
.sort-bar-label {
    font-size: 0.85rem;
    color: #9a9a9a;
    padding-top: 8px;
    text-align: right;
}
div[data-testid="stButton"] button[kind="secondary"] {
    padding: 0.25rem 0.6rem;
}
</style>
"""

# Keep the loaded dt cached
@st.cache_data
def load_data():
    movies = pd.read_csv("data/movies_clean.csv")
    for col in ["genre_names", "keyword_names"]:
        if col in movies.columns:
            # Convert string back to list
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

@st.cache_data
def get_unique_genres(movies):
    # set() to avoid repeated genre
    genres = set()
    # add all genres into set
    for g in movies["genre_names"]:
        genres.update(g)
    return sorted(genres)

def _title_sort_key(t):
    t = str(t)
    # Title come with alpha first
    return (0 if t[:1].isalpha() else 1, t.lower())

@st.cache_data
def get_sorted_titles(movies):
    return sorted(movies["title"].unique(), key=_title_sort_key)

# Create actual poster URL
def poster_url(path, large=False):
    if isinstance(path, str) and path.strip():
        return (POSTER_BASE_LARGE if large else POSTER_BASE) + path
    return "https://via.placeholder.com/500x750?text=No+Poster"

# Extracts only year from release date
def release_year(value):
    if isinstance(value, str) and len(value) >= 4:
        return value[:4]
    return "—"

# Converts rating into 5 stars 
def render_stars(vote_average):
    if pd.isna(vote_average):
        return '<span class="stars">No rating yet</span>'
    stars = max(0, min(5, round(vote_average / 2)))
    filled = "★" * stars
    empty = "☆" * (5 - stars)
    return f'<span class="stars">{filled}{empty}</span> {vote_average:.1f}/10'

# ---------------------------------------------------------------------------
# Card grid
# ---------------------------------------------------------------------------

def movie_card_html(row, score_label=None, score_value=None):
    title = html_escape.escape(str(row["title"]))
    poster = poster_url(row.get("poster_path"))
    # Decide what text goes under the title
    if score_label and score_value is not None and pd.notna(score_value):
        sub = html_escape.escape(f"{score_label}: {score_value:.2f}")
    else:
        # If no text display undet title, display year
        sub = html_escape.escape(release_year(row.get("release_date")))
    return f"""
    <div class="movie-card">
        <div class="poster-wrap"><img src="{poster}" /></div>
        <div class="movie-card-title">{title}</div>
        <div class="movie-card-sub">{sub}</div>
    </div>
    """

def show_movie_grid(df, score_col=None, cols_n=5):
    # 5 per row
    cols = st.columns(cols_n)
    for i, (_, row) in enumerate(df.iterrows()):
        with cols[i % cols_n]:
            tmdb_id = row["tmdbId"]
            score_val = row[score_col] if score_col and score_col in row else None
            with st.container(key=f"card_{tmdb_id}_{i}"):
                # Display the movie cards
                st.markdown(
                    movie_card_html(row, score_label=score_col.replace("_", " ") if score_col else None,
                                      score_value=score_val),
                    unsafe_allow_html=True,
                )
                if st.button("View details", key=f"open_{tmdb_id}_{i}", use_container_width=True):
                    st.query_params["movie"] = str(tmdb_id)
                    st.rerun()


# ---------------------------------------------------------------------------
# Sort bar
# ---------------------------------------------------------------------------

def _on_sort_field_change(sort_key, asc_key, default_ascending_map=None):
    default_ascending_map = default_ascending_map or SORT_DEFAULT_ASCENDING
    field = st.session_state[sort_key]
    ascending = default_ascending_map.get(field, False)
    sticky_set(sort_key, field)
    sticky_set(asc_key, ascending)


def render_sort_controls(drop_col, dir_col, sort_key="sort_by", asc_key="sort_asc", label_col=None,
                          options=None, default_ascending_map=None):
    options = options or SORT_OPTIONS
    default_ascending_map = default_ascending_map or SORT_DEFAULT_ASCENDING
    sort_by_default = sticky_get(sort_key, "Best Match")
    ascending = sticky_get(asc_key, False)

    if label_col is not None:
        with label_col:
            st.markdown('<div class="sort-bar-label">Sort by</div>', unsafe_allow_html=True)
    with drop_col:
        with st.container(key="sb_sort_by"):
            st.selectbox(
                "Sort by", options,
                index=options.index(sort_by_default) if sort_by_default in options else 0,
                key=sort_key, label_visibility="collapsed",
                on_change=_on_sort_field_change, args=(sort_key, asc_key, default_ascending_map),
            )
    with dir_col:
        is_best_match = sticky_get(sort_key, "Best Match") == "Best Match"
        if st.button(
            "▲" if ascending else "▼", key=f"{asc_key}_toggle",
            help="Toggle ascending / descending", disabled=is_best_match,
        ):
            sticky_set(asc_key, not ascending)
            st.rerun()

    return sticky_get(sort_key, "Best Match"), sticky_get(asc_key, False)


def render_sort_bar(sort_key="sort_by", asc_key="sort_asc"):
    spacer, label_col, drop_col, dir_col = st.columns([3, 0.8, 1.8, 0.5])
    return render_sort_controls(drop_col, dir_col, sort_key=sort_key, asc_key=asc_key, label_col=label_col)


def apply_sort(df: pd.DataFrame, sort_by: str, ascending: bool, field_map=None) -> pd.DataFrame:
    field_map = field_map or SORT_FIELD
    if df.empty or sort_by == "Best Match":
        return df
    col = field_map.get(sort_by)
    if not col or col not in df.columns:
        return df
    if col == "title":
        order = sorted(range(len(df)), key=lambda i: _title_sort_key(df.iloc[i]["title"]), reverse=not ascending)
        return df.iloc[order]
    return df.sort_values(col, ascending=ascending)


# ---------------------------------------------------------------------------
# Search-mode
# ---------------------------------------------------------------------------

def _text_match_score(text, q):
    text = text.lower()
    if text == q:
        return 4
    if text.startswith(q):
        return 3
    if q in text.split():
        return 2
    if q in text:
        return 1
    return 0


def filter_by_keyword(movies, query):
    q = query.lower().strip()
    if not q:
        return pd.DataFrame()

    def match_score(row):
        title_score = _text_match_score(str(row["title"]), q) * 2
        kw_score = max((_text_match_score(k, q) for k in row["keyword_names"]), default=0)
        return max(title_score, kw_score)

    scored = movies.copy()
    scored["_kw_match"] = scored.apply(match_score, axis=1)
    scored = scored[scored["_kw_match"] > 0]
    scored = scored.sort_values(["_kw_match", "popularity"], ascending=[False, False])
    return scored.drop(columns="_kw_match")


def filter_by_genre(movies, selected_genres):
    if not selected_genres:
        return pd.DataFrame()
    selected_set = set(selected_genres)

    def match_count(gs):
        return len(selected_set.intersection(gs))

    scored = movies.copy()
    scored["_genre_match_count"] = scored["genre_names"].apply(match_count)
    scored = scored[scored["_genre_match_count"] > 0]
    scored = scored.sort_values(["_genre_match_count", "vote_average"], ascending=[False, False])
    return scored.drop(columns="_genre_match_count")


def filter_by_overview(movies, query):
    q = query.lower().strip()
    if not q:
        return pd.DataFrame()
    return movies[movies["overview"].str.lower().str.contains(q, na=False)].sort_values(
        "popularity", ascending=False
    )


def filter_by_rating(movies, min_rating, max_rating):
    band = movies[(movies["vote_average"] >= min_rating) & (movies["vote_average"] <= max_rating)]
    return band.sort_values("popularity", ascending=False)


# ---------------------------------------------------------------------------
# Collaborative filtering — Picking a userId
# ---------------------------------------------------------------------------

def recommend_from_liked_movies(ratings_df, movies_df, selected_ids, min_rating=4.0, top_n=10):
    if not selected_ids:
        return pd.DataFrame(), 0

    # users who rated at least one of the picks highly
    liked_by_others = ratings_df[
        ratings_df["tmdbId"].isin(selected_ids) & (ratings_df["rating"] >= min_rating)
    ]
    similar_users = liked_by_others["userId"].unique()
    if len(similar_users) == 0:
        return pd.DataFrame(), 0

    # what else that crowd rated, excluding the movies already picked
    others_ratings = ratings_df[
        ratings_df["userId"].isin(similar_users) & ~ratings_df["tmdbId"].isin(selected_ids)
    ]
    if others_ratings.empty:
        return pd.DataFrame(), len(similar_users)

    agg = others_ratings.groupby("tmdbId")["rating"].agg(avg_rating="mean", num_ratings="count")
    # a couple of corroborating raters says more than one lone 5-star vote
    agg["score"] = agg["avg_rating"] * np.log1p(agg["num_ratings"])
    agg = agg.sort_values("score", ascending=False).head(top_n)

    result = movies_df[movies_df["tmdbId"].isin(agg.index)].copy()
    result = result.merge(agg[["avg_rating", "num_ratings"]], left_on="tmdbId", right_index=True)
    return result.sort_values("avg_rating", ascending=False), len(similar_users)


# ---------------------------------------------------------------------------
# Movie detail page
# ---------------------------------------------------------------------------

def show_movie_detail(movies, content_model, tmdb_id):
    match = movies[movies["tmdbId"] == tmdb_id]
    if match.empty:
        st.warning("Couldn't find that movie.")
        if st.button("← Back to search"):
            st.query_params.clear()
            st.rerun()
        return

    row = match.iloc[0]

    if st.button("← Back to search"):
        st.query_params.clear()
        st.rerun()

    left, right = st.columns([1, 2])
    with left:
        st.image(poster_url(row.get("poster_path"), large=True), use_container_width=True)

    with right:
        st.markdown(f"## {row['title']}  ({release_year(row.get('release_date'))})")
        st.markdown(render_stars(row.get("vote_average")), unsafe_allow_html=True)
        st.caption(f"{int(row['vote_count'])} ratings" if pd.notna(row.get("vote_count")) else "")

        genres = row.get("genre_names") or []
        if genres:
            badges = "".join(f'<span class="genre-badge">{html_escape.escape(g)}</span>' for g in genres)
            st.markdown(badges, unsafe_allow_html=True)

        st.markdown("#### Overview")
        st.write(row.get("overview", "No overview available."))

    st.divider()
    st.markdown("### More like this")
    recs = content_model.recommend(row["title"], top_n=10)
    if recs.empty:
        st.caption("No similar movies found.")
    else:
        show_movie_grid(recs, score_col="similarity_score")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Movie Recommender", layout="wide")
st.markdown(CARD_CSS, unsafe_allow_html=True)
st.title("🎬 Movie Recommendation System")

movies, ratings = load_data()
content_model = load_content_model(movies)

raw_movie_id = st.query_params.get("movie")
selected_movie_id = None
if raw_movie_id not in (None, ""):
    try:
        selected_movie_id = int(raw_movie_id)
    except (TypeError, ValueError):
        selected_movie_id = None

if selected_movie_id is not None:
    show_movie_detail(movies, content_model, selected_movie_id)
else:
    tab1, tab2 = st.tabs(["Content-Based", "Collaborative"])

    with tab1:
        st.subheader("Find movies")

        mode_col, input_col = st.columns([1, 2])
        with mode_col:
            with st.container(key="sb_search_mode"):
                search_mode = st.selectbox(
                    "Search by", SEARCH_MODES,
                    index=SEARCH_MODES.index(sticky_get("search_mode", "Movie Title")),
                    key="search_mode", on_change=sticky_save("search_mode"),
                )

        lock_dropdown_typing("sb_search_mode", "sb_sort_by")

        results = pd.DataFrame()

        with input_col:
            if search_mode == "Movie Title":
                title_options = get_sorted_titles(movies)
                persisted_title = sticky_get("title_pick")
                st.selectbox(
                    "Type or pick a movie title",
                    options=title_options,
                    index=title_options.index(persisted_title) if persisted_title in title_options else None,
                    placeholder="Type any part of a title (e.g. spider)",
                    key="title_pick", on_change=sticky_save("title_pick"),
                )

            elif search_mode == "Keyword":
                st.text_input(
                    "Enter a keyword (e.g. zombie, travel)",
                    value=sticky_get("keyword_query", ""),
                    key="keyword_query", on_change=sticky_save("keyword_query"),
                )
                query = sticky_get("keyword_query", "")
                if query:
                    results = filter_by_keyword(movies, query)

            elif search_mode == "Genre":
                all_genres = get_unique_genres(movies)
                persisted_genres = [g for g in sticky_get("genre_pick", []) if g in all_genres]
                st.multiselect(
                    "Pick one or more genres", all_genres,
                    default=persisted_genres,
                    key="genre_pick", on_change=sticky_save("genre_pick"),)
                picked_genres = sticky_get("genre_pick", [])
                if picked_genres:
                    results = filter_by_genre(movies, picked_genres)

            elif search_mode == "Overview":
                st.text_input(
                    "Describe a plot or theme (e.g. Garfield gets the one thing)",
                    value=sticky_get("overview_query", ""),
                    key="overview_query", on_change=sticky_save("overview_query"),
                )
                query = sticky_get("overview_query", "")
                if query:
                    results = filter_by_overview(movies, query)

            elif search_mode == "Rating":
                st.slider(
                    "Rating range", 0.0, 10.0,
                    value=sticky_get("rating_range", (6.0, 7.5)), step=0.1,
                    key="rating_range", on_change=sticky_save("rating_range"),
                )
                min_rating, max_rating = sticky_get("rating_range", (6.0, 7.5))
                results = filter_by_rating(movies, min_rating, max_rating)

        st.divider()

        if search_mode == "Movie Title":
            selected_title = sticky_get("title_pick")
            if selected_title:
                slider_col, label_col, drop_col, dir_col = st.columns([2.2, 0.8, 1.8, 0.5])
                with slider_col:
                    st.slider(
                        "Number of recommendations", 5, 20,
                        value=sticky_get("cb_top_n", 10),
                        key="cb_top_n", on_change=sticky_save("cb_top_n"),
                    )
                sort_by, ascending = render_sort_controls(
                    drop_col, dir_col, sort_key="cb_sort_by", asc_key="cb_sort_asc", label_col=label_col,
                    options=CB_SORT_OPTIONS, default_ascending_map=CB_SORT_DEFAULT_ASCENDING,
                )

                use_weighted = sort_by == "Weighted Rating"
                recs = content_model.recommend(
                    selected_title,
                    top_n=sticky_get("cb_top_n", 10),
                    use_popularity_filter=use_weighted,
                )
                if recs.empty:
                    st.warning("No recommendations found for that title.")
                else:
                    recs = apply_sort(recs, sort_by, ascending, field_map=CB_SORT_FIELD)
                    score_col = "weighted_rating" if use_weighted else "similarity_score"
                    show_movie_grid(recs, score_col=score_col)
            else:
                st.caption("Start typing a title above to search")
        else:
            if not results.empty:
                sort_by, ascending = render_sort_bar(sort_key="search_sort_by", asc_key="search_sort_asc")
                results = apply_sort(results, sort_by, ascending)
                st.caption(f"{len(results)} movie(s) found, click a poster to see details")
                show_movie_grid(results.head(20))
            elif search_mode in ("Keyword", "Overview"):
                st.caption("Type something above to search.")

    with tab2:
        st.subheader("Get personalized recommendations")

        CF_MODES = ["User ID method", "Select movies method"]
        with st.container(key="sb_cf_mode"):
            cf_mode = st.selectbox(
                "Method", CF_MODES,
                index=CF_MODES.index(sticky_get("cf_mode", CF_MODES[1])),
                key="cf_mode", on_change=sticky_save("cf_mode"),
            )
        lock_dropdown_typing("sb_cf_mode")

        if cf_mode == "User ID method":
            cf_model = load_cf_model(ratings)
            valid_users = sorted(ratings["userId"].unique())
            persisted_user = sticky_get("cf_user")
            user_choice = st.selectbox(
                "Pick a user ID", valid_users,
                index=valid_users.index(persisted_user) if persisted_user in valid_users else 0,
                key="cf_user", on_change=sticky_save("cf_user"),
            )
            top_n_cf = st.slider(
                "Number of recommendations", 5, 20,
                value=sticky_get("cf_top_n", 10),
                key="cf_top_n", on_change=sticky_save("cf_top_n"),
            )

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

        else:
            title_options = get_sorted_titles(movies)
            persisted_likes = [t for t in sticky_get("liked_titles", []) if t in title_options]
            st.multiselect(
                "Movies you like", title_options,
                default=persisted_likes,
                placeholder="Type to search and pick a few favorites",
                key="liked_titles", on_change=sticky_save("liked_titles"),
            )
            liked_titles = sticky_get("liked_titles", [])

            if liked_titles:
                top_n_anon = st.slider(
                    "Number of recommendations", 5, 20,
                    value=sticky_get("cf2_top_n", 10),
                    key="cf2_top_n", on_change=sticky_save("cf2_top_n"),
                )
                selected_ids = movies.loc[movies["title"].isin(liked_titles), "tmdbId"].tolist()
                recs, num_similar_users = recommend_from_liked_movies(
                    ratings, movies, selected_ids, top_n=top_n_anon
                )
                if recs.empty:
                    st.warning(
                        "Not enough overlapping rating data to build recommendations from "
                        "these picks yet. Try adding a few more movies."
                    )
                else:
                    st.caption(f"Based on {num_similar_users} other viewer(s) who also rated your picks highly.")
                    show_movie_grid(recs, score_col="avg_rating")
            else:
                st.caption("Pick a few movies you enjoy above to get started.")