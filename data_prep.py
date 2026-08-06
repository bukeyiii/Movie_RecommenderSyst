"""
data_prep.py
------------
ONE-STEP prep: filters a recent TMDB dataset down to a manageable size and
merges it with MovieLens ratings. No API key, no separate credits step.

Download 2 files first:

1) data/tmdb_full.csv
   From: https://www.kaggle.com/datasets/asaniczka/tmdb-movies-dataset-2023-930k-movies
   (save the CSV as data/tmdb_full.csv)

2) MovieLens ml-latest-small, unzipped into data/
   From: https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
   You need data/ratings.csv and data/links.csv from inside that zip.

Run:
    python data_prep.py

Produces:
    data/movies_clean.csv
    data/ratings_clean.csv
"""

import pandas as pd

DATA_DIR = "data"
MIN_RELEASE_DATE = "2023-01-01"
TOP_N = 4000  # keep it manageable


def split_field(x, sep=","):
    if not isinstance(x, str) or not x.strip():
        return []
    return [p.strip() for p in x.split(sep) if p.strip()]


def main():
    print("Loading TMDB dataset (large file, may take a moment)...")
    movies = pd.read_csv(f"{DATA_DIR}/tmdb_full.csv", low_memory=False)

    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    movies = movies.dropna(subset=["release_date", "title", "overview"])
    movies = movies[movies["overview"].str.len() > 20]
    movies = movies[movies["vote_count"].fillna(0) >= 5]

    print(f"Filtering to movies released {MIN_RELEASE_DATE}+ , keeping top {TOP_N} by popularity...")
    recent = movies[movies["release_date"] >= MIN_RELEASE_DATE]
    recent = recent.sort_values("popularity", ascending=False).head(TOP_N).copy()

    recent["genre_names"] = recent["genres"].apply(split_field)
    recent["keyword_names"] = recent["keywords"].apply(split_field)

    def make_soup(row):
        parts = row["genre_names"] * 2 + row["keyword_names"]  # weight genres a bit more
        parts = [str(p).replace(" ", "").lower() for p in parts if p]
        tagline = str(row.get("tagline", "") or "")
        return " ".join(parts) + " " + str(row.get("overview", "")) + " " + tagline

    recent["soup"] = recent.apply(make_soup, axis=1)
    recent["director"] = ""       # not available without the TMDB API — see README
    recent["cast_top5"] = [[] for _ in range(len(recent))]

    keep_cols = [
        "id", "title", "overview", "genre_names", "director", "cast_top5",
        "keyword_names", "release_date", "original_language", "runtime",
        "poster_path", "vote_average", "vote_count", "popularity", "soup",
    ]
    df_clean = recent[[c for c in keep_cols if c in recent.columns]].rename(columns={"id": "tmdbId"})
    df_clean = df_clean.drop_duplicates(subset=["tmdbId"])
    df_clean.to_csv(f"{DATA_DIR}/movies_clean.csv", index=False)
    print(f"Saved {len(df_clean)} movies -> data/movies_clean.csv")

    # ---- Ratings side (MovieLens ml-latest-small) ----
    print("Preparing ratings...")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    links = pd.read_csv(f"{DATA_DIR}/links.csv")
    links = links.dropna(subset=["tmdbId"])
    links["tmdbId"] = links["tmdbId"].astype(int)

    ratings = ratings.merge(links[["movieId", "tmdbId"]], on="movieId")
    ratings = ratings[ratings["tmdbId"].isin(df_clean["tmdbId"])]
    ratings_clean = ratings[["userId", "tmdbId", "rating"]]
    ratings_clean.to_csv(f"{DATA_DIR}/ratings_clean.csv", index=False)
    print(f"Saved {len(ratings_clean)} ratings -> data/ratings_clean.csv")

    if len(ratings_clean) < 500:
        print(
            "\nHeads up: MovieLens ratings predate most 2023+ releases, so overlap "
            "is naturally thin (classic cold-start problem). If you want a denser "
            "user-item matrix for the CF demo, lower MIN_RELEASE_DATE above (e.g. "
            "'2018-01-01') and rerun. Your content-based tab stays fully 'recent' "
            "either way since it doesn't depend on ratings."
        )


if __name__ == "__main__":
    main()
