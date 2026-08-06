"""
data_prep.py
------------
ONE-STEP prep: filters a recent TMDB dataset down to a manageable size and
merges it with MovieLens ratings. No API key, no separate credits step.

NOTE: matches the columns in TMDB_movie_dataset_v11.csv:
    id, title, release_date, vote_average, vote_count, overview,
    popularity, poster_path, genres, keywords
This dataset has no tagline / runtime / original_language columns, so those
are left out of the "soup" text and the cleaned output.

Files needed in data/ before running:
    data/tmdb_full.csv   <- TMDB_movie_dataset_v11.csv, renamed
    data/ratings.csv     <- MovieLens ml-latest-small ratings.csv
    data/links.csv       <- MovieLens ml-latest-small links.csv

Run:
    python data_prep.py

Produces:
    data/movies_clean.csv
    data/ratings_clean.csv
"""

import pandas as pd

DATA_DIR = "data"
TOP_N = 4000  # keep it manageable — dataset is filtered by popularity, not by date

# this dataset has no 'adult' flag column, so filter explicit content out via
# keyword matching on title/overview/keywords instead
ADULT_CONTENT_PATTERN = (
    r"porn|hentai|xxx|erotic|orgy|nudity|softcore|tits|boobs|busty|milf"
    r"|gangbang|threesome|stepmom|stepsister|onlyfans"
)


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

    adult_mask = (
        movies["title"].str.contains(ADULT_CONTENT_PATTERN, case=False, na=False, regex=True)
        | movies["overview"].str.contains(ADULT_CONTENT_PATTERN, case=False, na=False, regex=True)
        | movies["keywords"].str.contains(ADULT_CONTENT_PATTERN, case=False, na=False, regex=True)
    )
    print(f"Dropping {adult_mask.sum()} rows matching adult-content keywords...")
    movies = movies[~adult_mask]

    print(f"Keeping top {TOP_N} by popularity...")
    recent = movies.sort_values("popularity", ascending=False).head(TOP_N).copy()

    recent["genre_names"] = recent["genres"].apply(split_field)
    recent["keyword_names"] = recent["keywords"].apply(split_field)

    def make_soup(row):
        parts = row["genre_names"] * 2 + row["keyword_names"]  # weight genres a bit more
        parts = [str(p).replace(" ", "").lower() for p in parts if p]
        return " ".join(parts) + " " + str(row.get("overview", ""))

    recent["soup"] = recent.apply(make_soup, axis=1)
    recent["director"] = ""       # not available without the TMDB API — see README
    recent["cast_top5"] = [[] for _ in range(len(recent))]
    recent["release_date"] = recent["release_date"].dt.strftime("%Y-%m-%d")

    keep_cols = [
        "id", "title", "overview", "genre_names", "director", "cast_top5",
        "keyword_names", "release_date", "poster_path", "vote_average",
        "vote_count", "popularity", "soup",
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
            "\nHeads up: overlap between MovieLens ratings and your movie shortlist "
            "is thin (classic cold-start problem). If you want a denser user-item "
            "matrix for the CF demo, raise TOP_N above and rerun. Your content-based "
            "tab isn't affected either way since it doesn't depend on ratings."
        )


if __name__ == "__main__":
    main()
