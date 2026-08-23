import pandas as pd

# Store in data/..
DATA_DIR = "data"
# Take first 10000 in movie dataset
TOP_N = 10000

ADULT_CONTENT_PATTERN = (
    r"porn|hentai|xxx|erotic|nudity|softcore"
    r"|orgy|threesome|fetish\s*sex|hardcore\s*porn"
    r"|adult\s*video|pornstar|porn\s*star"
    r"|lust[- ]?fueled"
)

# Split it by (,) remove spaces and empty and return the result as a list
def split_field(x, sep=","):
    # Check the value isn't text or the text is empty then return an empty list
    if not isinstance(x, str) or not x.strip():
        return []
    return [p.strip() for p in x.split(sep) if p.strip()]


def main():
    # Read csv, then store in movie dataframes
    movies = pd.read_csv(f"{DATA_DIR}/TMDB_movie_dataset_v11.csv", low_memory=False)

    # Convert date into same format
    # errors = "Coerce" : return NaT if value is not date 
    movies["release_date"] = pd.to_datetime(movies["release_date"], errors="coerce")
    # Delete the movies no release date, title n overview
    movies = movies.dropna(subset=["release_date", "title", "overview"])
    # Remove short descriptions
    movies = movies[movies["overview"].str.len() > 20]
    # Remove movies with <= 10 votes
    # fillna(0) = If movie not count than become 0
    movies = movies[movies["vote_count"].fillna(0) > 10]

    # Store true or false of each movie
    adult_mask = (
        movies["title"].str.contains(ADULT_CONTENT_PATTERN, case=False, na=False, regex=True)
        | movies["overview"].str.contains(ADULT_CONTENT_PATTERN, case=False, na=False, regex=True)
        | movies["keywords"].str.contains(ADULT_CONTENT_PATTERN, case=False, na=False, regex=True)
    )
    # Removing the movie contain adult content
    print(f"Dropping {adult_mask.sum()} rows matching adult-content keywords...")
    movies = movies[~adult_mask]

    # Retrieve first 10000 movie sort by popularity
    print(f"Keeping top {TOP_N} by popularity...")
    top_movie = movies.sort_values("popularity", ascending=False).head(TOP_N).copy()
    top_movie["genre_names"] = top_movie["genres"].apply(split_field)
    top_movie["keyword_names"] = top_movie["keywords"].apply(split_field)

    # Soup = Genre + Keyword + Overview
    def make_soup(row):
        # list + list
        parts = row["genre_names"] + row["keyword_names"]
        parts = [str(p).replace(" ", "").lower() for p in parts if p]
        # Overview is string
        return " ".join(parts) + " " + str(row.get("overview", ""))

    # Apply soup to every row
    top_movie["soup"] = top_movie.apply(make_soup, axis=1)
    top_movie["release_date"] = top_movie["release_date"].dt.strftime("%Y-%m-%d")

    keep_cols = [
        "id", "title", "overview", "genre_names",
        "keyword_names", "release_date", "poster_path", "vote_average",
        "vote_count", "popularity", "soup",
    ]

    # Selects the columns from keep_cols that actually exist in top_movie
    df_clean = top_movie[[c for c in keep_cols if c in top_movie.columns]].rename(columns={"id": "tmdbId"})
    # Ensures each TMDB ID appears only once
    df_clean = df_clean.drop_duplicates(subset=["tmdbId"])
    # Saves final DataFrame
    # index=False : Don't save Pandas' row numbers as an extra column
    df_clean.to_csv(f"{DATA_DIR}/movies_clean.csv", index=False)
    print(f"Saved {len(df_clean)} movies -> data/movies_clean.csv")

    # ---- Ratings side (MovieLens ml-latest-small) ----
    print("Preparing ratings...")
    ratings = pd.read_csv(f"{DATA_DIR}/ratings.csv")
    links = pd.read_csv(f"{DATA_DIR}/links.csv")
    links = links.dropna(subset=["tmdbId"])
    # tmdbId -> Integer
    links["tmdbId"] = links["tmdbId"].astype(int)

    # Matching rating with links
    ratings = ratings.merge(links[["movieId", "tmdbId"]], on="movieId")
    # Keep rating movie that exist in movie clean
    ratings = ratings[ratings["tmdbId"].isin(df_clean["tmdbId"])]
    # TmdbId as movie identifier
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