"""
PlaydotG - Step 1: Data Cleaning & Preprocessing
--------------------------------------------------
Takes the raw IMDB Video Games dataset (imdb-videogames.csv) and produces
a cleaned dataset ready for TF-IDF vectorization.

Input : imdb-videogames.csv
Output: games_cleaned.csv
"""

import pandas as pd

INPUT_FILE = "imdb-videogames.csv"
OUTPUT_FILE = "games_cleaned.csv"

GENRE_COLUMNS = [
    "Action", "Adventure", "Comedy", "Crime", "Family",
    "Fantasy", "Mystery", "Sci-Fi", "Thriller"
]


def genre_string(row):
    """Collapse the 9 one-hot genre columns into a single comma-separated string."""
    return ", ".join([g for g in GENRE_COLUMNS if row[g]])


def clean_dataset(input_path: str) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    print(f"Raw rows: {len(df)}")

    # 1. Drop columns that are unused or too sparse to be useful
    df = df.drop(columns=["Unnamed: 0", "url", "certificate"], errors="ignore")

    # 2. Remove duplicate game titles (keep the first occurrence)
    before = len(df)
    df = df.drop_duplicates(subset="name", keep="first")
    print(f"Dropped {before - len(df)} duplicate titles")

    # 3. Drop rows with no genre tags at all (unusable for content-based filtering)
    before = len(df)
    df = df[df[GENRE_COLUMNS].sum(axis=1) > 0]
    print(f"Dropped {before - len(df)} rows with zero genres")

    # 4. Build a single combined genre string column
    df["genres"] = df.apply(genre_string, axis=1)

    # 5. Basic text cleanup on the plot column
    df["plot"] = df["plot"].astype(str).str.strip()

    # 6. Keep a tidy column order (retain one-hot genre columns too, useful for filters)
    final_cols = ["name", "year", "rating", "votes", "genres", "plot"] + GENRE_COLUMNS
    df = df[final_cols]

    print(f"Final rows: {len(df)}")
    return df


if __name__ == "__main__":
    cleaned = clean_dataset(INPUT_FILE)
    cleaned.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved cleaned dataset -> {OUTPUT_FILE}")
