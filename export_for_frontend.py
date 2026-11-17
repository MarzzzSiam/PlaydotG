"""
PlaydotG - Step 3: Export vectors for the web frontend
---------------------------------------------------------
The PlaydotG frontend (playdotg.html) runs entirely in the browser with no
backend server. To make that possible, this script precomputes everything
that would normally be done live in Python:

  - the shared TF-IDF vocabulary (600 words)
  - each word's IDF weight
  - each game's TF-IDF vector, already L2-normalized (stored sparsely as
    index/weight pairs to keep the file small)
  - the English stopword list, so the browser can tokenize user input the
    same way scikit-learn does

The browser then only needs to: tokenize the user's genres + free text,
multiply by the precomputed IDF weights, L2-normalize, and take a dot
product against each game's vector -- which is exactly cosine similarity,
since both vectors are unit length.

Output: playdotg_data.json  (~5 MB, embedded directly into the HTML file)
"""

import json
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS

DATA_FILE = "games_cleaned.csv"
OUTPUT_FILE = "playdotg_data.json"
GENRE_BOOST = 4
MAX_FEATURES = 600
PLOT_SNIPPET_LEN = 220


def make_document(row) -> str:
    genres = str(row["genres"]).replace(",", " ").lower()
    boosted_genres = (genres + " ") * GENRE_BOOST
    plot = str(row["plot"]).lower()
    return boosted_genres + plot


def main():
    df = pd.read_csv(DATA_FILE)
    df["doc"] = df.apply(make_document, axis=1)

    vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
    )
    tfidf_matrix = vectorizer.fit_transform(df["doc"])

    vocab = vectorizer.get_feature_names_out().tolist()
    idf = [round(float(x), 4) for x in vectorizer.idf_]

    games = []
    df = df.reset_index(drop=True)
    for i, row in df.iterrows():
        vec = tfidf_matrix[i]
        indices = vec.indices.tolist()
        weights = [round(float(w), 4) for w in vec.data.tolist()]

        plot = str(row["plot"])
        if len(plot) > PLOT_SNIPPET_LEN:
            plot = plot[: PLOT_SNIPPET_LEN - 3] + "..."

        games.append({
            "n": row["name"],                                        # name
            "y": int(row["year"]) if pd.notna(row["year"]) else None,  # year
            "r": round(float(row["rating"]), 1) if pd.notna(row["rating"]) else None,  # rating
            "g": row["genres"],                                      # genres string
            "p": plot,                                                # plot snippet
            "v": indices,                                             # sparse vector indices
            "w": weights,                                             # sparse vector weights
        })

    payload = {
        "vocab": vocab,
        "idf": idf,
        "games": games,
        "sw": sorted(ENGLISH_STOP_WORDS),
    }

    with open(OUTPUT_FILE, "w") as f:
        json.dump(payload, f, separators=(",", ":"))

    print(f"Vocabulary size : {len(vocab)}")
    print(f"Games exported  : {len(games)}")
    print(f"Saved -> {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
