"""
PlaydotG - Step 2: AI Recommendation Logic (Content-Based Filtering)
----------------------------------------------------------------------
Implements the IPO pipeline from the project brief:

    INPUT (user state)  ->  PROCESS (TF-IDF + cosine similarity)  ->  OUTPUT (Top-N list)

Pipeline:
    1. Ingestion  - take 3+ user inputs (genres + free-text mood)
    2. Scoring    - vectorize with TF-IDF, score every game with cosine similarity
    3. Sorting    - rank games by similarity score, descending
    4. Filtering  - truncate to the Top-N most relevant matches

Run this file directly to try a few example queries.
"""

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_FILE = "games_cleaned.csv"
GENRE_BOOST = 4          # how many times genre words are repeated to weight them higher
MAX_FEATURES = 600       # vocabulary size cap
DEFAULT_TOP_N = 5


class GameRecommender:
    """Content-based game recommender using TF-IDF + cosine similarity."""

    def __init__(self, data_path: str = DATA_FILE):
        self.df = pd.read_csv(data_path)
        self.df["doc"] = self.df.apply(self._make_document, axis=1)

        self.vectorizer = TfidfVectorizer(
            max_features=MAX_FEATURES,
            stop_words="english",
            token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z\-]+\b",
        )
        # Fit on the whole catalog once -- this is our shared vocabulary space
        self.item_matrix = self.vectorizer.fit_transform(self.df["doc"])

    @staticmethod
    def _make_document(row) -> str:
        """
        Combine genre tags (boosted) with plot text into one 'document' per game.
        Boosting genres means they carry more weight than incidental plot words,
        matching the 'specific tags should matter more than generic words' idea
        from the TF-IDF slide.
        """
        genres = str(row["genres"]).replace(",", " ").lower()
        boosted_genres = (genres + " ") * GENRE_BOOST
        plot = str(row["plot"]).lower()
        return boosted_genres + plot

    def recommend(self, genres: list[str], mood_text: str = "", top_n: int = DEFAULT_TOP_N) -> pd.DataFrame:
        """
        Ingestion  : genres (list) + free-text mood description
        Process    : build the same kind of document for the query, transform
                     it into the existing TF-IDF vocabulary space, score against
                     every game with cosine similarity
        Sorting    : rank descending
        Filtering  : slice to top_n
        """
        query_doc = ((" ".join(genres).lower() + " ") * GENRE_BOOST) + mood_text.lower()
        query_vector = self.vectorizer.transform([query_doc])

        scores = cosine_similarity(query_vector, self.item_matrix).flatten()

        results = self.df.copy()
        results["match_score"] = scores
        results = results.sort_values("match_score", ascending=False)
        results = results[results["match_score"] > 0]

        return results.head(top_n)[["name", "year", "genres", "rating", "match_score"]]


if __name__ == "__main__":
    engine = GameRecommender()

    examples = [
        (["Action", "Sci-Fi"], "space marines robots war"),
        (["Fantasy", "Adventure"], "dragons magic kingdom"),
        (["Comedy", "Family"], "funny lighthearted for kids"),
    ]

    for genres, mood in examples:
        print(f"\nQuery -> genres={genres}, mood='{mood}'")
        print("-" * 60)
        recs = engine.recommend(genres, mood, top_n=5)
        for _, row in recs.iterrows():
            score_pct = round(row["match_score"] * 100, 1)
            print(f"{score_pct:>5.1f}%  {row['name']}  ({row['year']})  [{row['genres']}]")
