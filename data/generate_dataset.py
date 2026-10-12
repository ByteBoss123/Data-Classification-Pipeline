"""
generate_dataset.py
Generates a realistic TMDb-style movie metadata dataset (CSV).
In production this would be replaced by pulling from s3://cinetag-raw-ingest/
"""

import pandas as pd
import numpy as np
import json
import os

np.random.seed(42)

MOVIES = [
    ("The Dark Knight", 2008, ["Action","Crime","Drama"], 8.9, 152, 185_000_000, "Christopher Nolan"),
    ("Inception", 2010, ["Action","Science Fiction","Thriller"], 8.8, 148, 836_000_000, "Christopher Nolan"),
    ("Interstellar", 2014, ["Adventure","Drama","Science Fiction"], 8.6, 169, 675_000_000, "Christopher Nolan"),
    ("Parasite", 2019, ["Comedy","Drama","Thriller"], 8.5, 132, 258_000_000, "Bong Joon-ho"),
    ("Mad Max: Fury Road", 2015, ["Action","Adventure","Science Fiction"], 8.1, 120, 378_000_000, "George Miller"),
    ("Her", 2013, ["Drama","Romance","Science Fiction"], 8.0, 126, 47_000_000, "Spike Jonze"),
    ("Get Out", 2017, ["Horror","Mystery","Thriller"], 7.7, 104, 255_000_000, "Jordan Peele"),
    ("The Grand Budapest Hotel", 2014, ["Comedy","Crime","Drama"], 8.1, 99, 174_000_000, "Wes Anderson"),
    ("Arrival", 2016, ["Drama","Mystery","Science Fiction"], 8.0, 116, 203_000_000, "Denis Villeneuve"),
    ("Whiplash", 2014, ["Drama","Music"], 8.5, 107, 49_000_000, "Damien Chazelle"),
    ("Moonlight", 2016, ["Drama","Romance"], 7.4, 111, 65_000_000, "Barry Jenkins"),
    ("Hereditary", 2018, ["Drama","Horror","Mystery"], 7.3, 127, 80_000_000, "Ari Aster"),
    ("The Revenant", 2015, ["Adventure","Drama","Thriller"], 8.0, 156, 533_000_000, "Alejandro Iñárritu"),
    ("La La Land", 2016, ["Comedy","Drama","Music","Romance"], 8.0, 128, 447_000_000, "Damien Chazelle"),
    ("1917", 2019, ["Drama","Thriller","War"], 8.3, 119, 384_000_000, "Sam Mendes"),
    ("Joker", 2019, ["Crime","Drama","Thriller"], 8.4, 122, 1_079_000_000, "Todd Phillips"),
    ("Blade Runner 2049", 2017, ["Drama","Mystery","Science Fiction"], 8.0, 164, 259_000_000, "Denis Villeneuve"),
    ("Ex Machina", 2014, ["Drama","Science Fiction","Thriller"], 7.7, 108, 36_000_000, "Alex Garland"),
    ("The Witch", 2015, ["Drama","Horror","Mystery"], 6.9, 92, 40_000_000, "Robert Eggers"),
    ("Annihilation", 2018, ["Adventure","Drama","Science Fiction"], 7.5, 115, 43_000_000, "Alex Garland"),
    ("Avengers: Endgame", 2019, ["Action","Adventure","Science Fiction"], 8.4, 181, 2_797_000_000, "Anthony Russo"),
    ("Spider-Man: Into the Spider-Verse", 2018, ["Action","Adventure","Animation"], 8.4, 117, 375_000_000, "Bob Persichetti"),
    ("Roma", 2018, ["Drama"], 7.7, 135, 1_000_000, "Alfonso Cuarón"),
    ("The Lighthouse", 2019, ["Drama","Fantasy","Horror"], 7.4, 109, 11_000_000, "Robert Eggers"),
    ("Marriage Story", 2019, ["Drama","Romance"], 7.9, 137, 2_000_000, "Noah Baumbach"),
    ("Knives Out", 2019, ["Comedy","Crime","Drama","Mystery"], 7.9, 130, 312_000_000, "Rian Johnson"),
    ("Portrait of a Lady on Fire", 2019, ["Drama","Romance"], 8.1, 122, 7_000_000, "Céline Sciamma"),
    ("Midsommar", 2019, ["Drama","Horror","Mystery"], 7.1, 148, 27_000_000, "Ari Aster"),
    ("Once Upon a Time in Hollywood", 2019, ["Comedy","Drama"], 7.6, 161, 374_000_000, "Quentin Tarantino"),
    ("The Irishman", 2019, ["Biography","Crime","Drama"], 7.8, 209, 250_000_000, "Martin Scorsese"),
]

ALL_GENRES = ["Action","Adventure","Animation","Biography","Comedy","Crime",
              "Drama","Fantasy","Horror","Music","Mystery","Romance",
              "Science Fiction","Thriller","War"]

def expand_movies(base, n=500):
    rows = []
    for _ in range(n):
        m = base[np.random.randint(len(base))]
        # add synthetic variation
        noise_rating = float(np.clip(m[3] + np.random.normal(0, 0.4), 1, 10))
        noise_runtime = int(np.clip(m[4] + np.random.randint(-15, 15), 60, 240))
        noise_revenue = int(np.clip(m[5] * np.random.uniform(0.3, 2.0), 0, 3e9))
        # occasionally corrupt genre list to simulate dirty data
        genres = list(m[2])
        if np.random.random() < 0.05:
            genres = []  # missing genres — will be caught by validation
        if np.random.random() < 0.08:
            genres.append("Unknown")  # dirty label
        rows.append({
            "movie_id": f"tmdb_{len(rows)+1:05d}",
            "title": m[0],
            "release_year": m[1] + np.random.randint(-2, 3),
            "genres_raw": json.dumps(genres),
            "vote_average": round(noise_rating, 1),
            "runtime_min": noise_runtime,
            "revenue": noise_revenue,
            "director": m[6],
            "overview": f"A {m[2][0].lower()} film directed by {m[6]}.",
            "primary_genre": m[2][0] if genres else None,
        })
    return rows

def main():
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs("data/processed", exist_ok=True)

    rows = expand_movies(MOVIES, n=500)
    df = pd.DataFrame(rows)
    out = "data/raw/tmdb_movies_raw.csv"
    df.to_csv(out, index=False)
    print(f"[generate_dataset] Saved {len(df)} records → {out}")
    print(df[["movie_id","title","primary_genre","vote_average","runtime_min"]].head(8).to_string())
    return df

if __name__ == "__main__":
    main()
