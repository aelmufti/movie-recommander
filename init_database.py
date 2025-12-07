from lancedb import connect
import ollama
import pandas as pd

def embed(text: str):
    return ollama.embed("nomic-embed-text", text)["embeddings"][0]

def build_description(row):
    parts = []
    if pd.notna(row.get("overview")):
        parts.append(row["overview"])
    if pd.notna(row.get("tagline")):
        parts.append(row["tagline"])
    if pd.notna(row.get("genres")):
        parts.append("Genres: " + row["genres"])
    if pd.notna(row.get("keywords")):
        parts.append("Keywords: " + row["keywords"])
    if pd.notna(row.get("cast")):
        parts.append("Cast: " + row["cast"])
    return " | ".join(parts)

print("Loading CSV...")
df = pd.read_csv("movies.csv")

print("Building descriptions...")
df["description"] = df.apply(build_description, axis=1)

print("Connecting to database...")
db = connect("./movie_vectors.lancedb")

table_df = df[["title", "description"]].copy()

print(f"Creating embeddings for {len(table_df)} movies...")
vectors = []
for i, text in enumerate(table_df["description"]):
    vectors.append(embed(text))
    if (i + 1) % 10 == 0:
        print(f"  Embedded {i + 1}/{len(table_df)} movies...")

table_df["vector"] = vectors

print("Creating database table...")
db.create_table("movies", data=table_df, mode="overwrite")

print("✅ Database initialized successfully!")