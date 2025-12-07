from lancedb import connect
import ollama

# Connexion rapide à la DB existante
db = connect("./movie_vectors.lancedb")
tbl = db.open_table("movies")  # ⚡ Rapide - juste ouvre la table existante

def embed(text: str):
    return ollama.embed("nomic-embed-text", text)["embeddings"][0]

def recommend_movie(prompt: str) -> str:
    """Semantic search + LLM reasoning → final movie recommendation."""
    
    query_vec = embed(prompt)
    matches = tbl.search(query_vec).limit(5).to_pandas()

    system_prompt = """
You are a movie recommendation expert.
From the candidate movies, choose *one* best movie that matches the user's prompt.
Answer format:
Movie Title — short reason.
"""

    response = ollama.chat(
        model="llama3",
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": f"User wants: {prompt}\n\nHere are candidate movies:\n{matches}"
            }
        ]
    )

    return response["message"]["content"]