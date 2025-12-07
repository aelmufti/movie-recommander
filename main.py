from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llm_controls import recommend_movie
import requests

app = FastAPI(title="Ollama Movie Recommender API")


class PromptRequest(BaseModel):
    prompt: str
    model: str = "llama3"


@app.post("/recommend")
async def recommend(request: PromptRequest):
    try:
        movie = recommend_movie(request.prompt)
        return {"recommendation": movie}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")



@app.post("/generate")
async def generate_response(request: PromptRequest):
    try:
        ollama_payload = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": False  
        }

        response = requests.post(
            "http://localhost:11434/api/generate",
            json=ollama_payload
        )
        response.raise_for_status()

        ollama_response = response.json()
        return {"response": ollama_response.get("response", "")}

    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Ollama API error: {str(e)}")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Server error: {str(e)}")


@app.get("/")
async def root():
    return {
        "message": "Welcome to the Ollama-FastAPI Movie Recommender!",
        "endpoints": {
            "/recommend": "POST - Send 10-word vibe, get movie recommendation",
            "/generate": "POST - Raw LLM call"
        }
    }
