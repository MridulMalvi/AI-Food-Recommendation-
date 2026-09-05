"""FastAPI entry point for the Gwalior Food Recommendation Chatbot."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from backend.chatbot import FoodChatbot
from backend.recommender import FoodRecommender


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")  # Optional local file; never expose its contents to the frontend.

recommender = FoodRecommender(
    PROJECT_ROOT / "data" / "gwalior_food_dataset_330_unique.csv",
    PROJECT_ROOT / "backend" / "model" / "gwalior_food_recommendation_model.pkl",
)
chatbot = FoodChatbot(recommender.cuisines, recommender.areas)

app = FastAPI(title="Gwalior Food Recommendation API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be empty.")
        return value


@app.get("/api/status")
@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok", "service": "Gwalior Food Recommendation API"}


@app.post("/chat")
def chat(request: ChatRequest) -> dict:
    try:
        # Preserve preferences from this browser session while allowing the newest message to override them.
        preferences = chatbot.empty_preferences()
        for entry in request.history:
            if entry.role == "user":
                extracted = chatbot.extract_preferences(entry.content)
                preferences.update({key: value for key, value in extracted.items() if value is not None})
        current = chatbot.extract_preferences(request.message)
        preferences.update({key: value for key, value in current.items() if value is not None})
        recommendations, note = recommender.recommend(preferences)
        return {
            "success": True,
            "message": chatbot.generate_response(request.message, recommendations, preferences, note),
            "preferences": preferences,
            "recommendations": recommendations,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process this request: {exc}") from exc


# Mount static frontend files so the web app can be used directly from http://127.0.0.1:8000
frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

