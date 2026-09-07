"""FastAPI entry point for the Gwalior Food Recommendation Chatbot.

Endpoints:
  GET  /api/status      — health check
  POST /api/chat        — send a message, get recommendations + AI reply
  POST /api/favorite    — add/remove a dish from the user's liked list
  GET  /api/favorites   — fetch all current liked dishes
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import backend.recommender as rec_module
from backend.chatbot import FoodChatbot

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# Initialise chatbot (recommender module loads at import time)
chatbot = FoodChatbot(
    cuisines=rec_module.get_cuisines(),
    areas=rec_module.get_areas(),
)

# ---------------------------------------------------------------------------
# In-memory user profile store
# ---------------------------------------------------------------------------

user_profiles: dict[str, dict[str, Any]] = {
    "default_user": {"liked_ids": []},
}

# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Gwalior Food Recommendation API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)
    user_id: str = Field(default="default_user")

    @field_validator("message")
    @classmethod
    def message_must_not_be_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message cannot be empty.")
        return value


class FavoriteRequest(BaseModel):
    """Minimal payload for toggling a dish like/unlike."""
    item_id: int = Field(description="The numeric ID of the dish to like/unlike.")
    is_favorite: bool = Field(default=True, description="True = like, False = unlike.")
    # Optional metadata stored alongside the liked ID for display purposes
    dish: Optional[str] = None
    restaurant: Optional[str] = None
    cuisine: Optional[str] = None
    price: Optional[float] = None
    rating: Optional[float] = None
    area: Optional[str] = None
    dietary: Optional[str] = None
    spicy: Optional[bool] = None
    image_url: Optional[str] = None

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/api/status")
@app.get("/status")
def status() -> dict[str, str]:
    return {"status": "ok", "service": "Gwalior Food Recommendation API v2"}


@app.post("/api/chat")
@app.post("/chat")
def chat(request: ChatRequest) -> dict[str, Any]:
    """Send a chat message; returns AI reply + structured recommendation cards."""
    try:
        profile = user_profiles.setdefault(request.user_id, {"liked_ids": []})
        liked_ids: list[int] = profile.get("liked_ids", [])

        history_dicts = [{"role": h.role, "content": h.content} for h in request.history]

        reply_text, recommendations, preferences, category_info = chatbot.chat_with_agent(
            message=request.message,
            history=history_dicts,
            user_liked_ids=liked_ids,
        )

        # When a specific category is detected, fetch dedicated results
        category_recs: list[dict[str, Any]] = []
        category_meta: dict[str, str] | None = None
        if category_info:
            cat_key, cat_label, cat_emoji = category_info
            category_recs = rec_module.recommend_by_category(
                category_key=cat_key,
                top_n=5,
                user_liked_ids=liked_ids,
            )
            category_meta = {"key": cat_key, "label": cat_label, "emoji": cat_emoji}

        return {
            "success": True,
            "message": reply_text,
            "preferences": preferences,
            "recommendations": recommendations,
            "personalized": len(liked_ids) > 0,
            "category": category_meta,
            "category_recommendations": category_recs,
        }
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to process request: {exc}") from exc


@app.get("/api/category/{category_key}")
def category_results(category_key: str, top_n: int = 5) -> dict[str, Any]:
    """Return top dishes for a specific cuisine / food category."""
    try:
        # Resolve via CATEGORY_MAP first
        info = rec_module.CATEGORY_MAP.get(category_key.lower().replace("-", " "))
        label = info["label"] if info else category_key.replace("-", " ").title()
        emoji = info["emoji"] if info else "🍽️"
        results = rec_module.recommend_by_category(
            category_key=category_key.lower().replace("-", " "),
            top_n=min(top_n, 10),
        )
        return {"success": True, "category": {"key": category_key, "label": label, "emoji": emoji}, "results": results}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.post("/api/favorite")
@app.post("/favorite")
def toggle_favorite(item: FavoriteRequest) -> dict[str, Any]:
    """Add or remove a dish ID from the default user's liked list."""
    try:
        profile = user_profiles.setdefault("default_user", {"liked_ids": []})
        liked_ids: list[int] = profile["liked_ids"]

        if item.is_favorite:
            if item.item_id not in liked_ids:
                liked_ids.append(item.item_id)
            action = "added"
        else:
            profile["liked_ids"] = [i for i in liked_ids if i != item.item_id]
            action = "removed"

        return {
            "success": True,
            "action": action,
            "item_id": item.item_id,
            "total_liked": len(profile["liked_ids"]),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to update favorite: {exc}") from exc


@app.get("/api/favorites")
@app.get("/favorites")
@app.get("/api/favorite")
def get_favorites() -> dict[str, Any]:
    """Return the current user's liked dish IDs."""
    profile = user_profiles.get("default_user", {"liked_ids": []})
    return {
        "success": True,
        "liked_ids": profile.get("liked_ids", []),
        "total": len(profile.get("liked_ids", [])),
    }


# ---------------------------------------------------------------------------
# Mount static frontend
# ---------------------------------------------------------------------------

frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
