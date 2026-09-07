"""FastAPI entry point for the Gwalior Food Recommendation Chatbot.

Endpoints
---------
  GET  /api/status              — health check
  POST /api/auth/register       — create a new account
  POST /api/auth/login          — authenticate, receive JWT
  GET  /api/auth/me             — return current user info
  POST /api/chat                — send a message, get recommendations + AI reply
  POST /api/favorite            — add/remove a dish from liked list
  POST /api/dislike             — add/remove a dish from disliked list
  POST /api/click               — record a dish view/click
  GET  /api/memory              — return current user's full memory profile
  GET  /api/favorites           — fetch liked dish IDs
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

import backend.recommender as rec_module
from backend.auth import create_access_token, get_current_user, get_optional_user, hash_password, verify_password
from backend.chatbot import FoodChatbot
import backend.db as db_module

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
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(title="Gwalior Food Recommendation API", version="3.0.0")

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


class RegisterRequest(BaseModel):
    username: str = Field(min_length=3, max_length=30)
    password: str = Field(min_length=6, max_length=128)

    @field_validator("username")
    @classmethod
    def username_alphanumeric(cls, v: str) -> str:
        v = v.strip()
        if not v.replace("_", "").replace("-", "").isalnum():
            raise ValueError("Username may only contain letters, numbers, _ and -")
        return v.lower()


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=30)
    password: str = Field(min_length=1, max_length=128)

    @field_validator("username")
    @classmethod
    def normalise(cls, v: str) -> str:
        return v.strip().lower()


class HistoryItem(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=1000)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=1000)
    history: list[HistoryItem] = Field(default_factory=list, max_length=12)
    user_id: str = Field(default="default_user")   # kept for backward compat

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


class DislikeRequest(BaseModel):
    item_id: int
    is_disliked: bool = True
    dish: Optional[str] = None


class ClickRequest(BaseModel):
    item_id: int
    dish: Optional[str] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_memory(username: str) -> dict[str, Any]:
    """Load user memory from DB, return default dict on error."""
    try:
        return db_module.get_memory(username)
    except Exception:
        return {"liked_ids": [], "disliked_ids": [], "clicked_ids": [], "preferences_history": []}


# ---------------------------------------------------------------------------
# Auth Endpoints
# ---------------------------------------------------------------------------


@app.post("/api/auth/register")
def register(req: RegisterRequest) -> dict[str, Any]:
    """Create a new user account."""
    existing = db_module.get_user(req.username)
    if existing:
        raise HTTPException(status_code=409, detail="Username already taken.")
    hashed = hash_password(req.password)
    ok = db_module.create_user(req.username, hashed)
    if not ok:
        raise HTTPException(status_code=409, detail="Username already taken.")
    token = create_access_token(req.username)
    return {"success": True, "username": req.username, "access_token": token, "token_type": "bearer"}


@app.post("/api/auth/login")
def login(req: LoginRequest) -> dict[str, Any]:
    """Authenticate and return a JWT."""
    user_doc = db_module.get_user(req.username)
    if not user_doc or not verify_password(req.password, user_doc["hashed_password"]):
        raise HTTPException(status_code=401, detail="Invalid username or password.")
    token = create_access_token(req.username)
    return {"success": True, "username": req.username, "access_token": token, "token_type": "bearer"}


@app.get("/api/auth/me")
def me(username: str = Depends(get_current_user)) -> dict[str, Any]:
    """Return the currently authenticated user's info."""
    mem = _load_memory(username)
    return {
        "success": True,
        "username": username,
        "liked_count": len(mem.get("liked_ids", [])),
        "disliked_count": len(mem.get("disliked_ids", [])),
        "clicked_count": len(mem.get("clicked_ids", [])),
    }


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


@app.get("/api/status")
@app.get("/status")
def status() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "Gwalior Food Recommendation API v3",
        "db": "connected" if db_module.is_available() else "in-memory fallback",
    }


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------


@app.post("/api/chat")
@app.post("/chat")
def chat(
    request: ChatRequest,
    username: str | None = Depends(get_optional_user),
) -> dict[str, Any]:
    """Send a chat message; returns AI reply + structured recommendation cards."""
    try:
        # Resolve user_id: prefer JWT username, fall back to request field
        user_id = username or request.user_id or "default_user"

        # Load full memory from DB
        mem = _load_memory(user_id)
        liked_ids: list[int] = mem.get("liked_ids", [])
        disliked_ids: list[int] = mem.get("disliked_ids", [])
        clicked_ids: list[int] = mem.get("clicked_ids", [])

        history_dicts = [{"role": h.role, "content": h.content} for h in request.history]

        reply_text, recommendations, preferences, category_info = chatbot.chat_with_agent(
            message=request.message,
            history=history_dicts,
            user_liked_ids=liked_ids,
        )

        # Pass full memory into recommender for richer personalisation
        if recommendations:
            recommendations = rec_module.recommend(
                mood=preferences.get("mood"),
                budget=preferences.get("budget"),
                location=preferences.get("location"),
                dietary=preferences.get("dietary"),
                spicy=preferences.get("spicy"),
                user_liked_ids=liked_ids,
                user_disliked_ids=disliked_ids,
                user_clicked_ids=clicked_ids,
                top_n=5,
            )

        # Persist preference snapshot
        if any(v is not None for v in preferences.values()):
            try:
                db_module.record_preferences(user_id, preferences)
            except Exception:
                pass

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


# ---------------------------------------------------------------------------
# Favorite / Dislike / Click
# ---------------------------------------------------------------------------


@app.post("/api/favorite")
@app.post("/favorite")
def toggle_favorite(
    item: FavoriteRequest,
    username: str | None = Depends(get_optional_user),
) -> dict[str, Any]:
    """Add or remove a dish ID from the user's liked list."""
    try:
        user_id = username or "default_user"
        db_module.toggle_liked(user_id, item.item_id, item.is_favorite)
        mem = _load_memory(user_id)
        return {
            "success": True,
            "action": "added" if item.is_favorite else "removed",
            "item_id": item.item_id,
            "total_liked": len(mem.get("liked_ids", [])),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to update favorite: {exc}") from exc


@app.post("/api/dislike")
def toggle_dislike(
    item: DislikeRequest,
    username: str | None = Depends(get_optional_user),
) -> dict[str, Any]:
    """Add or remove a dish from the user's disliked list."""
    try:
        user_id = username or "default_user"
        db_module.toggle_disliked(user_id, item.item_id, item.is_disliked)
        mem = _load_memory(user_id)
        return {
            "success": True,
            "action": "disliked" if item.is_disliked else "un-disliked",
            "item_id": item.item_id,
            "total_disliked": len(mem.get("disliked_ids", [])),
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to update dislike: {exc}") from exc


@app.post("/api/click")
def record_click(
    item: ClickRequest,
    username: str | None = Depends(get_optional_user),
) -> dict[str, Any]:
    """Record that the user viewed / clicked on a dish card."""
    try:
        user_id = username or "default_user"
        db_module.record_click(user_id, item.item_id)
        return {"success": True, "item_id": item.item_id}
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# Memory / Favorites
# ---------------------------------------------------------------------------


@app.get("/api/memory")
def get_memory_profile(username: str = Depends(get_current_user)) -> dict[str, Any]:
    """Return the full memory profile for the authenticated user."""
    mem = _load_memory(username)
    return {
        "success": True,
        "username": username,
        "liked_ids": mem.get("liked_ids", []),
        "disliked_ids": mem.get("disliked_ids", []),
        "clicked_ids": mem.get("clicked_ids", []),
        "preferences_history": mem.get("preferences_history", [])[-10:],  # last 10
    }


@app.get("/api/favorites")
@app.get("/favorites")
@app.get("/api/favorite")
def get_favorites(username: str | None = Depends(get_optional_user)) -> dict[str, Any]:
    """Return the current user's liked dish IDs."""
    user_id = username or "default_user"
    mem = _load_memory(user_id)
    liked = mem.get("liked_ids", [])
    return {"success": True, "liked_ids": liked, "total": len(liked)}


@app.get("/api/category/{category_key}")
def category_results(category_key: str, top_n: int = 5) -> dict[str, Any]:
    """Return top dishes for a specific cuisine / food category."""
    try:
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


# ---------------------------------------------------------------------------
# Mount static frontend (MUST be last — catch-all)
# ---------------------------------------------------------------------------

frontend_dir = PROJECT_ROOT / "frontend"
if frontend_dir.is_dir():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
