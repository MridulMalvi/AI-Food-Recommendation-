"""MongoDB connection and user-memory helpers for Gwalior Foodie.

Collections
-----------
users        : { _id, username (unique), hashed_password, created_at }
user_memory  : { user_id (unique), liked_ids[], disliked_ids[],
                 clicked_ids[], preferences_history[] }

Falls back gracefully to an in-memory dict store when MONGODB_URI is not set,
so the app still runs during local development without MongoDB.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Any

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MONGODB_URI = os.getenv("MONGODB_URI", "").strip()
DB_NAME = "gwalior_foodie"

# ---------------------------------------------------------------------------
# MongoDB client (lazy init)
# ---------------------------------------------------------------------------

_client = None
_db = None


def _get_db():
    """Return the pymongo Database, connecting on first call."""
    global _client, _db
    if _db is not None:
        return _db
    if not MONGODB_URI:
        raise RuntimeError(
            "MONGODB_URI is not set. Add it to .env to enable persistent storage."
        )
    try:
        from pymongo import MongoClient
        from pymongo.server_api import ServerApi

        _client = MongoClient(MONGODB_URI, server_api=ServerApi("1"), serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")          # verify connection
        _db = _client[DB_NAME]

        # Ensure indexes
        _db["users"].create_index("username", unique=True)
        _db["user_memory"].create_index("user_id", unique=True)
        logger.info("Connected to MongoDB Atlas — database: %s", DB_NAME)
        return _db
    except Exception as exc:
        logger.error("MongoDB connection failed: %s", exc)
        raise


def is_available() -> bool:
    """Return True if MongoDB is reachable."""
    try:
        _get_db()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# In-memory fallback store (used when MongoDB is unavailable)
# ---------------------------------------------------------------------------

_mem_users: dict[str, dict[str, Any]] = {}
_mem_memory: dict[str, dict[str, Any]] = {}


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(username: str, hashed_password: str) -> bool:
    """Insert a new user. Returns False if username already exists."""
    doc = {
        "username": username,
        "hashed_password": hashed_password,
        "created_at": datetime.now(timezone.utc),
    }
    try:
        db = _get_db()
        db["users"].insert_one(doc)
        return True
    except Exception:
        # Fallback to in-memory
        if username in _mem_users:
            return False
        _mem_users[username] = doc
        return True


def get_user(username: str) -> dict[str, Any] | None:
    """Fetch a user document by username."""
    try:
        db = _get_db()
        return db["users"].find_one({"username": username})
    except Exception:
        return _mem_users.get(username)


# ---------------------------------------------------------------------------
# User Memory CRUD
# ---------------------------------------------------------------------------

def _default_memory(user_id: str) -> dict[str, Any]:
    return {
        "user_id": user_id,
        "liked_ids": [],
        "disliked_ids": [],
        "clicked_ids": [],
        "preferences_history": [],
    }


def get_memory(user_id: str) -> dict[str, Any]:
    """Return the memory doc for user_id, creating it if absent."""
    try:
        db = _get_db()
        doc = db["user_memory"].find_one({"user_id": user_id})
        if doc is None:
            doc = _default_memory(user_id)
            db["user_memory"].insert_one(doc)
        return doc
    except Exception:
        if user_id not in _mem_memory:
            _mem_memory[user_id] = _default_memory(user_id)
        return _mem_memory[user_id]


def toggle_liked(user_id: str, dish_id: int, is_favorite: bool) -> None:
    """Add or remove a dish from the user's liked list."""
    try:
        db = _get_db()
        op = {"$addToSet": {"liked_ids": dish_id}} if is_favorite else {"$pull": {"liked_ids": dish_id}}
        db["user_memory"].update_one({"user_id": user_id}, op, upsert=True)
        # Also remove from disliked if now liked
        if is_favorite:
            db["user_memory"].update_one({"user_id": user_id}, {"$pull": {"disliked_ids": dish_id}})
    except Exception:
        mem = get_memory(user_id)
        if is_favorite:
            if dish_id not in mem["liked_ids"]:
                mem["liked_ids"].append(dish_id)
            mem["disliked_ids"] = [i for i in mem["disliked_ids"] if i != dish_id]
        else:
            mem["liked_ids"] = [i for i in mem["liked_ids"] if i != dish_id]


def toggle_disliked(user_id: str, dish_id: int, is_disliked: bool) -> None:
    """Add or remove a dish from the user's disliked list."""
    try:
        db = _get_db()
        op = {"$addToSet": {"disliked_ids": dish_id}} if is_disliked else {"$pull": {"disliked_ids": dish_id}}
        db["user_memory"].update_one({"user_id": user_id}, op, upsert=True)
        # Also remove from liked if now disliked
        if is_disliked:
            db["user_memory"].update_one({"user_id": user_id}, {"$pull": {"liked_ids": dish_id}})
    except Exception:
        mem = get_memory(user_id)
        if is_disliked:
            if dish_id not in mem["disliked_ids"]:
                mem["disliked_ids"].append(dish_id)
            mem["liked_ids"] = [i for i in mem["liked_ids"] if i != dish_id]
        else:
            mem["disliked_ids"] = [i for i in mem["disliked_ids"] if i != dish_id]


def record_click(user_id: str, dish_id: int) -> None:
    """Record that the user clicked / viewed a dish."""
    try:
        db = _get_db()
        db["user_memory"].update_one(
            {"user_id": user_id},
            {"$addToSet": {"clicked_ids": dish_id}},
            upsert=True,
        )
    except Exception:
        mem = get_memory(user_id)
        if dish_id not in mem["clicked_ids"]:
            mem["clicked_ids"].append(dish_id)


def record_preferences(user_id: str, prefs: dict[str, Any]) -> None:
    """Append a preference snapshot to the history (max 50 entries)."""
    entry = {**prefs, "ts": datetime.now(timezone.utc).isoformat()}
    try:
        db = _get_db()
        db["user_memory"].update_one(
            {"user_id": user_id},
            {
                "$push": {
                    "preferences_history": {
                        "$each": [entry],
                        "$slice": -50,    # keep latest 50
                    }
                }
            },
            upsert=True,
        )
    except Exception:
        mem = get_memory(user_id)
        mem["preferences_history"].append(entry)
        mem["preferences_history"] = mem["preferences_history"][-50:]
