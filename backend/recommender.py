"""TF-IDF + Cosine Similarity + Quality Scaling recommender for Gwalior food.

Algorithm:
  1. Build a 'content soup' for each dish from mood_tags, meal_type, dietary,
     cuisine, spicy, and area — then fit a TfidfVectorizer at startup.
  2. Compute a base_quality score (0-1) by scaling 'rating' (higher = better)
     and 'delivery_minutes' (lower = better) via MinMaxScaler.
  3. At query time:
     - Build a query soup from user inputs.
     - Compute cosine similarity between query and all dishes.
     - Apply a personal_boost when a dish's cuisine matches the cuisines of
       the user's liked items.
     - Combine scores: final = 0.55*sim + 0.30*quality + 0.15*personal_boost
     - Hard-filter by budget and dietary before scoring.
  4. Return top-N results with id, dish, restaurant, price, rating,
     match_percent, reason, and other display fields.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "gwalior_food_dataset_330_unique.csv"

REQUIRED_COLUMNS = {
    "id", "restaurant", "dish", "cuisine", "price", "rating", "city", "area",
    "dietary", "spicy", "mood_tags", "meal_type", "distance_km",
    "delivery_minutes", "image_url",
}

# Score weights (must sum to 1.0)
W_SIM = 0.55
W_QUALITY = 0.30
W_BOOST = 0.15


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_tags(value: Any) -> list[str]:
    """Parse comma-separated tag strings like 'comfort, happy' into a list."""
    raw = str(value or "").strip("[] ")
    return [item.strip().strip("'\"").lower() for item in raw.split(",") if item.strip().strip("'\"")]


def _to_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes", "y"}


def _build_soup(row: pd.Series) -> str:
    """Concatenate all content features into a single searchable text blob."""
    parts = []
    # mood_tags — repeat twice to upweight mood matching
    moods = _split_tags(row.get("mood_tags", ""))
    parts.extend(moods * 2)
    # meal_type
    parts.extend(_split_tags(row.get("meal_type", "")))
    # dietary
    dietary = str(row.get("dietary", "")).strip().lower()
    if dietary:
        parts.append(dietary)
        parts.append(dietary.replace("-", "").replace(" ", ""))  # "nonvegetarian"
    # cuisine
    cuisine = str(row.get("cuisine", "")).strip().lower()
    if cuisine:
        parts.extend(cuisine.split())
    # spicy
    if _to_bool(row.get("spicy", False)):
        parts.extend(["spicy", "hot", "masaledar"])
    else:
        parts.extend(["mild", "nonspicy"])
    # area
    area = str(row.get("area", "")).strip().lower()
    if area:
        parts.extend(area.split())
    return " ".join(parts)


def _build_query_soup(
    mood: str | None,
    dietary: str | None,
    spicy: bool | None,
    location: str | None,
) -> str:
    """Build a TF-IDF query soup from user inputs."""
    parts: list[str] = []
    if mood:
        parts.extend([mood.strip().lower()] * 2)
    if dietary:
        d = dietary.strip().lower()
        parts.append(d)
        parts.append(d.replace("-", "").replace(" ", ""))
    if spicy is True:
        parts.extend(["spicy", "hot", "masaledar"])
    elif spicy is False:
        parts.extend(["mild", "nonspicy"])
    if location:
        parts.extend(location.strip().lower().split())
    return " ".join(parts) if parts else "food"


def _human_reason(
    mood: str | None,
    dietary: str | None,
    spicy: bool | None,
    location: str | None,
    budget: int | float | None,
    rating: float,
    delivery_minutes: int,
    cuisine: str,
    is_boosted: bool,
    sim_norm: float,
    quality_score: float,
) -> str:
    """Generate a rich, preference-aware explanation sentence for each recommendation.

    Mentions every user preference that was specified, then adds objective dish
    qualities (rating, delivery speed) to explain *why* this dish ranked highly.
    """
    # ── Preference clauses (what the user asked for) ────────────────────────
    pref_parts: list[str] = []
    if mood:
        pref_parts.append(f"a {mood}-food mood")
    if dietary:
        pref_parts.append(f"{dietary.lower()} options")
    if spicy is True:
        pref_parts.append("spicy food")
    elif spicy is False:
        pref_parts.append("mild (non-spicy) food")
    if budget is not None:
        pref_parts.append(f"a budget under \u20b9{int(budget)}")
    if location:
        pref_parts.append(f"food near {location}")

    # ── Quality clauses (why this dish scores highly) ────────────────────────
    quality_parts: list[str] = []
    if is_boosted:
        quality_parts.append(f"you\u2019ve liked {cuisine} before")
    if rating >= 4.5:
        quality_parts.append(f"it\u2019s top-rated (\u2605{rating:.1f})")
    elif rating >= 4.2:
        quality_parts.append(f"it\u2019s highly rated (\u2605{rating:.1f})")
    if delivery_minutes <= 20:
        quality_parts.append(f"fast delivery in just {delivery_minutes}\u202fmin")
    elif delivery_minutes <= 35 and not quality_parts:
        quality_parts.append(f"delivers in {delivery_minutes}\u202fmin")
    # fallback quality signal
    if not quality_parts and quality_score >= 0.7:
        quality_parts.append("it\u2019s a top pick in Gwalior")
    elif not quality_parts:
        quality_parts.append("it\u2019s a popular choice in Gwalior")

    # ── Assemble the sentence ────────────────────────────────────────────────
    if pref_parts:
        # "Recommended because you selected spicy food and a budget under ₹250 — ..."
        pref_str = " and ".join(
            [", ".join(pref_parts[:-1]), pref_parts[-1]] if len(pref_parts) > 1
            else pref_parts
        )
        quality_str = ", ".join(quality_parts[:2])
        return f"Recommended because you selected {pref_str} \u2014 {quality_str}."
    else:
        # No explicit preferences — lead with quality
        quality_str = " and ".join(
            [", ".join(quality_parts[:-1]), quality_parts[-1]] if len(quality_parts) > 1
            else quality_parts
        )
        return f"Recommended because {quality_str}."


# ---------------------------------------------------------------------------
# Module-level state — loaded once at import time
# ---------------------------------------------------------------------------

_df: pd.DataFrame | None = None
_tfidf: TfidfVectorizer | None = None
_tfidf_matrix = None       # sparse matrix
_quality_scores: np.ndarray | None = None


def _load() -> None:
    """Load dataset, fit TF-IDF, and compute quality scores (once at startup)."""
    global _df, _tfidf, _tfidf_matrix, _quality_scores

    if not DATA_PATH.is_file():
        raise RuntimeError(
            f"Dataset not found at {DATA_PATH}. "
            "Place 'gwalior_food_dataset_330_unique.csv' inside the 'data/' folder."
        )

    df = pd.read_csv(DATA_PATH)

    # --- Validate required columns ---
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise RuntimeError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

    # --- Normalise string columns ---
    for col in ("restaurant", "dish", "cuisine", "city", "area", "dietary"):
        df[col] = df[col].fillna("").astype(str).str.strip()

    # --- Normalise numeric columns ---
    for col in ("price", "rating", "distance_km", "delivery_minutes"):
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["spicy"] = df["spicy"].apply(_to_bool)
    df["mood_list"] = df["mood_tags"].apply(_split_tags)
    df["meal_list"] = df["meal_type"].apply(_split_tags)

    # --- Build content soup for TF-IDF ---
    df["content_soup"] = df.apply(_build_soup, axis=1)

    # --- Fit TF-IDF ---
    tfidf = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        stop_words=None,   # keep food domain words like "non"
    )
    tfidf_matrix = tfidf.fit_transform(df["content_soup"])

    # --- Compute base quality score (0-1) ---
    # rating: higher → better (scale 0-5 → 0-1)
    # delivery_minutes: lower → better (invert after scaling)
    scaler = MinMaxScaler()
    quality_raw = df[["rating", "delivery_minutes"]].copy()
    scaled = scaler.fit_transform(quality_raw)
    rating_norm = scaled[:, 0]
    delivery_norm = 1.0 - scaled[:, 1]     # invert: faster = higher score
    quality_scores = 0.6 * rating_norm + 0.4 * delivery_norm

    _df = df.reset_index(drop=True)
    _tfidf = tfidf
    _tfidf_matrix = tfidf_matrix
    _quality_scores = quality_scores


# Load on import
_load()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def recommend(
    mood: str | None = None,
    budget: int | float | None = None,
    location: str | None = None,
    dietary: str | None = None,
    spicy: bool | None = None,
    user_liked_ids: list[int] | None = None,
    top_n: int = 5,
) -> list[dict[str, Any]]:
    """Return up to top_n food recommendations as a list of dicts.

    Hard filters (applied in order, each with a graceful fallback):
        city     → always "Gwalior"
        dietary  → Vegetarian / Non-Vegetarian
        spicy    → True (spicy only) / False (mild only) / None (both)
        mood     → must appear in dish mood_tags
        budget   → price <= budget
        location → area matches (relaxed if too few results)

    TF-IDF cosine similarity then ranks *within* the filtered pool to prefer
    matching cuisine, meal_type, and area nuances.

    Returns
    -------
    list of dicts with keys:
        id, dish, restaurant, cuisine, price, rating, area, dietary, spicy,
        delivery_minutes, distance_km, image_url, match_percent, reason.
    """
    assert _df is not None, "Recommender not loaded — call _load() first."

    top_n = max(1, min(top_n, 10))
    MIN_POOL = max(top_n, 8)   # minimum candidates before we relax a filter

    # ------------------------------------------------------------------ #
    # 1. Start with all Gwalior dishes                                    #
    # ------------------------------------------------------------------ #
    base = _df[_df["city"].str.casefold() == "gwalior"].copy()

    # ------------------------------------------------------------------ #
    # 2. Hard filter: dietary (strict — user said veg/non-veg explicitly) #
    # ------------------------------------------------------------------ #
    if dietary:
        d_norm = dietary.strip().casefold()
        pool = base[base["dietary"].str.casefold() == d_norm]
        base = pool if not pool.empty else base   # fallback: ignore dietary

    # ------------------------------------------------------------------ #
    # 3. Hard filter: spicy / mild                                        #
    # ------------------------------------------------------------------ #
    if spicy is not None:
        pool = base[base["spicy"] == bool(spicy)]
        base = pool if not pool.empty else base   # fallback: ignore spicy

    # ------------------------------------------------------------------ #
    # 4. Hard filter: mood tag (dish must advertise this mood)            #
    # ------------------------------------------------------------------ #
    if mood:
        mood_norm = mood.strip().lower()
        pool = base[base["mood_list"].apply(lambda tags: mood_norm in tags)]
        if len(pool) >= top_n:
            base = pool
        elif len(pool) > 0:
            base = pool   # use even a small set — better than ignoring mood

    # ------------------------------------------------------------------ #
    # 5. Hard filter: budget                                              #
    # ------------------------------------------------------------------ #
    if budget is not None:
        try:
            pool = base[base["price"] <= float(budget)]
            base = pool if not pool.empty else base   # fallback: ignore budget
        except (ValueError, TypeError):
            pass

    # ------------------------------------------------------------------ #
    # 6. Soft filter: location / area                                     #
    # Only apply if enough dishes remain after mood+spicy+dietary.        #
    # ------------------------------------------------------------------ #
    if location:
        loc_norm = location.strip().casefold()
        pool = base[base["area"].str.casefold() == loc_norm]
        if len(pool) >= top_n:
            base = pool
        # else: location becomes a TF-IDF preference, not a hard cut

    if base.empty:
        # Ultimate fallback — return top-rated Gwalior dishes
        base = _df[_df["city"].str.casefold() == "gwalior"].copy()

    # ------------------------------------------------------------------ #
    # 7. TF-IDF cosine similarity — rank within the filtered pool         #
    # The query emphasises cuisine/meal_type/area preference signals.     #
    # ------------------------------------------------------------------ #
    query_soup = _build_query_soup(mood, dietary, spicy, location)
    query_vec = _tfidf.transform([query_soup])

    cand_indices = base.index.tolist()
    cand_matrix = _tfidf_matrix[cand_indices]
    sim_scores = cosine_similarity(query_vec, cand_matrix).flatten()

    # Normalise sim scores within this pool so they spread across [0, 1]
    sim_min, sim_max = sim_scores.min(), sim_scores.max()
    if sim_max > sim_min:
        sim_scores_norm = (sim_scores - sim_min) / (sim_max - sim_min)
    else:
        sim_scores_norm = sim_scores   # all equal — quality will decide

    # ------------------------------------------------------------------ #
    # 8. Quality scores                                                   #
    # ------------------------------------------------------------------ #
    quality = _quality_scores[cand_indices]

    # ------------------------------------------------------------------ #
    # 9. Personal boost (cuisine from liked dishes)                       #
    # ------------------------------------------------------------------ #
    liked_cuisines: set[str] = set()
    if user_liked_ids:
        liked_rows = _df[_df["id"].isin(user_liked_ids)]
        liked_cuisines = set(liked_rows["cuisine"].str.casefold().unique())

    boost = np.zeros(len(cand_indices))
    if liked_cuisines:
        for i, idx in enumerate(cand_indices):
            if _df.loc[idx, "cuisine"].casefold() in liked_cuisines:
                boost[i] = 1.0

    # ------------------------------------------------------------------ #
    # 10. Combined final score                                            #
    # ------------------------------------------------------------------ #
    final_scores = W_SIM * sim_scores_norm + W_QUALITY * quality + W_BOOST * boost

    # ------------------------------------------------------------------ #
    # 11. Shuffle tie-breaks to avoid returning the exact same order      #
    #     when multiple dishes share identical final scores.              #
    # ------------------------------------------------------------------ #
    noise = np.random.default_rng().uniform(0, 0.005, len(final_scores))
    final_scores = final_scores + noise

    top_local_indices = np.argsort(final_scores)[::-1][:top_n]

    # ------------------------------------------------------------------ #
    # 12. Build output dicts                                              #
    # ------------------------------------------------------------------ #
    results: list[dict[str, Any]] = []
    for local_i in top_local_indices:
        global_idx = cand_indices[local_i]
        row = _df.loc[global_idx]
        is_boosted = bool(boost[local_i] > 0)
        # Express match as percentage of raw final score, scaled nicely
        match_pct = int(round(final_scores[local_i] * 100))
        match_pct = max(1, min(match_pct, 99))

        results.append({
            "id": int(row["id"]),
            "dish": row["dish"],
            "restaurant": row["restaurant"],
            "cuisine": row["cuisine"],
            "price": float(row["price"]),
            "rating": float(row["rating"]),
            "area": row["area"],
            "dietary": row["dietary"],
            "spicy": bool(row["spicy"]),
            "delivery_minutes": int(row["delivery_minutes"]),
            "distance_km": float(row["distance_km"]),
            "image_url": str(row.get("image_url", "") or ""),
            "match_percent": match_pct,
            "reason": _human_reason(
                mood=mood,
                dietary=dietary,
                spicy=spicy,
                location=location,
                budget=budget,
                rating=float(row["rating"]),
                delivery_minutes=int(row["delivery_minutes"]),
                cuisine=row["cuisine"],
                is_boosted=is_boosted,
                sim_norm=float(sim_scores_norm[local_i]),
                quality_score=float(quality[local_i]),
            ),
        })

    return results


# ---------------------------------------------------------------------------
# Convenience accessors (used by chatbot for LLM system prompt context)
# ---------------------------------------------------------------------------

def get_cuisines() -> list[str]:
    """Sorted list of unique cuisine types in the dataset."""
    assert _df is not None
    return sorted(c for c in _df["cuisine"].unique() if c)


def get_areas() -> list[str]:
    """Sorted list of unique areas in the Gwalior dataset."""
    assert _df is not None
    return sorted(a for a in _df["area"].unique() if a)


# ---------------------------------------------------------------------------
# Category map — maps canonical category keys to display info + keywords
# ---------------------------------------------------------------------------

CATEGORY_MAP: dict[str, dict[str, Any]] = {
    "chinese":      {"label": "Chinese",       "emoji": "🍜", "keywords": ["chinese", "noodles", "fried rice", "manchurian", "chow mein"]},
    "italian":      {"label": "Italian",        "emoji": "🍝", "keywords": ["italian", "pasta", "pizza"]},
    "north indian": {"label": "North Indian",   "emoji": "🥘", "keywords": ["north indian", "punjabi", "mughlai"]},
    "south indian": {"label": "South Indian",   "emoji": "🥥", "keywords": ["south indian", "idli", "dosa", "kerala"]},
    "mughlai":      {"label": "Mughlai",        "emoji": "🍖", "keywords": ["mughlai", "awadhi"]},
    "street food":  {"label": "Street Food",    "emoji": "🌮", "keywords": ["street food", "chaat", "golgappa", "pani puri"]},
    "fast food":    {"label": "Fast Food",      "emoji": "🍔", "keywords": ["fast food", "burger", "sandwich", "wrap"]},
    "dessert":      {"label": "Desserts",       "emoji": "🍮", "keywords": ["dessert", "sweet", "mithai", "gulab jamun", "ice cream", "kheer", "halwa", "barfi", "ladoo"]},
    "snacks":       {"label": "Snacks",         "emoji": "🥨", "keywords": ["snack", "snacks", "namkeen", "bhujia", "samosa"]},
    "biryani":      {"label": "Biryani",        "emoji": "🍚", "keywords": ["biryani", "pulao"]},
    "pizza":        {"label": "Pizza",          "emoji": "🍕", "keywords": ["pizza"]},
    "burger":       {"label": "Burgers",        "emoji": "🍔", "keywords": ["burger"]},
    "continental":  {"label": "Continental",    "emoji": "🥗", "keywords": ["continental"]},
    "rajasthani":   {"label": "Rajasthani",     "emoji": "🫕", "keywords": ["rajasthani", "dal baati"]},
    "tandoor":      {"label": "Tandoor",        "emoji": "🔥", "keywords": ["tandoor", "tandoori"]},
}


def resolve_category(text: str) -> tuple[str, str, str] | None:
    """Return (key, label, emoji) if text matches a known category, else None."""
    t = text.strip().casefold()
    for key, info in CATEGORY_MAP.items():
        for kw in info["keywords"]:
            if kw in t:
                return key, info["label"], info["emoji"]
    return None


def recommend_by_category(
    category_key: str,
    top_n: int = 5,
    user_liked_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    """Return top_n dishes that match the given category key.

    Matching strategy (in priority order):
      1. Exact cuisine column match (case-insensitive).
      2. Partial cuisine column match.
      3. Keyword match in dish name, meal_type, or mood_tags.
      4. TF-IDF similarity fallback using the category name as query.
    """
    assert _df is not None, "Recommender not loaded."

    top_n = max(1, min(top_n, 10))
    info = CATEGORY_MAP.get(category_key)
    keywords: list[str] = info["keywords"] if info else [category_key]
    label: str = info["label"] if info else category_key.title()

    gwalior = _df[_df["city"].str.casefold() == "gwalior"].copy()

    def _keyword_mask(df: pd.DataFrame) -> pd.Series:
        mask = pd.Series(False, index=df.index)
        for kw in keywords:
            mask |= (
                df["cuisine"].str.casefold().str.contains(kw, na=False)
                | df["dish"].str.casefold().str.contains(kw, na=False)
                | df["meal_type"].str.casefold().str.contains(kw, na=False)
                | df["mood_tags"].str.casefold().str.contains(kw, na=False)
            )
        return mask

    pool = gwalior[_keyword_mask(gwalior)]

    # Fallback: TF-IDF similarity with category label as query
    if pool.empty:
        pool = gwalior

    cand_indices = pool.index.tolist()
    query_soup = " ".join(keywords)
    query_vec = _tfidf.transform([query_soup])
    cand_matrix = _tfidf_matrix[cand_indices]
    sim_scores = cosine_similarity(query_vec, cand_matrix).flatten()

    # Normalise
    s_min, s_max = sim_scores.min(), sim_scores.max()
    if s_max > s_min:
        sim_norm = (sim_scores - s_min) / (s_max - s_min)
    else:
        sim_norm = sim_scores

    quality = _quality_scores[cand_indices]

    # Personal boost
    liked_cuisines: set[str] = set()
    if user_liked_ids:
        liked_rows = _df[_df["id"].isin(user_liked_ids)]
        liked_cuisines = set(liked_rows["cuisine"].str.casefold().unique())

    boost = np.zeros(len(cand_indices))
    if liked_cuisines:
        for i, idx in enumerate(cand_indices):
            if _df.loc[idx, "cuisine"].casefold() in liked_cuisines:
                boost[i] = 1.0

    final_scores = W_SIM * sim_norm + W_QUALITY * quality + W_BOOST * boost
    noise = np.random.default_rng().uniform(0, 0.005, len(final_scores))
    final_scores = final_scores + noise

    top_local = np.argsort(final_scores)[::-1][:top_n]

    results: list[dict[str, Any]] = []
    for local_i in top_local:
        global_idx = cand_indices[local_i]
        row = _df.loc[global_idx]
        is_boosted = bool(boost[local_i] > 0)
        match_pct = max(1, min(int(round(final_scores[local_i] * 100)), 99))
        results.append({
            "id": int(row["id"]),
            "dish": row["dish"],
            "restaurant": row["restaurant"],
            "cuisine": row["cuisine"],
            "price": float(row["price"]),
            "rating": float(row["rating"]),
            "area": row["area"],
            "dietary": row["dietary"],
            "spicy": bool(row["spicy"]),
            "delivery_minutes": int(row["delivery_minutes"]),
            "distance_km": float(row["distance_km"]),
            "image_url": str(row.get("image_url", "") or ""),
            "match_percent": match_pct,
            "reason": _human_reason(
                mood=None,
                dietary=None,
                spicy=None,
                location=None,
                budget=None,
                rating=float(row["rating"]),
                delivery_minutes=int(row["delivery_minutes"]),
                cuisine=row["cuisine"],
                is_boosted=is_boosted,
                sim_norm=float(sim_norm[local_i]),
                quality_score=float(quality[local_i]),
            ),
        })

    return results

