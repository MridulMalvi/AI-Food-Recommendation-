"""Dataset filtering and trained-model ranking for Gwalior food recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd


MODEL_FEATURES = [
    "price", "rating", "distance_km", "delivery_minutes",
    "cuisine", "dietary", "spicy", "area",
]
REQUIRED_COLUMNS = {
    "id", "restaurant", "dish", "cuisine", "price", "rating", "city", "area",
    "dietary", "spicy", "mood_tags", "meal_type", "distance_km",
    "delivery_minutes", "image_url",
}


class FoodRecommender:
    """Applies user constraints first, then scores the remaining rows with the supplied model."""

    def __init__(self, data_path: Path, model_path: Path) -> None:
        if not data_path.is_file():
            raise RuntimeError(f"Dataset not found. Expected it at: {data_path}")
        if not model_path.is_file():
            raise RuntimeError(
                "Trained model not found. Place it at "
                "backend/model/gwalior_food_recommendation_model.pkl"
            )

        self.data = pd.read_csv(data_path)
        missing = REQUIRED_COLUMNS.difference(self.data.columns)
        if missing:
            raise RuntimeError(f"Dataset is missing required columns: {', '.join(sorted(missing))}")

        # Keep the original display values and add normalized values only for matching.
        for column in ("restaurant", "dish", "cuisine", "city", "area", "dietary"):
            self.data[column] = self.data[column].fillna("").astype(str).str.strip()
        for column in ("price", "rating", "distance_km", "delivery_minutes"):
            self.data[column] = pd.to_numeric(self.data[column], errors="coerce").fillna(0)
        self.data["spicy"] = self.data["spicy"].map(self._to_bool)
        self.data["mood_list"] = self.data["mood_tags"].apply(self._split_tags)
        self.data["meal_list"] = self.data["meal_type"].apply(self._split_tags)
        
        # Compatibility shim for pipelines serialized with older scikit-learn versions
        try:
            import sklearn.compose._column_transformer as _ct
            if not hasattr(_ct, "_RemainderColsList"):
                _ct._RemainderColsList = type("_RemainderColsList", (list,), {})
        except Exception:
            pass

        self.model = joblib.load(model_path)  # The model is loaded once during application startup.

    @staticmethod
    def _to_bool(value: Any) -> bool:
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    @staticmethod
    def _split_tags(value: Any) -> list[str]:
        raw = str(value or "").strip("[] ")
        return [item.strip().strip("'\"").lower() for item in raw.split(",") if item.strip().strip("'\"")]

    @property
    def cuisines(self) -> list[str]:
        return sorted(item for item in self.data["cuisine"].unique() if item)

    @property
    def areas(self) -> list[str]:
        return sorted(item for item in self.data["area"].unique() if item)

    def filter_candidates(self, preferences: dict[str, Any], ignored: set[str] | None = None) -> pd.DataFrame:
        """Hard-filter records. `ignored` is only used by the explicit fallback sequence."""
        ignored = ignored or set()
        candidates = self.data[self.data["city"].str.casefold() == "gwalior"].copy()

        budget = preferences.get("budget")
        if budget is not None and "budget" not in ignored:
            candidates = candidates[candidates["price"] <= float(budget)]
        dietary = preferences.get("dietary")
        if dietary and "dietary" not in ignored:
            candidates = candidates[candidates["dietary"].str.casefold() == str(dietary).casefold()]
        if preferences.get("spicy") is not None and "spicy" not in ignored:
            candidates = candidates[candidates["spicy"] == bool(preferences["spicy"])]
        for key, column in (("cuisine", "cuisine"), ("area", "area")):
            value = preferences.get(key)
            if value and key not in ignored:
                candidates = candidates[candidates[column].str.casefold() == str(value).casefold()]
        for key, column in (("mood", "mood_list"), ("meal_type", "meal_list")):
            value = preferences.get(key)
            if value and key not in ignored:
                normalized = str(value).strip().lower()
                candidates = candidates[candidates[column].apply(lambda tags: normalized in tags)]
        return candidates

    def rank_candidates(self, candidates: pd.DataFrame) -> pd.DataFrame:
        """Score candidates with the exact eight feature columns used to train the supplied pipeline."""
        if candidates.empty:
            return candidates
        feature_frame = candidates.loc[:, MODEL_FEATURES].copy()
        feature_frame["spicy"] = feature_frame["spicy"].astype(bool)
        try:
            scores = self.model.predict(feature_frame)
        except Exception as exc:  # Give a useful error for an incompatible uploaded model.
            raise RuntimeError(f"The trained model could not score the dataset: {exc}") from exc
        ranked = candidates.copy()
        # quality_score is expected to be a 0-1 target; clipping keeps the public match indicator stable.
        ranked["recommendation_score"] = np.clip(np.asarray(scores, dtype=float), 0, 1)
        return ranked.sort_values("recommendation_score", ascending=False, kind="stable")

    def recommend(self, preferences: dict[str, Any], top_n: int = 5) -> tuple[list[dict[str, Any]], str | None]:
        """Return up to five records, progressively relaxing only when there is no exact result."""
        candidates = self.filter_candidates(preferences)
        note: str | None = None
        if candidates.empty:
            relaxations = [
                ({"area"}, "I couldn't find an exact match, so I relaxed the area preference."),
                ({"area", "mood", "meal_type"}, "I couldn't find an exact match, so I relaxed area and occasion preferences."),
                ({"area", "mood", "meal_type", "cuisine"}, "I couldn't find an exact match, so I also relaxed the cuisine preference."),
                ({"area", "mood", "meal_type", "cuisine", "spicy"}, "I couldn't find an exact match, so I relaxed the spice preference."),
                ({"area", "mood", "meal_type", "cuisine", "spicy", "dietary"}, "I couldn't find an exact match, so these are the closest Gwalior options."),
                ({"area", "mood", "meal_type", "cuisine", "spicy", "dietary", "budget"}, "No option met every constraint, so these are the closest Gwalior options."),
            ]
            for ignored, message in relaxations:
                candidates = self.filter_candidates(preferences, ignored)
                if not candidates.empty:
                    note = message
                    break

        ranked = self.rank_candidates(candidates).head(max(1, min(top_n, 5)))
        records: list[dict[str, Any]] = []
        for _, row in ranked.iterrows():
            records.append({
                "id": int(row["id"]), "restaurant": row["restaurant"], "dish": row["dish"],
                "cuisine": row["cuisine"], "price": float(row["price"]), "rating": float(row["rating"]),
                "city": row["city"], "area": row["area"], "dietary": row["dietary"],
                "spicy": bool(row["spicy"]), "mood_tags": list(row["mood_list"]),
                "meal_type": list(row["meal_list"]), "distance_km": float(row["distance_km"]),
                "delivery_minutes": int(row["delivery_minutes"]), "image_url": str(row["image_url"] or ""),
                "recommendation_score": round(float(row["recommendation_score"]), 4),
            })
        return records, note
