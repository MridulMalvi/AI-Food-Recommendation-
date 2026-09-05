"""Preference extraction with an optional OpenRouter LLM and an offline-safe fallback."""

from __future__ import annotations

import json
import os
import re
from typing import Any

import requests


PREFERENCE_KEYS = ("budget", "cuisine", "dietary", "spicy", "area", "mood", "meal_type")


class FoodChatbot:
    def __init__(self, cuisines: list[str] | None = None, areas: list[str] | None = None) -> None:
        self.cuisines = cuisines or []
        self.areas = areas or []
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.model = os.getenv("OPENROUTER_MODEL", "openai/gpt-4o-mini")

    @staticmethod
    def empty_preferences() -> dict[str, Any]:
        return {key: None for key in PREFERENCE_KEYS}

    def extract_preferences(self, message: str) -> dict[str, Any]:
        """Use an LLM when configured; any API failure intentionally falls back to local parsing."""
        if self.api_key:
            try:
                return self._extract_with_llm(message)
            except (requests.RequestException, ValueError, KeyError, json.JSONDecodeError):
                pass
        return self._extract_with_rules(message)

    def _extract_with_llm(self, message: str) -> dict[str, Any]:
        prompt = (
            "Extract food preferences from this message. Return ONLY a JSON object with keys "
            "budget, cuisine, dietary, spicy, area, mood, meal_type. Use null for absent values. "
            f"Allowed cuisines: {self.cuisines}. Allowed areas: {self.areas}. "
            "dietary must be Vegetarian or Non-Vegetarian; spicy must be true, false, or null. "
            f"Message: {message!r}"
        )
        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json={"model": self.model, "messages": [{"role": "user", "content": prompt}], "temperature": 0},
            timeout=12,
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content.replace("```json", "").replace("```", "").strip())
        return self._clean_preferences(parsed)

    def _extract_with_rules(self, message: str) -> dict[str, Any]:
        text = message.casefold()
        preferences = self.empty_preferences()
        budget_match = re.search(r"(?:under|below|less than|within|budget(?: of)?|₹|rs\.?|inr)\s*₹?\s*(\d{2,5})", text)
        if not budget_match:
            budget_match = re.search(r"₹\s*(\d{2,5})", text)
        if budget_match:
            preferences["budget"] = int(budget_match.group(1))
        if re.search(r"\b(non[- ]?veg(?:etarian)?|chicken|mutton|meat|egg)\b", text):
            preferences["dietary"] = "Non-Vegetarian"
        elif re.search(r"\b(vegetarian|veg|vegan)\b", text):
            preferences["dietary"] = "Vegetarian"
        if re.search(r"\b(not spicy|non[- ]?spicy|mild|less spicy|without spice)\b", text):
            preferences["spicy"] = False
        elif re.search(r"\b(spicy|hot|masaledar)\b", text):
            preferences["spicy"] = True
        for cuisine in self.cuisines:
            if cuisine.casefold() in text:
                preferences["cuisine"] = cuisine
                break
        for area in self.areas:
            if area.casefold() in text:
                preferences["area"] = area
                break
        mood_map = {"comfort": "comfort", "tired": "tired", "happy": "happy", "social": "social", "sad": "sad", "celebrat": "celebration"}
        for word, mood in mood_map.items():
            if word in text:
                preferences["mood"] = mood
                break
        for meal in ("breakfast", "lunch", "dinner", "snacks", "snack"):
            if re.search(rf"\b{meal}\b", text):
                preferences["meal_type"] = "snacks" if meal == "snack" else meal
                break
        return preferences

    def _clean_preferences(self, raw: dict[str, Any]) -> dict[str, Any]:
        preferences = self.empty_preferences()
        for key in PREFERENCE_KEYS:
            if key in raw and raw[key] not in ("", None):
                preferences[key] = raw[key]
        if preferences["budget"] is not None:
            preferences["budget"] = max(0, int(float(preferences["budget"])))
        if preferences["dietary"]:
            value = str(preferences["dietary"]).casefold()
            preferences["dietary"] = "Non-Vegetarian" if "non" in value else "Vegetarian" if "veg" in value else None
        if preferences["spicy"] is not None:
            if isinstance(preferences["spicy"], str):
                preferences["spicy"] = preferences["spicy"].strip().casefold() in {"true", "1", "yes"}
            else:
                preferences["spicy"] = bool(preferences["spicy"])
        for key, choices in (("cuisine", self.cuisines), ("area", self.areas)):
            if preferences[key]:
                preferences[key] = next((item for item in choices if item.casefold() == str(preferences[key]).casefold()), None)
        for key in ("mood", "meal_type"):
            if preferences[key]:
                preferences[key] = str(preferences[key]).strip().lower()
        return preferences

    def generate_response(self, message: str, recommendations: list[dict[str, Any]], preferences: dict[str, Any], note: str | None = None) -> str:
        if not recommendations:
            return "I couldn't find any suitable Gwalior dishes right now. Try a higher budget or fewer preferences."
        phrases = []
        if preferences.get("dietary"):
            phrases.append(preferences["dietary"].lower())
        if preferences.get("spicy") is True:
            phrases.append("spicy")
        if preferences.get("cuisine"):
            phrases.append(preferences["cuisine"])
        description = " ".join(phrases) or "great"
        budget_text = f" under ₹{preferences['budget']}" if preferences.get("budget") is not None else ""
        result = f"Here are {len(recommendations)} {description} options in Gwalior{budget_text}."
        return f"{note} {result}" if note else result
