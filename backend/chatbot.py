"""LangChain Agent with OpenRouter tool-calling and offline fallback for Gwalior Food Recommendations."""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

PREFERENCE_KEYS = ("mood", "budget", "location", "dietary", "spicy")


class FoodRecommendationInput(BaseModel):
    mood: Optional[str] = Field(
        default=None,
        description=(
            "User's current mood or craving vibe. "
            "Examples: happy, tired, comfort, social, sad, celebration, quick."
        ),
    )
    budget: Optional[int] = Field(
        default=None,
        description="Maximum price per dish in INR (₹). E.g. 200, 300, 500.",
    )
    location: Optional[str] = Field(
        default=None,
        description=(
            "Neighbourhood or area in Gwalior where the user wants to eat. "
            "E.g. City Centre, Lashkar, Morar, Thatipur, Phoolbagh, Hazira, "
            "Naya Bazar, Jayendraganj, Padav, Fort Road."
        ),
    )
    dietary: Optional[str] = Field(
        default=None,
        description="Dietary preference: exactly 'Vegetarian' or 'Non-Vegetarian'.",
    )
    spicy: Optional[bool] = Field(
        default=None,
        description=(
            "True for spicy/hot/masaledar food, False for mild/non-spicy, "
            "None if unspecified."
        ),
    )


class FoodChatbot:
    """LangChain-powered conversational agent for Gwalior food discovery."""

    def __init__(
        self,
        cuisines: list[str] | None = None,
        areas: list[str] | None = None,
        recommender: Any = None,
    ) -> None:
        self.cuisines = cuisines or []
        self.areas = areas or []
        self.recommender = recommender  # kept for interface compatibility
        self.api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        self.model_name = os.getenv("OPENROUTER_MODEL", "mistralai/mistral-7b-instruct:free")
        self.base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._llm = None
        self._init_llm()

    # ------------------------------------------------------------------
    # LLM initialisation
    # ------------------------------------------------------------------

    def _init_llm(self) -> None:
        if not self.api_key:
            logger.info("OPENROUTER_API_KEY not set — using rule-based fallback only.")
            return
        try:
            from langchain_openai import ChatOpenAI

            self._llm = ChatOpenAI(
                model=self.model_name,
                openai_api_key=self.api_key,
                openai_api_base=self.base_url,
                temperature=0.4,
                timeout=20,
                max_retries=2,
                default_headers={
                    "HTTP-Referer": "http://localhost:8000",
                    "X-Title": "Gwalior Food Recommender",
                },
            )
            logger.info("LangChain ChatOpenAI initialised with model: %s", self.model_name)
        except Exception as exc:
            logger.warning("Could not initialise LangChain ChatOpenAI: %s", exc)
            self._llm = None

    # ------------------------------------------------------------------
    # Preference helpers
    # ------------------------------------------------------------------

    @staticmethod
    def empty_preferences() -> dict[str, Any]:
        return {key: None for key in PREFERENCE_KEYS}

    def _normalise_dietary(self, value: str | None) -> str | None:
        if not value:
            return None
        v = value.casefold()
        if "non" in v or "chicken" in v or "mutton" in v or "meat" in v or "fish" in v:
            return "Non-Vegetarian"
        if "veg" in v:
            return "Vegetarian"
        return None

    def _normalise_area(self, value: str | None) -> str | None:
        if not value or not self.areas:
            return value
        target = value.casefold()
        exact = next((a for a in self.areas if a.casefold() == target), None)
        if exact:
            return exact
        partial = next(
            (a for a in self.areas if target in a.casefold() or a.casefold() in target),
            None,
        )
        return partial  # None if no match — will be ignored gracefully

    def extract_preferences(self, message: str) -> dict[str, Any]:
        """Rule-based preference extraction for the offline fallback path."""
        text = message.casefold()
        prefs = self.empty_preferences()

        # Budget
        m = re.search(
            r"(?:under|below|less\s+than|within|budget(?:\s+of)?|₹|rs\.?|inr)\s*₹?\s*(\d{2,5})",
            text,
        )
        if not m:
            m = re.search(r"₹\s*(\d{2,5})", text)
        if m:
            prefs["budget"] = int(m.group(1))

        # Dietary
        if re.search(r"\b(non[- ]?veg(?:etarian)?|chicken|mutton|meat|fish|egg)\b", text):
            prefs["dietary"] = "Non-Vegetarian"
        elif re.search(r"\b(vegetarian|veg|vegan|shakahari)\b", text):
            prefs["dietary"] = "Vegetarian"

        # Spicy
        if re.search(r"\b(not\s+spicy|non[- ]?spicy|mild|less\s+spicy|without\s+spice|feeka)\b", text):
            prefs["spicy"] = False
        elif re.search(r"\b(spicy|hot|masaledar|chatpata|teekha)\b", text):
            prefs["spicy"] = True

        # Area / location
        for area in self.areas:
            if area.casefold() in text:
                prefs["location"] = area
                break

        # Mood
        mood_map = {
            "comfort": "comfort", "tired": "tired", "happy": "happy",
            "social": "social", "sad": "sad", "celebrat": "celebration",
            "party": "celebration", "quick": "quick", "hungry": "comfort",
        }
        for word, mood in mood_map.items():
            if word in text:
                prefs["mood"] = mood
                break

        return prefs

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_category(message: str) -> tuple[str, str, str] | None:
        """Return (key, label, emoji) if the message targets a specific food category."""
        import backend.recommender as rec_module
        return rec_module.resolve_category(message)

    def chat_with_agent(
        self,
        message: str,
        history: list[dict[str, str]] | None = None,
        user_liked_ids: list[int] | None = None,
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any], dict[str, str] | None]:
        """
        Returns (reply_text, recommendations, preferences, category_info).

        category_info is {"key": ..., "label": ..., "emoji": ...} when a specific
        cuisine / dish-type was detected, otherwise None.

        Tries the LangChain agent first; falls back to rule-based extraction
        if no API key is configured or the agent errors.
        """
        liked_ids = user_liked_ids or []
        category_info = self._detect_category(message)

        if self._llm:
            try:
                reply, recs, prefs = self._run_langchain_agent(message, history or [], liked_ids)
                return reply, recs, prefs, category_info
            except Exception as exc:
                logger.warning("LangChain agent error — falling back: %s", exc)

        reply, recs, prefs = self._run_local_fallback(message, history or [], liked_ids)
        return reply, recs, prefs, category_info

    # ------------------------------------------------------------------
    # LangChain agent path
    # ------------------------------------------------------------------

    def _run_langchain_agent(
        self,
        message: str,
        history: list[dict[str, str]],
        user_liked_ids: list[int],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
        from langchain_core.tools import tool

        # Import the functional recommend API from the new recommender
        import backend.recommender as rec_module

        captured: dict[str, Any] = {
            "recommendations": [],
            "preferences": self.empty_preferences(),
            "tool_called": False,
        }

        # ---- Define the LangChain tool --------------------------------
        @tool("get_food_recommendations", args_schema=FoodRecommendationInput)
        def get_food_recommendations(
            mood: Optional[str] = None,
            budget: Optional[int] = None,
            location: Optional[str] = None,
            dietary: Optional[str] = None,
            spicy: Optional[bool] = None,
        ) -> str:
            """
            Call this tool whenever the user asks for food suggestions, restaurants,
            dish recommendations, or dining options in Gwalior.
            Extract the user's mood, budget (INR), location (area in Gwalior),
            dietary preference, and spicy preference, then call this tool.
            """
            # Normalise inputs
            norm_dietary = self._normalise_dietary(dietary)
            norm_location = self._normalise_area(location)

            preferences = {
                "mood": mood,
                "budget": budget,
                "location": norm_location,
                "dietary": norm_dietary,
                "spicy": spicy,
            }

            results = rec_module.recommend(
                mood=mood,
                budget=budget,
                location=norm_location,
                dietary=norm_dietary,
                spicy=spicy,
                user_liked_ids=user_liked_ids,
                top_n=5,
            )

            captured["recommendations"] = results
            captured["preferences"] = preferences
            captured["tool_called"] = True

            if not results:
                return (
                    "No dishes found in Gwalior for these constraints. "
                    "Suggest adjusting budget or dietary filter."
                )

            lines = []
            for r in results:
                spicy_label = "🌶 spicy" if r["spicy"] else "mild"
                lines.append(
                    f"- {r['dish']} at {r['restaurant']} "
                    f"(₹{r['price']:.0f}, ★{r['rating']:.1f}, {r['area']}, "
                    f"{r['cuisine']}, {spicy_label}, "
                    f"{r['delivery_minutes']} min, {r['match_percent']}% match)"
                )

            summary = "Recommendations found:\n" + "\n".join(lines)
            if user_liked_ids:
                summary += f"\n[Personalised for {len(user_liked_ids)} liked dish(es)]"
            return summary

        # ---- Build message list --------------------------------------
        system_prompt = (
            "You are 'Gwalior Foodie' — an enthusiastic, friendly local food AI for "
            "Gwalior, Madhya Pradesh, India.\n"
            "Rules:\n"
            "1. For greetings or casual questions ('Hi', 'Who are you?'), reply warmly "
            "without calling any tool.\n"
            "2. For ANY food-related intent (cravings, restaurants, budget meals, "
            "cuisines, dietary needs, moods), ALWAYS call `get_food_recommendations` "
            "with the extracted parameters.\n"
            "3. After the tool returns, give a warm, mouth-watering 2-3 sentence "
            "response highlighting the top dish. Mention prices with '₹'.\n"
            "4. Be concise, engaging, and friendly."
        )

        msgs: list[Any] = [SystemMessage(content=system_prompt)]
        for item in history[-6:]:
            role = item.get("role")
            content = item.get("content", "")
            if role == "user":
                msgs.append(HumanMessage(content=content))
            elif role == "assistant":
                msgs.append(AIMessage(content=content))
        msgs.append(HumanMessage(content=message))

        # ---- First LLM call -----------------------------------------
        llm_with_tools = self._llm.bind_tools([get_food_recommendations])
        ai_msg = llm_with_tools.invoke(msgs)

        if hasattr(ai_msg, "tool_calls") and ai_msg.tool_calls:
            msgs.append(ai_msg)
            for tc in ai_msg.tool_calls:
                tool_result = get_food_recommendations.invoke(tc["args"])
                msgs.append(ToolMessage(content=str(tool_result), tool_call_id=tc["id"]))

            # Second call to generate the conversational summary
            final = self._llm.invoke(msgs)
            reply = str(final.content).strip()
            return reply, captured["recommendations"], captured["preferences"]

        # No tool call — conversational response only
        return str(ai_msg.content).strip(), [], self.empty_preferences()

    # ------------------------------------------------------------------
    # Local rule-based fallback path
    # ------------------------------------------------------------------

    def _run_local_fallback(
        self,
        message: str,
        history: list[dict[str, str]],
        user_liked_ids: list[int],
    ) -> tuple[str, list[dict[str, Any]], dict[str, Any]]:
        import backend.recommender as rec_module

        # Greetings → no recommendations
        casual = {"hi", "hello", "hey", "namaste", "good morning", "good evening",
                  "who are you", "what can you do", "help"}
        if message.strip().casefold() in casual:
            return (
                "Hello! I'm your Gwalior Food Guide 🍛\n"
                "Tell me what you're craving — mood, budget, area, or diet — "
                "and I'll find the best dishes for you!\n"
                "_(e.g. 'spicy vegetarian under ₹250 near City Centre')_",
                [],
                self.empty_preferences(),
            )

        # Accumulate preferences across history + current message
        prefs = self.empty_preferences()
        for entry in history:
            if entry.get("role") == "user":
                extracted = self.extract_preferences(entry.get("content", ""))
                prefs.update({k: v for k, v in extracted.items() if v is not None})

        current = self.extract_preferences(message)
        prefs.update({k: v for k, v in current.items() if v is not None})

        results = rec_module.recommend(
            mood=prefs.get("mood"),
            budget=prefs.get("budget"),
            location=prefs.get("location"),
            dietary=prefs.get("dietary"),
            spicy=prefs.get("spicy"),
            user_liked_ids=user_liked_ids,
            top_n=5,
        )

        reply = self._generate_local_reply(results, prefs)
        return reply, results, prefs

    def _generate_local_reply(
        self,
        recommendations: list[dict[str, Any]],
        preferences: dict[str, Any],
    ) -> str:
        if not recommendations:
            return (
                "I couldn't find anything matching those exact filters right now. "
                "Try a higher budget or explore other areas!"
            )

        desc_parts: list[str] = []
        if preferences.get("dietary"):
            desc_parts.append(preferences["dietary"].lower())
        if preferences.get("spicy") is True:
            desc_parts.append("spicy")
        elif preferences.get("spicy") is False:
            desc_parts.append("mild")
        if preferences.get("mood"):
            desc_parts.append(f"{preferences['mood']} mood")

        desc = " & ".join(desc_parts) if desc_parts else "delicious"
        budget_txt = f" under ₹{preferences['budget']}" if preferences.get("budget") else ""
        loc_txt = f" near {preferences['location']}" if preferences.get("location") else ""
        top = recommendations[0]

        return (
            f"Here are {len(recommendations)} great {desc} picks in Gwalior"
            f"{budget_txt}{loc_txt}! "
            f"You might love **{top['dish']}** from *{top['restaurant']}* "
            f"(₹{top['price']:.0f}, ★{top['rating']:.1f}) — {top['reason']}"
        )
