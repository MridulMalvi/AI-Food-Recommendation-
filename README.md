# 🍛 Gwalior Foodie — AI Food Recommendation System

> A conversational AI-powered food recommendation web application for Gwalior, Madhya Pradesh, India. Discover the best local dishes based on your mood, budget, location, and dietary preferences — with personalization that learns from what you like.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Technology Stack](#-technology-stack)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Dataset](#-dataset)
- [Recommendation Algorithm](#-recommendation-algorithm)
- [Personalization](#-personalization)
- [Project Structure](#-project-structure)
- [Known Limitations & Future Improvements](#-known-limitations--future-improvements)
- [Submission](#-submission)

---

## 🎯 Project Overview

Gwalior Foodie is a full-stack AI application that combines **Natural Language Understanding**, **Content-Based Filtering**, and **Personalization** to recommend the most relevant local food dishes. Users interact via a chat interface — asking things like _"I'm tired, want spicy food under ₹300 near City Centre"_ — and receive structured recommendation cards with match scores, AI-generated reasons, and one-click "Like" functionality.

**Key Features:**
- 💬 Conversational chat interface powered by a LangChain Agent
- 🤖 TF-IDF + Cosine Similarity content-based recommendation engine
- ⭐ Quality scoring via MinMaxScaler on rating and delivery time
- ❤️ Like/Favorite system with real-time personalization boosts
- 📱 Fully responsive HTML/CSS/JS frontend — no framework required
- 🔌 Graceful offline fallback (works without an API key)

---

## 🛠 Technology Stack

| Layer | Technology |
|---|---|
| **Backend Framework** | FastAPI (Python 3.10+) |
| **Conversational AI** | LangChain + LangChain-OpenAI + OpenRouter API |
| **LLM Model** | Mistral 7B Instruct (free tier via OpenRouter) |
| **Recommendation Engine** | Scikit-learn: TF-IDF Vectorizer + Cosine Similarity + MinMaxScaler |
| **Data Processing** | Pandas, NumPy |
| **Frontend** | Vanilla HTML5, CSS3, JavaScript (ES2020) |
| **Font** | Google Fonts: DM Sans, Playfair Display |
| **Environment** | python-dotenv |
| **Server** | Uvicorn (ASGI) |

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.10 or higher
- `pip` package manager

### 1. Clone the repository
```bash
git clone <your-repo-url>
cd AI-Food-Recommendation
```

### 2. Create and activate a virtual environment
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python -m venv venv
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:
```env
OPENROUTER_API_KEY=sk-or-your-key-here
OPENROUTER_MODEL=mistralai/mistral-7b-instruct:free
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

> **Note:** The application works without an API key using a rule-based fallback that still provides full TF-IDF recommendations. Get a free key at [openrouter.ai](https://openrouter.ai).

---

## 🚀 Running the Application

```bash
uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

Then open **http://127.0.0.1:8000** in your browser.

The frontend is served automatically from the `frontend/` directory.

### API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/api/status` | Health check |
| `POST` | `/api/chat` | Send a chat message, receive AI reply + recommendations |
| `POST` | `/api/favorite` | Like or unlike a dish by ID |
| `GET`  | `/api/favorites` | Fetch all currently liked dish IDs |

---

## 📊 Dataset

**File:** `data/gwalior_food_dataset_330_unique.csv`

- **330 unique food dishes** from restaurants across Gwalior city
- **15 columns:** `id`, `restaurant`, `dish`, `cuisine`, `price`, `rating`, `city`, `area`, `dietary`, `spicy`, `mood_tags`, `meal_type`, `distance_km`, `delivery_minutes`, `image_url`
- **10 cuisines:** North Indian, South Indian, Chinese, Street Food, Mughlai, Fast Food, Italian, Bakery, Desserts, Cafe
- **10 Gwalior areas:** City Centre, Lashkar, Morar, Thatipur, Phoolbagh, Hazira, Naya Bazar, Jayendraganj, Padav, Fort Road
- **Source:** Synthetically generated dataset representative of Gwalior's food scene (created for academic/assignment purposes)

---

## 🧠 Recommendation Algorithm

The system uses a **hybrid TF-IDF + Rule-Based Content Filtering** approach. No external model files are required — everything is computed at startup from the CSV.

### Step 1 — Content Soup Construction

For each dish, a text "soup" is built by concatenating:
```
mood_tags (×2 weight)  +  meal_type  +  dietary  +  cuisine  +  spicy/mild  +  area
```
Mood tags are repeated twice to upweight mood-matching in the TF-IDF space.

### Step 2 — TF-IDF Vectorization

```python
TfidfVectorizer(analyzer="word", ngram_range=(1, 2))
```
The vectorizer is fit on all 330 dish soups at startup. At query time, the user's inputs (mood, dietary, spicy, location) are converted into the same soup format and transformed into a query vector.

### Step 3 — Cosine Similarity

```python
sim_score = cosine_similarity(query_vector, dish_matrix)
```
Measures how similar each dish's content soup is to the user's query soup.

### Step 4 — Quality Score

```python
base_quality = 0.6 × norm(rating) + 0.4 × (1 - norm(delivery_minutes))
```
`MinMaxScaler` normalises both features to [0, 1]. Delivery time is inverted (faster = better).

### Step 5 — Personal Boost

```python
personal_boost = 1.0  if  dish.cuisine ∈ liked_cuisines  else  0.0
```
If the dish's cuisine matches the cuisine of any previously liked dish, it receives a boost.

### Step 6 — Final Score & Ranking

```python
final_score = 0.55 × sim_score + 0.30 × base_quality + 0.15 × personal_boost
```

Results are sorted by `final_score` descending. Each result includes a `match_percent` (1–99) and a human-readable `reason` string.

---

## ❤️ Personalization

Personalization works by storing liked dish IDs server-side (in-memory) in a `user_profiles` dictionary:

```python
user_profiles = {"default_user": {"liked_ids": [12, 45, 78]}}
```

**How it flows:**
1. User clicks the ❤️ heart button on a recommendation card.
2. The frontend sends `POST /api/favorite` with `{ item_id: 42, is_favorite: true }`.
3. The backend adds `42` to `user_profiles["default_user"]["liked_ids"]`.
4. On the next chat request, `liked_ids` is passed to `recommend()`.
5. The recommender looks up the cuisines of all liked dishes and applies a `+0.15` weight boost to any dish sharing those cuisines.
6. Results now prioritize cuisines the user has demonstrated affinity for.

The liked IDs are also stored in `localStorage` for UI state persistence across page refreshes.

---

## 📁 Project Structure

```
AI-Food-Recommendation/
├── backend/
│   ├── main.py          # FastAPI app — routes, user profiles, startup
│   ├── chatbot.py       # LangChain Agent + rule-based fallback
│   └── recommender.py   # TF-IDF engine — loads CSV, computes scores
├── data/
│   └── gwalior_food_dataset_330_unique.csv
├── frontend/
│   ├── index.html       # App shell, card template
│   ├── style.css        # All styles (no framework)
│   └── script.js        # Fetch API, card rendering, favorites
├── .env                 # API keys (not committed to git)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## ⚠️ Known Limitations & Future Improvements

### Known Limitations

| Limitation | Details |
|---|---|
| **In-memory storage** | User profiles and liked IDs are lost on server restart. No database. |
| **Single user** | Currently hardcoded to `"default_user"` — no auth or multi-user support. |
| **Synthetic dataset** | 330-row dataset is synthetically generated; real-world data would improve quality. |
| **Image URLs** | Placeholder images from LoremFlickr; may not always load or may be cached inaccurately. |
| **Cold-start personalization** | With zero likes, the personal boost is 0 and all weight falls on TF-IDF + quality. |
| **LLM rate limits** | OpenRouter free-tier models may throttle; the rule-based fallback ensures availability. |

### Future Improvements

- [ ] **Persistent storage** — Replace in-memory `user_profiles` with SQLite or PostgreSQL
- [ ] **User authentication** — JWT-based auth to support multiple users
- [ ] **Collaborative filtering** — Use dish ratings across users (matrix factorization / ALS)
- [ ] **Real dataset** — Scrape actual Gwalior restaurant data from Swiggy/Zomato
- [ ] **Voice input** — Web Speech API integration for hands-free ordering
- [ ] **Dish images** — Integrate a food photo API (Unsplash / Pexels with food tags)
- [ ] **Feedback loop** — Let users rate recommendations to continuously improve the model
- [ ] **Streaming responses** — Server-Sent Events for progressive AI reply streaming
- [ ] **PWA support** — Offline-capable Progressive Web App with service workers

---

## 📋 Submission

```
╔══════════════════════════════════════════════╗
║       AI FOOD RECOMMENDATION SYSTEM          ║
║              Final Submission                ║
╠══════════════════════════════════════════════╣
║  Candidate Name  :  Mridul Malvi             ║
║  Project         :  Gwalior Foodie           ║
║  Tech Stack      :  FastAPI · LangChain ·    ║
║                     TF-IDF · Scikit-learn    ║
║  Dataset         :  330 Gwalior dishes (CSV) ║
║  Algorithm       :  TF-IDF + Cosine Sim +    ║
║                     Quality Scaling +        ║
║                     Personal Boost Hybrid    ║
╚══════════════════════════════════════════════╝
```
