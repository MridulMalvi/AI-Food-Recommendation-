// ============================================================
//  Gwalior Foodie – Frontend Script v3
//  Auth-aware: reads JWT from localStorage, sends Authorization
//  header with every request. Handles like, dislike, click.
// ============================================================

const BASE_ORIGIN =
  window.location.origin &&
  window.location.origin !== "null" &&
  !window.location.protocol.startsWith("file")
    ? window.location.origin
    : "http://127.0.0.1:8000";

const API_URL      = `${BASE_ORIGIN}/api/chat`;
const FAVORITE_URL = `${BASE_ORIGIN}/api/favorite`;
const DISLIKE_URL  = `${BASE_ORIGIN}/api/dislike`;
const CLICK_URL    = `${BASE_ORIGIN}/api/click`;

// ----------------------------------------------------------------
// Auth helpers
// ----------------------------------------------------------------

function getToken() {
  return localStorage.getItem("gf_token") || "";
}

function getUsername() {
  return localStorage.getItem("gf_username") || "";
}

function authHeaders() {
  const token = getToken();
  return token
    ? { "Content-Type": "application/json", "Authorization": `Bearer ${token}` }
    : { "Content-Type": "application/json" };
}

function logout() {
  localStorage.removeItem("gf_token");
  localStorage.removeItem("gf_username");
  localStorage.removeItem("gwalior_foodie_favorites");
  localStorage.removeItem("gwalior_foodie_dislikes");
  window.location.replace("login.html");
}

// Redirect to login if no token present
if (!getToken()) {
  window.location.replace("login.html");
}

// ----------------------------------------------------------------
// DOM references
// ----------------------------------------------------------------

const conversation  = document.querySelector("#conversation");
const form          = document.querySelector("#chat-form");
const input         = document.querySelector("#message-input");
const sendButton    = document.querySelector("#send-button");
const noteEl        = document.querySelector("#connection-note");
const cardTemplate  = document.querySelector("#card-template");

// User pill in header
const userPill      = document.getElementById("user-pill");
const userNameEl    = document.getElementById("user-name");
const logoutBtn     = document.getElementById("logout-btn");

// Show username in header
const username = getUsername();
if (username && userPill) {
  userNameEl.textContent = username;
  userPill.hidden = false;
}

if (logoutBtn) {
  logoutBtn.addEventListener("click", logout);
}

// Personalised welcome message
const welcomeBubble = document.querySelector(".welcome-message .bubble");
if (welcomeBubble && username) {
  welcomeBubble.innerHTML =
    `<strong>Welcome back, ${username}! 👋</strong><br>` +
    `I remember your taste — looking for a quick dinner, comforting vegetarian food, or something spicy? Just ask!`;
}

// ----------------------------------------------------------------
// Chat history
// ----------------------------------------------------------------

const history = [];

// ----------------------------------------------------------------
// In-browser cache of liked / disliked IDs (hydrated from localStorage)
// ----------------------------------------------------------------

const favoriteDishIds = new Set(
  JSON.parse(localStorage.getItem("gwalior_foodie_favorites") || "[]")
);
const dislikedDishIds = new Set(
  JSON.parse(localStorage.getItem("gwalior_foodie_dislikes") || "[]")
);

function saveFavoritesToStorage() {
  localStorage.setItem("gwalior_foodie_favorites", JSON.stringify(Array.from(favoriteDishIds)));
}

function saveDislikesToStorage() {
  localStorage.setItem("gwalior_foodie_dislikes", JSON.stringify(Array.from(dislikedDishIds)));
}

// ----------------------------------------------------------------
// UI helpers
// ----------------------------------------------------------------

function scrollToLatest() {
  conversation.scrollTop = conversation.scrollHeight;
}

function addMessage(text, role) {
  const article = document.createElement("article");
  article.className = `message ${role}-message`;
  if (role === "assistant") {
    article.innerHTML = '<div class="avatar">✦</div>';
  }
  const bubble = document.createElement("div");
  bubble.className = "bubble";
  bubble.textContent = text;
  article.appendChild(bubble);
  conversation.appendChild(article);
  scrollToLatest();
  return article;
}

function showTyping() {
  const row = document.createElement("article");
  row.className = "message assistant-message";
  row.id = "typing";
  row.innerHTML =
    '<div class="avatar">✦</div><div class="bubble typing"><i></i><i></i><i></i></div>';
  conversation.appendChild(row);
  scrollToLatest();
}

// ----------------------------------------------------------------
// Favorite / Like button logic
// ----------------------------------------------------------------

async function toggleFavorite(item, favButton) {
  const isCurrentlyFav = favoriteDishIds.has(item.id);
  const nextState = !isCurrentlyFav;

  // Optimistic UI update
  if (nextState) {
    favoriteDishIds.add(item.id);
    dislikedDishIds.delete(item.id);  // un-dislike if liked
    favButton.classList.add("active");
    favButton.setAttribute("title", "Favorited! ❤️");
    favButton.setAttribute("aria-label", "Remove from favorites");
  } else {
    favoriteDishIds.delete(item.id);
    favButton.classList.remove("active");
    favButton.setAttribute("title", "Favorite this dish");
    favButton.setAttribute("aria-label", "Save to favorites");
  }
  saveFavoritesToStorage();
  saveDislikesToStorage();

  // Sync with backend
  try {
    const payload = {
      item_id: item.id,
      is_favorite: nextState,
      dish: item.dish,
      restaurant: item.restaurant,
      cuisine: item.cuisine || null,
      price: item.price || null,
      rating: item.rating || null,
      area: item.area || null,
      dietary: item.dietary || null,
      spicy: item.spicy !== undefined ? item.spicy : null,
      image_url: item.image_url || null,
    };

    const resp = await fetch(FAVORITE_URL, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(payload),
    });

    if (resp.status === 401) { logout(); return; }
    if (!resp.ok) console.warn("Favorite endpoint returned:", resp.status);
  } catch (err) {
    console.error("Could not sync favorite with backend:", err);
  }
}

// ----------------------------------------------------------------
// Dislike button logic
// ----------------------------------------------------------------

async function toggleDislike(item, dislikeBtn, favBtn) {
  const isCurrentlyDisliked = dislikedDishIds.has(item.id);
  const nextState = !isCurrentlyDisliked;

  // Optimistic UI update
  if (nextState) {
    dislikedDishIds.add(item.id);
    favoriteDishIds.delete(item.id);  // un-like if disliked
    dislikeBtn.classList.add("active");
    dislikeBtn.setAttribute("title", "Not interested ✓");
    dislikeBtn.setAttribute("aria-label", "Remove dislike");
    favBtn.classList.remove("active");
    favBtn.setAttribute("title", "Favorite this dish");
  } else {
    dislikedDishIds.delete(item.id);
    dislikeBtn.classList.remove("active");
    dislikeBtn.setAttribute("title", "Not interested");
    dislikeBtn.setAttribute("aria-label", "Not interested");
  }
  saveFavoritesToStorage();
  saveDislikesToStorage();

  try {
    const resp = await fetch(DISLIKE_URL, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ item_id: item.id, is_disliked: nextState, dish: item.dish }),
    });
    if (resp.status === 401) { logout(); return; }
  } catch (err) {
    console.error("Could not sync dislike:", err);
  }
}

// ----------------------------------------------------------------
// Click / view tracking
// ----------------------------------------------------------------

async function recordClick(item) {
  try {
    await fetch(CLICK_URL, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ item_id: item.id, dish: item.dish }),
    });
  } catch (_) { /* silent */ }
}

// ----------------------------------------------------------------
// Card rendering
// ----------------------------------------------------------------

function makeCard(item) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);

  // --- Photo ---
  const img = card.querySelector("img");
  img.src = item.image_url || "";
  img.alt = item.dish;
  img.addEventListener("error", () => img.removeAttribute("src"));

  // --- Match badge ---
  const matchPct =
    item.match_percent !== undefined
      ? item.match_percent
      : Math.round((item.recommendation_score || 0) * 100);
  card.querySelector(".match").textContent = `${matchPct}% Match`;

  // --- Basic info ---
  card.querySelector("h3").textContent = item.dish;
  card.querySelector(".restaurant").textContent = item.restaurant;
  card.querySelector(".rating").textContent = `★ ${Number(item.rating).toFixed(1)}`;
  card.querySelector(".price").textContent = `₹${Math.round(item.price)}`;
  card.querySelector(".location").textContent = `${item.cuisine} · ${item.area}`;

  // --- AI Reason ---
  const reasonEl = card.querySelector(".reason");
  if (reasonEl) {
    reasonEl.textContent = item.reason || "";
    reasonEl.hidden = !item.reason;
  }

  // --- Delivery info ---
  card.querySelector(".delivery").textContent =
    `🚴 ${item.delivery_minutes} min   ·   📍 ${Number(item.distance_km).toFixed(1)} km`;

  // --- Badges ---
  const badges = card.querySelector(".badges");
  const dietBadge = document.createElement("span");
  dietBadge.className = "badge";
  dietBadge.textContent = `🥗 ${item.dietary}`;
  badges.appendChild(dietBadge);

  if (item.spicy) {
    const spicyBadge = document.createElement("span");
    spicyBadge.className = "badge spicy";
    spicyBadge.textContent = "🌶 Spicy";
    badges.appendChild(spicyBadge);
  }

  // --- Favorite button ---
  const favBtn = card.querySelector(".fav-btn");
  if (favoriteDishIds.has(item.id)) {
    favBtn.classList.add("active");
    favBtn.setAttribute("title", "Favorited! ❤️");
    favBtn.setAttribute("aria-label", "Remove from favorites");
  }
  favBtn.addEventListener("click", (e) => {
    e.stopPropagation();
    toggleFavorite(item, favBtn);
  });

  // --- Dislike button ---
  const dislikeBtn = card.querySelector(".dislike-btn");
  if (dislikeBtn) {
    if (dislikedDishIds.has(item.id)) {
      dislikeBtn.classList.add("active");
      dislikeBtn.setAttribute("title", "Not interested ✓");
    }
    dislikeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      toggleDislike(item, dislikeBtn, favBtn);
    });
  }

  // --- Record click when card is viewed ---
  card.addEventListener("click", () => recordClick(item), { once: true });

  return card;
}

function addRecommendations(items) {
  if (!items || !items.length) return;
  const row = document.createElement("article");
  row.className = "message assistant-message";
  row.innerHTML = '<div class="avatar">✦</div>';
  const grid = document.createElement("div");
  grid.className = "food-grid";
  items.forEach((item) => grid.appendChild(makeCard(item)));
  row.appendChild(grid);
  conversation.appendChild(row);
  scrollToLatest();
}

function showCategoryPage(category, items) {
  if (!items || !items.length) return;

  const row = document.createElement("article");
  row.className = "message assistant-message";
  row.innerHTML = '<div class="avatar">✦</div>';

  const spotlight = document.createElement("div");
  spotlight.className = "category-spotlight";

  const header = document.createElement("div");
  header.className = "spotlight-header";
  header.innerHTML = `
    <span class="spotlight-emoji" role="img" aria-hidden="true">${category.emoji}</span>
    <div class="spotlight-title-wrap">
      <span class="spotlight-eyebrow">Category Spotlight</span>
      <h2 class="spotlight-title">${category.label}</h2>
      <p class="spotlight-subtitle">Top picks in Gwalior</p>
    </div>
    <span class="spotlight-count">${items.length} dishes</span>
  `;
  spotlight.appendChild(header);

  const trackWrap = document.createElement("div");
  trackWrap.className = "spotlight-track-wrap";
  const track = document.createElement("div");
  track.className = "spotlight-track";
  items.forEach((item) => track.appendChild(makeCard(item)));
  trackWrap.appendChild(track);
  spotlight.appendChild(trackWrap);

  row.appendChild(spotlight);
  conversation.appendChild(row);
  scrollToLatest();
}

// ----------------------------------------------------------------
// Chat send / receive
// ----------------------------------------------------------------

async function sendMessage(message) {
  const clean = message.trim();
  if (!clean) return;

  addMessage(clean, "user");
  history.push({ role: "user", content: clean });
  input.value = "";
  input.disabled = true;
  sendButton.disabled = true;
  showTyping();

  try {
    const resp = await fetch(API_URL, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        message: clean,
        history: history.slice(0, -1).slice(-10),
        user_id: username || "default_user",
      }),
    });

    if (resp.status === 401) {
      logout();
      return;
    }

    const payload = await resp.json();
    if (!resp.ok) {
      throw new Error(payload.detail || "The service could not process that request.");
    }

    document.querySelector("#typing")?.remove();
    addMessage(payload.message, "assistant");
    history.push({ role: "assistant", content: payload.message });

    if (payload.category && payload.category_recommendations?.length) {
      showCategoryPage(payload.category, payload.category_recommendations);
    } else {
      addRecommendations(payload.recommendations || []);
    }

    const isPersonalised = payload.personalized;
    if (noteEl) {
      noteEl.textContent = isPersonalised
        ? `✨ Personalised for ${username || "you"} · TF-IDF + Cosine Similarity`
        : "Recommendations powered by TF-IDF + Cosine Similarity · LangChain Agent";
    }
  } catch (err) {
    document.querySelector("#typing")?.remove();
    addMessage(
      "I can't reach the recommendation service right now. Please ensure the server is running.",
      "assistant"
    );
    if (noteEl) noteEl.textContent = err.message;
  } finally {
    input.disabled = false;
    sendButton.disabled = false;
    input.focus();
  }
}

// ----------------------------------------------------------------
// Event listeners
// ----------------------------------------------------------------

form.addEventListener("submit", (e) => {
  e.preventDefault();
  sendMessage(input.value);
});

document.querySelectorAll("[data-prompt]").forEach((btn) =>
  btn.addEventListener("click", () => sendMessage(btn.dataset.prompt))
);
