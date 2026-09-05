const API_URL = (window.location.origin && window.location.origin !== "null" && !window.location.protocol.startsWith("file"))
  ? `${window.location.origin}/chat`
  : "http://127.0.0.1:8000/chat";
const conversation = document.querySelector("#conversation");
const form = document.querySelector("#chat-form");
const input = document.querySelector("#message-input");
const sendButton = document.querySelector("#send-button");
const note = document.querySelector("#connection-note");
const cardTemplate = document.querySelector("#card-template");
const history = [];

function scrollToLatest() { conversation.scrollTop = conversation.scrollHeight; }
function addMessage(text, role) {
  const article = document.createElement("article");
  article.className = `message ${role}-message`;
  if (role === "assistant") article.innerHTML = '<div class="avatar">✦</div>';
  const bubble = document.createElement("div"); bubble.className = "bubble"; bubble.textContent = text;
  article.appendChild(bubble); conversation.appendChild(article); scrollToLatest();
  return article;
}
function showTyping() {
  const row = document.createElement("article"); row.className = "message assistant-message"; row.id = "typing";
  row.innerHTML = '<div class="avatar">✦</div><div class="bubble typing"><i></i><i></i><i></i></div>';
  conversation.appendChild(row); scrollToLatest();
}
function makeCard(item) {
  const card = cardTemplate.content.firstElementChild.cloneNode(true);
  const image = card.querySelector("img");
  image.src = item.image_url || ""; image.alt = item.dish;
  image.addEventListener("error", () => { image.removeAttribute("src"); });
  card.querySelector(".match").textContent = `Match ${Math.round(item.recommendation_score * 100)}%`;
  card.querySelector("h3").textContent = item.dish;
  card.querySelector(".restaurant").textContent = item.restaurant;
  card.querySelector(".rating").textContent = `★ ${Number(item.rating).toFixed(1)}`;
  card.querySelector(".price").textContent = `₹${Math.round(item.price)}`;
  card.querySelector(".location").textContent = `${item.cuisine} · ${item.area}`;
  const badges = card.querySelector(".badges");
  const diet = document.createElement("span"); diet.className = "badge"; diet.textContent = `🥗 ${item.dietary}`; badges.appendChild(diet);
  if (item.spicy) { const spicy = document.createElement("span"); spicy.className = "badge spicy"; spicy.textContent = "🌶 Spicy"; badges.appendChild(spicy); }
  card.querySelector(".delivery").textContent = `🚴 ${item.delivery_minutes} min   ·   📍 ${Number(item.distance_km).toFixed(1)} km`;
  return card;
}
function addRecommendations(items) {
  if (!items.length) return;
  const row = document.createElement("article"); row.className = "message assistant-message";
  row.innerHTML = '<div class="avatar">✦</div>';
  const grid = document.createElement("div"); grid.className = "food-grid";
  items.forEach(item => grid.appendChild(makeCard(item))); row.appendChild(grid); conversation.appendChild(row); scrollToLatest();
}
async function sendMessage(message) {
  const cleanMessage = message.trim(); if (!cleanMessage) return;
  addMessage(cleanMessage, "user"); history.push({ role: "user", content: cleanMessage });
  input.value = ""; input.disabled = true; sendButton.disabled = true; showTyping();
  try {
    const response = await fetch(API_URL, { method:"POST", headers:{ "Content-Type":"application/json" }, body:JSON.stringify({ message:cleanMessage, history:history.slice(0, -1).slice(-10) }) });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.detail || "The service could not process that request.");
    document.querySelector("#typing")?.remove(); addMessage(payload.message, "assistant"); history.push({ role:"assistant", content:payload.message }); addRecommendations(payload.recommendations || []);
    note.textContent = "Recommendations are from the Gwalior dataset and ranked with the trained model.";
  } catch (error) {
    document.querySelector("#typing")?.remove(); addMessage("I can’t reach the recommendation service right now. Please make sure the FastAPI server is running, then try again.", "assistant");
    note.textContent = error.message;
  } finally { input.disabled = false; sendButton.disabled = false; input.focus(); }
}
form.addEventListener("submit", event => { event.preventDefault(); sendMessage(input.value); });
document.querySelectorAll("[data-prompt]").forEach(button => button.addEventListener("click", () => sendMessage(button.dataset.prompt)));
