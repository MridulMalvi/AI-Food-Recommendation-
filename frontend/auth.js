// ============================================================
//  Gwalior Foodie — Auth Page Script (login.html)
//  Handles register / login, stores JWT in localStorage.
// ============================================================

const BASE_ORIGIN =
  window.location.origin &&
  window.location.origin !== "null" &&
  !window.location.protocol.startsWith("file")
    ? window.location.origin
    : "http://127.0.0.1:8000";

const REGISTER_URL = `${BASE_ORIGIN}/api/auth/register`;
const LOGIN_URL    = `${BASE_ORIGIN}/api/auth/login`;

// ---------------------------------------------------------------------------
// Redirect if already logged in
// ---------------------------------------------------------------------------
if (localStorage.getItem("gf_token")) {
  window.location.replace("index.html");
}

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

const tabLogin    = document.getElementById("tab-login");
const tabRegister = document.getElementById("tab-register");
const panelLogin  = document.getElementById("panel-login");
const panelReg    = document.getElementById("panel-register");

function showTab(which) {
  const isLogin = which === "login";
  tabLogin.classList.toggle("active", isLogin);
  tabRegister.classList.toggle("active", !isLogin);
  tabLogin.setAttribute("aria-selected", isLogin);
  tabRegister.setAttribute("aria-selected", !isLogin);
  panelLogin.hidden = !isLogin;
  panelReg.hidden = isLogin;
}

tabLogin.addEventListener("click", () => showTab("login"));
tabRegister.addEventListener("click", () => showTab("register"));
document.getElementById("goto-register").addEventListener("click", () => showTab("register"));
document.getElementById("goto-login").addEventListener("click", () => showTab("login"));

// ---------------------------------------------------------------------------
// Password visibility toggles
// ---------------------------------------------------------------------------

function wireToggle(btnId, inputId) {
  const btn = document.getElementById(btnId);
  const inp = document.getElementById(inputId);
  if (!btn || !inp) return;
  btn.addEventListener("click", () => {
    const isHidden = inp.type === "password";
    inp.type = isHidden ? "text" : "password";
    btn.textContent = isHidden ? "🙈" : "👁";
  });
}

wireToggle("toggle-login-pw", "login-password");
wireToggle("toggle-reg-pw", "reg-password");

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function setLoading(formId, loading) {
  const btn = document.querySelector(`#${formId} .auth-submit`);
  const txt = btn.querySelector(".btn-text");
  const spin = btn.querySelector(".btn-spinner");
  btn.disabled = loading;
  txt.hidden = loading;
  spin.hidden = !loading;
}

function showError(elId, msg) {
  const el = document.getElementById(elId);
  el.textContent = msg;
  el.hidden = !msg;
}

function saveSession(token, username) {
  localStorage.setItem("gf_token", token);
  localStorage.setItem("gf_username", username);
}

// ---------------------------------------------------------------------------
// Login
// ---------------------------------------------------------------------------

document.getElementById("form-login").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("login-error", "");
  const username = document.getElementById("login-username").value.trim().toLowerCase();
  const password = document.getElementById("login-password").value;

  if (!username || !password) {
    showError("login-error", "Please fill in all fields.");
    return;
  }

  setLoading("form-login", true);
  try {
    const resp = await fetch(LOGIN_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "Login failed. Please check your credentials.");
    }
    saveSession(data.access_token, data.username);
    window.location.replace("index.html");
  } catch (err) {
    showError("login-error", err.message);
  } finally {
    setLoading("form-login", false);
  }
});

// ---------------------------------------------------------------------------
// Register
// ---------------------------------------------------------------------------

document.getElementById("form-register").addEventListener("submit", async (e) => {
  e.preventDefault();
  showError("reg-error", "");

  const username = document.getElementById("reg-username").value.trim().toLowerCase();
  const password = document.getElementById("reg-password").value;
  const confirm  = document.getElementById("reg-confirm").value;

  if (!username || !password || !confirm) {
    showError("reg-error", "Please fill in all fields.");
    return;
  }
  if (password.length < 6) {
    showError("reg-error", "Password must be at least 6 characters.");
    return;
  }
  if (password !== confirm) {
    showError("reg-error", "Passwords do not match.");
    return;
  }

  setLoading("form-register", true);
  try {
    const resp = await fetch(REGISTER_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await resp.json();
    if (!resp.ok) {
      throw new Error(data.detail || "Registration failed. Try a different username.");
    }
    saveSession(data.access_token, data.username);
    window.location.replace("index.html");
  } catch (err) {
    showError("reg-error", err.message);
  } finally {
    setLoading("form-register", false);
  }
});
