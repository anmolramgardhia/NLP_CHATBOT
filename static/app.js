/**
 * app.js — Frontend chat interface
 * NLP Chatbot Project — Restaurant Booking Domain
 *
 * Connects to FastAPI backend at POST /chat
 * Displays entity chips extracted by the NER pipeline
 */

const API_URL = "/chat";

// Persist session_id across turns so dialogue state is maintained
let sessionId = null;

// ── DOM refs ──────────────────────────────────────────────────────────────────
const chatMessages = document.getElementById("chat-messages");
const userInput    = document.getElementById("user-input");
const sendBtn      = document.getElementById("send-btn");

// ── Event listeners ───────────────────────────────────────────────────────────
sendBtn.addEventListener("click", sendMessage);
userInput.addEventListener("keypress", (e) => {
  if (e.key === "Enter" && !e.shiftKey) sendMessage();
});

// ── Core send/receive ─────────────────────────────────────────────────────────
async function sendMessage() {
  const text = userInput.value.trim();
  if (!text) return;

  // Render user bubble immediately
  appendMessage(text, "user-message");
  userInput.value = "";
  setInputEnabled(false);

  // Typing indicator
  const typingId = showTypingIndicator();

  try {
    const res = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: text, session_id: sessionId }),
    });

    if (!res.ok) throw new Error(`Server error: ${res.status}`);

    const data = await res.json();

    // Persist session for multi-turn dialogue
    sessionId = data.session_id;

    removeTypingIndicator(typingId);
    appendMessage(data.reply, "bot-message");

    // Show entity chips if any entities were detected
    const hasEntities = Object.keys(data.entities).length > 0;
    if (hasEntities) {
      appendEntityChips(data.entities, data.intent, data.sentiment);
    }

  } catch (err) {
    removeTypingIndicator(typingId);
    appendMessage("Sorry, I couldn't connect to the server. Is the backend running?", "bot-message error-message");
    console.error("Chat error:", err);
  } finally {
    setInputEnabled(true);
    userInput.focus();
  }
}

// ── Message rendering ─────────────────────────────────────────────────────────
function appendMessage(text, className) {
  const div = document.createElement("div");
  div.classList.add("message", ...className.split(" "));
  div.textContent = text;
  chatMessages.appendChild(div);
  scrollToBottom();
  return div;
}

/**
 * Renders extracted NER entities as coloured chips below the bot's reply.
 * Also shows intent badge and sentiment indicator.
 */
function appendEntityChips(entities, intent, sentiment) {
  const container = document.createElement("div");
  container.classList.add("entity-chips");

  const ENTITY_CONFIG = {
    date:        { emoji: "📅", label: "Date" },
    time:        { emoji: "🕐", label: "Time" },
    party_size:  { emoji: "👥", label: "Party" },
    person_name: { emoji: "👤", label: "Name" },
    dietary:     { emoji: "🥗", label: "Diet" },
  };

  for (const [key, value] of Object.entries(entities)) {
    if (!value || (Array.isArray(value) && value.length === 0)) continue;

    const config = ENTITY_CONFIG[key] || { emoji: "🏷️", label: key };
    const displayValue = Array.isArray(value) ? value.join(", ") : value;

    const chip = document.createElement("span");
    chip.classList.add("chip", `chip-${key}`);
    chip.innerHTML = `${config.emoji} <strong>${config.label}:</strong> ${displayValue}`;
    container.appendChild(chip);
  }

  // Intent badge
  if (intent && intent !== "out_of_scope") {
    const badge = document.createElement("span");
    badge.classList.add("chip", "chip-intent");
    badge.innerHTML = `🎯 <strong>Intent:</strong> ${intent.replace(/_/g, " ")}`;
    container.appendChild(badge);
  }

  // Sentiment indicator
  const SENTIMENT_MAP = {
    positive:   { emoji: "😊", cls: "sentiment-positive" },
    neutral:    { emoji: "😐", cls: "sentiment-neutral"  },
    negative:   { emoji: "😕", cls: "sentiment-negative" },
    frustrated: { emoji: "😤", cls: "sentiment-frustrated" },
  };
  const sm = SENTIMENT_MAP[sentiment] || SENTIMENT_MAP.neutral;
  const sentChip = document.createElement("span");
  sentChip.classList.add("chip", sm.cls);
  sentChip.textContent = sm.emoji;
  sentChip.title = `Sentiment: ${sentiment}`;
  container.appendChild(sentChip);

  chatMessages.appendChild(container);
  scrollToBottom();
}

// ── Typing indicator ──────────────────────────────────────────────────────────
function showTypingIndicator() {
  const id  = "typing-" + Date.now();
  const div = document.createElement("div");
  div.classList.add("message", "bot-message", "typing-indicator");
  div.id = id;
  div.innerHTML = "<span></span><span></span><span></span>";
  chatMessages.appendChild(div);
  scrollToBottom();
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

// ── Helpers ───────────────────────────────────────────────────────────────────
function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function setInputEnabled(enabled) {
  userInput.disabled = !enabled;
  sendBtn.disabled   = !enabled;
}
