"""
main.py — FastAPI application entry point
NLP Chatbot Project — Restaurant Booking Domain

Serves:
  POST /chat          — Main chat endpoint
  GET  /              — Serves the frontend UI
  GET  /health        — Health check

Run with:
  uvicorn main:app --reload
Then open: http://localhost:8000
"""

import logging
import uuid
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel

# ── Path fix so `src/` imports work ──────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.preprocess import clean_text
from src.intent_classifier import TFIDFIntentClassifier, load_intent_data
from src.ner import NERExtractor
from src.sentiment import SentimentDetector
from src.dialogue import DialogueStateTracker
from src.response import ResponseGenerator

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(title="NLP Restaurant Chatbot", version="1.0.0")

# ── Pydantic models ───────────────────────────────────────────────────────────
class ChatRequest(BaseModel):
    message: str
    session_id: str | None = None  # optional; auto-generated if not provided

class ChatResponse(BaseModel):
    reply: str
    session_id: str
    intent: str
    confidence: float
    entities: dict
    sentiment: str

# ── Global pipeline objects (loaded once at startup) ─────────────────────────
intent_clf: TFIDFIntentClassifier = None
ner_extractor: NERExtractor = None
sentiment_detector: SentimentDetector = None
dialogue_tracker: DialogueStateTracker = None
response_generator: ResponseGenerator = None

MODEL_PATH = Path(__file__).parent / "models" / "tfidf_pipeline.joblib"

@app.on_event("startup")
async def load_pipeline():
    """Load and initialise all NLP components once at server startup."""
    global intent_clf, ner_extractor, sentiment_detector, dialogue_tracker, response_generator

    logger.info("Loading NLP pipeline...")

    # Intent classifier — train on the fly if no saved model exists
    if MODEL_PATH.exists():
        logger.info(f"Loading saved TF-IDF model from {MODEL_PATH}")
        intent_clf = TFIDFIntentClassifier.load()
    else:
        logger.info("No saved model found — training TF-IDF classifier now...")
        texts, labels = load_intent_data()
        intent_clf = TFIDFIntentClassifier()
        intent_clf.fit(texts, labels)
        intent_clf.save()
        logger.info("TF-IDF model trained and saved.")

    ner_extractor      = NERExtractor()
    sentiment_detector = SentimentDetector()
    dialogue_tracker   = DialogueStateTracker()
    response_generator = ResponseGenerator()

    logger.info("Pipeline ready ✓")


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": intent_clf is not None}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Runs the full NLP pipeline: preprocess → intent → NER → sentiment → dialogue → response.
    """
    if not req.message or not req.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    session_id = req.session_id or str(uuid.uuid4())
    user_text  = req.message.strip()

    # 1. Preprocess
    cleaned = clean_text(user_text)

    # 2. Intent classification
    intent_result = intent_clf.predict(cleaned)
    intent        = intent_result["intent"]
    confidence    = intent_result["confidence"]

    # 3. Named entity recognition
    entities = ner_extractor.extract(cleaned)

    # 4. Sentiment analysis
    sentiment_result = sentiment_detector.analyse(cleaned)
    sentiment_label  = sentiment_result["label"]

    # 5. Dialogue state management (slot filling)
    turn_result = dialogue_tracker.process_turn(
        session_id=session_id,
        user_text=cleaned,
        intent=intent,
        entities=entities,
        confidence=confidence,
    )

    # 6. Response generation
    state = turn_result["state"]
    if turn_result["action"] == "clarify":
        reply = response_generator.clarify(
            turn_result["clarifying_question"], sentiment_label
        )
    elif intent == "out_of_scope":
        # Use LLM fallback for truly out-of-scope queries
        reply = response_generator.llm_fallback(state.history, user_text)
    else:
        reply = response_generator.generate(intent, state, sentiment_label)

    # Record bot turn in history
    state.add_turn("assistant", reply)

    # Build a clean entities dict for the frontend (no internal 'raw' key)
    clean_entities = {k: v for k, v in entities.items() if k != "raw" and v}

    logger.info(f"[{session_id[:8]}] intent={intent} ({confidence:.2f}) | sentiment={sentiment_label}")

    return ChatResponse(
        reply=reply,
        session_id=session_id,
        intent=intent,
        confidence=round(confidence, 4),
        entities=clean_entities,
        sentiment=sentiment_label,
    )


# ── Serve frontend ────────────────────────────────────────────────────────────
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
def serve_ui():
    return FileResponse("static/index.html")


# ── CLI fallback (python main.py) ─────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
