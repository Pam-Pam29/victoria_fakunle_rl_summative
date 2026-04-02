"""
Sista Health RL — FastAPI REST Endpoint
Loads the best PPO model and serves action predictions.

Usage:
    uvicorn api:app --reload

POST /predict
    Body: { "language": 2, "domain": 1, "topic": 3, "literacy": 0 }
    Returns: { "action": 1, "action_name": "Voice Note", "description": "..." }
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import numpy as np
import os

from stable_baselines3 import PPO

MODEL_PATH = os.path.join("models", "pg", "ppo", "best_ppo_model.zip")

app = FastAPI(
    title="Sista Health RL API",
    description="Reinforcement learning agent that selects the optimal health communication format for Nigerian women.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = None

@app.on_event("startup")
def load_model():
    global model
    try:
        model = PPO.load(MODEL_PATH)
        print(f" PPO model loaded from {MODEL_PATH}")
    except Exception as e:
        print(f" Failed to load model: {e}")


# ── Action metadata ───────────────────────────────────────────────────────────
ACTIONS = {
    0: {
        "name": "Text Response",
        "description": "A written reply. Optimal for high-literacy users.",
        "emoji": "💬"
    },
    1: {
        "name": "Voice Note",
        "description": "An audio message in the user's language. Optimal for low-literacy and Pidgin-speaking users.",
        "emoji": "🎙️"
    },
    2: {
        "name": "Resource Link",
        "description": "An external reference (clinic, hotline, or article). Best when topic is relevant and user has literacy >= medium.",
        "emoji": "🔗"
    },
    3: {
        "name": "Clarify",
        "description": "A clarifying question. Optimal at the start of a session or for sensitive topics.",
        "emoji": "❓"
    },
}

LANGUAGES = {0: "English", 1: "Yoruba", 2: "Pidgin"}
DOMAINS   = {0: "Sexual Health", 1: "Maternal Health"}
LITERACY  = {0: "Low", 1: "Medium", 2: "High"}
TOPICS    = {
    0: "FGM Complications", 1: "VVF Causes", 2: "Cultural Barriers",
    3: "Early Marriage", 4: "TBA Dangers", 5: "Contraception",
    6: "STIs and HIV", 7: "Antenatal Care", 8: "Postpartum Care"
}


# ── Request / Response schemas ────────────────────────────────────────────────
class UserContext(BaseModel):
    language: int = Field(..., ge=0, le=2, description="0=English, 1=Yoruba, 2=Pidgin")
    domain:   int = Field(..., ge=0, le=1, description="0=Sexual Health, 1=Maternal Health")
    topic:    int = Field(..., ge=0, le=8, description="0-8, see TOPICS mapping")
    literacy: int = Field(..., ge=0, le=2, description="0=Low, 1=Medium, 2=High")
    # FIX: removed urgency — not part of the observation space

class PredictionResponse(BaseModel):
    action:       int
    action_name:  str
    description:  str
    emoji:        str
    user_profile: dict
    model_used:   str


# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {
        "message": "Sista Health RL API is running.",
        "endpoints": {
            "POST /predict": "Get action recommendation for a user context",
            "GET  /health":  "Health check",
            "GET  /docs":    "Interactive API documentation"
        }
    }

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictionResponse)
def predict(context: UserContext):
    if model is None:
        return {"error": "Model not loaded. Check server logs."}

    # FIX: observation matches environment exactly — [language, domain, topic, literacy, step]
    obs = np.array([
        context.language,
        context.domain,
        context.topic,
        context.literacy,
        0  # step — always 0 at session start
    ], dtype=np.float32)

    action, _ = model.predict(obs, deterministic=True)
    action = int(action)

    return PredictionResponse(
        action=action,
        action_name=ACTIONS[action]["name"],
        description=ACTIONS[action]["description"],
        emoji=ACTIONS[action]["emoji"],
        user_profile={
            "language": LANGUAGES[context.language],
            "domain":   DOMAINS[context.domain],
            "topic":    TOPICS[context.topic],
            "literacy": LITERACY[context.literacy],
        },
        model_used="PPO — best_ppo_model.zip (Run 1, Mean Reward 114.33)"
    )