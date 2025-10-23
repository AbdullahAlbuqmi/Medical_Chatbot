import os
import requests
import logging
from typing import Any, Optional

from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# Try import Guardrails; if missing, app still runs but guard will be disabled.
try:
    from guardrails import Guard
except Exception:
    Guard = None  # type: ignore

# ---------- config & logging ----------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- FastAPI Init ----------
app = FastAPI(title="Medical AI Chatbots API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Env + Guardrails ----------
HF_API_KEY = os.environ.get("HF_API_KEY")
if not HF_API_KEY:
    logger.warning("HF_API_KEY is not set. HF requests will fail until you set this env var.")

API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"} if HF_API_KEY else {}

# load the rail file safely
guard: Optional[Any] = None
if Guard is not None:
    try:
        guard = Guard.from_rail("medical_guard.rail")
        logger.info("Guardrails loaded from medical_guard.rail")
    except Exception:
        logger.exception("Failed to load medical_guard.rail — Guardrails disabled for now.")
        guard = None
else:
    logger.warning("guardrails package not available. Install guardrails-ai to enable rules.")

# ---------- Schema ----------
class ChatRequest(BaseModel):
    user_input: str
    session_id: str

# ---------- Histories (in-memory; replace with DB/Redis in production) ----------
simple_history: dict = {}
advanced_history: dict = {}

# ---------- Prompts ----------
SIMPLE_PROMPT = """
You are a friendly medical assistant focused on symptom guidance.
Ask minimal follow-up questions and provide simple advice.
Keep answers short and calming. Always encourage seeking real doctors for serious symptoms.
"""

ADVANCED_PROMPT = """
You are a highly skilled medical research assistant.
You analyze studies, compare treatments, summarize clinical guidelines, and explain conditions in-depth.
Use structured responses and cite medical reasoning.
Ask follow-up questions if needed, no more than 3 questions, and provide advanced advice or guidance.
Always state uncertainty and do not assume a final diagnosis without full context.
"""

# ---------- Helper: extract a safe string from Guardrails output ----------
def _extract_reply_from_validated(validated_output: Any) -> Optional[str]:
    """
    Try multiple strategies to extract a string reply from validated_output.
    Return None if extraction failed.
    """
    # 1) If it's already a dict-like
    try:
        if isinstance(validated_output, dict):
            return validated_output.get("reply") or validated_output.get("answer") or validated_output.get("response")
    except Exception:
        pass

    # 2) If supports item access (some versions)
    try:
        reply = validated_output["reply"]
        if isinstance(reply, str):
            return reply
        else:
            return str(reply)
    except Exception:
        pass

    # 3) Attribute access
    try:
        reply = getattr(validated_output, "reply", None)
        if isinstance(reply, str):
            return reply
        if reply is not None:
            return str(reply)
    except Exception:
        pass

    # 4) Look for .value() or .dict() (pydantic models)
    try:
        if hasattr(validated_output, "dict"):
            d = validated_output.dict()
            return d.get("reply") or d.get("answer") or d.get("response") or None
    except Exception:
        pass

    try:
        if hasattr(validated_output, "value"):
            val = validated_output.value
            if isinstance(val, str):
                return val
            if isinstance(val, dict):
                return val.get("reply") or val.get("answer") or None
    except Exception:
        pass

    # 5) Fallback to stringifying (for debug only) — but prefer None so caller can return safe message
    try:
        s = str(validated_output)
        if s and len(s.strip()) > 0:
            return s
    except Exception:
        pass

    return None

# ---------- Helper LLM Call ----------
def call_deepseek(history: list) -> str:
    """
    Send conversation history to HF model, validate with Guardrails if available,
    and always return a safe string (never return a raw object).
    """
    # 1) Call HF API
    try:
        resp = requests.post(
            API_URL,
            headers=HEADERS,
            json={"messages": history, "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita"},
            timeout=20,
        )
    except Exception:
        logger.exception("Request to HF failed")
        return "Sorry, the model service is currently unavailable."

    # 2) Parse JSON
    try:
        result = resp.json()
    except Exception:
        logger.error("HF response is not JSON. status=%s text=%s", getattr(resp, "status_code", None), getattr(resp, "text", None))
        return "Sorry, the model returned an unexpected response."

    # 3) Check HF error shape
    if isinstance(result, dict) and "error" in result:
        logger.error("HF API returned an error: %s", result["error"])
        return "Sorry, the model service returned an error."

    # 4) Extract raw reply robustly
    try:
        raw_reply = result["choices"][0]["message"]["content"]
    except Exception:
        logger.error("Unexpected HF result structure: %s", result)
        return "Sorry, the model returned an unexpected result format."

    logger.info("RAW_REPLY (truncated): %s", raw_reply[:400])

    # 5) If Guardrails available, validate
    if guard is not None:
        try:
            validated_output = guard.parse(raw_reply)
        except Exception:
            logger.exception("Guardrails parse() raised an exception")
            return "Sorry, I cannot provide that information right now. Please consult a healthcare professional."

        reply_text = _extract_reply_from_validated(validated_output)
        if reply_text:
            return reply_text
        else:
            logger.warning("Guardrails validated output but no 'reply' could be extracted. validated_output type=%s", type(validated_output))
            return "Sorry, I cannot provide that information. Please consult a healthcare professional."
    else:
        # If guard not loaded: return a safe truncated raw reply and append reminder
        safe = (raw_reply.strip()[:200] + "...") if len(raw_reply) > 200 else raw_reply.strip()
        return f"{safe}\n\n⚠ Note: Validation rules are not loaded. Please consult a healthcare professional for final advice."

# ---------- Endpoints ----------
@app.post("/simple_chat")
def simple_chat(req: ChatRequest):
    # initialize history for session
    if req.session_id not in simple_history:
        simple_history[req.session_id] = [{"role": "system", "content": SIMPLE_PROMPT}]

    # append user message
    simple_history[req.session_id].append({"role": "user", "content": req.user_input})

    # call model
    reply = call_deepseek(simple_history[req.session_id])

    # ensure reply is string
    if not isinstance(reply, str):
        logger.warning("Reply not string, converting. type=%s", type(reply))
        reply = str(reply)

    # append assistant reply to history as plain text
    simple_history[req.session_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}


@app.post("/advanced_chat")
def advanced_chat(req: ChatRequest):
    if req.session_id not in advanced_history:
        advanced_history[req.session_id] = [{"role": "system", "content": ADVANCED_PROMPT}]

    advanced_history[req.session_id].append({"role": "user", "content": req.user_input})
    reply = call_deepseek(advanced_history[req.session_id])

    if not isinstance(reply, str):
        logger.warning("Reply not string, converting. type=%s", type(reply))
        reply = str(reply)

    advanced_history[req.session_id].append({"role": "assistant", "content": reply})
    return {"reply": reply}