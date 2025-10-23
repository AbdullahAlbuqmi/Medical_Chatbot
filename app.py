import os
import requests
import logging
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrails import Guard

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

# Make sure medical_guard.rail path is correct relative to where you run the app
guard = Guard.from_rail("medical_guard.rail")

# ---------- Schema ----------
class ChatRequest(BaseModel):
    user_input: str
    session_id: str

# ---------- Histories ----------
simple_history = {}
advanced_history = {}

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

# ---------- Helper LLM Call ----------
def call_deepseek(history):
    # call HF
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

    # parse JSON
    try:
        result = resp.json()
    except Exception:
        logger.error("HF response is not JSON. status=%s text=%s", resp.status_code, resp.text)
        return "Sorry, the model returned an unexpected response."

    # check HF error
    if isinstance(result, dict) and "error" in result:
        logger.error("HF API returned an error: %s", result["error"])
        return "Sorry, the model service returned an error."

    # extract raw reply robustly
    try:
        raw_reply = result["choices"][0]["message"]["content"]
    except Exception:
        logger.error("Unexpected HF result structure: %s", result)
        return "Sorry, the model returned an unexpected result format."

    logger.info("RAW_REPLY (truncated): %s", raw_reply[:400])

    # validate with Guardrails
    try:
        validated_output = guard.parse(raw_reply)
    except Exception:
        logger.exception("Guardrails parse() raised an exception")
        return "Sorry, I cannot provide that information right now. Please consult a healthcare professional."

    # extract final reply from validated_output (multiple fallbacks)
    reply_text = None

    # If it's a dict-like
    try:
        if isinstance(validated_output, dict):
            reply_text = validated_output.get("reply") or validated_output.get("answer")
    except Exception:
        pass

    # If supports item access
    if reply_text is None:
        try:
            reply_text = validated_output["reply"]
        except Exception:
            pass

    # If attribute access
    if reply_text is None:
        reply_text = getattr(validated_output, "reply", None)

    # Some Guardrails versions wrap result in .value or .content
    if reply_text is None:
        candidate = getattr(validated_output, "value", None) or getattr(validated_output, "content", None)
        if isinstance(candidate, dict):
            reply_text = candidate.get("reply") or candidate.get("answer")
        elif isinstance(candidate, str):
            reply_text = candidate

    # final fallback (do not return raw_reply directly if unsafe)
    if not reply_text:
        logger.warning(
            "Could not extract 'reply' from validated_output. type=%s dir(sample)=%s",
            type(validated_output),
            dir(validated_output)[:50],
        )
        return "Sorry, I cannot provide that information. Please consult a healthcare professional."

    return reply_text

# ---------- Endpoints ----------
@app.post("/simple_chat")
def simple_chat(req: ChatRequest):
    if req.session_id not in simple_history:
        simple_history[req.session_id] = [{"role": "system", "content": SIMPLE_PROMPT}]

    simple_history[req.session_id].append({"role": "user", "content": req.user_input})
    reply = call_deepseek(simple_history[req.session_id])
    simple_history[req.session_id].append({"role": "assistant", "content": reply})
    return {"reply": reply}


@app.post("/advanced_chat")
def advanced_chat(req: ChatRequest):
    if req.session_id not in advanced_history:
        advanced_history[req.session_id] = [{"role": "system", "content": ADVANCED_PROMPT}]

    advanced_history[req.session_id].append({"role": "user", "content": req.user_input})
    reply = call_deepseek(advanced_history[req.session_id])
    advanced_history[req.session_id].append({"role": "assistant", "content": reply})
    return {"reply": reply}