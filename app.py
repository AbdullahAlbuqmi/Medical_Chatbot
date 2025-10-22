import requests
from fastapi import FastAPI
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrails import Guard

# FastAPI Init
app = FastAPI(title="Medical AI Chatbots API")

# CORS MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins (or specify your frontend URL)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (POST, GET, etc.)
    allow_headers=["*"],  # Allows all headers
)

import os
HF_API_KEY = os.environ.get("HF_API_KEY")
API_URL = "https://router.huggingface.co/v1/chat/completions"
HEADERS = {"Authorization": f"Bearer {HF_API_KEY}"}
guard = Guard.from_rail("medical_guard.rail")

# Message Schema
class ChatRequest(BaseModel):
    user_input: str
    session_id: str  # You can use frontend ID or user token

# Chat Histories 
simple_history = {}
advanced_history = {}


# System Prompts

SIMPLE_PROMPT = """
You are a friendly medical assistant focused on symptom guidance.
Ask minimal follow-up questions and provide simple advice.
Keep answers short and calming. Always encourage seeking real doctors for serious symptoms.
"""

# Advanced Prompts

ADVANCED_PROMPT = """
You are a highly skilled medical research assistant.
You analyze studies, compare treatments, summarize clinical guidelines, and explain conditions in-depth.
Use structured responses and cite medical reasoning.
Ask follow-up questions if needed, no more than 3 questions, and provide advanced advice or guidance.
Always state uncertainty and do not assume a final diagnosis without full context.
"""

# Helper LLM Call

def call_deepseek(history):
    response = requests.post(API_URL, headers=HEADERS, json={
        "messages": history,
        "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita"
    })
    result = response.json()
    raw_reply = result["choices"][0]["message"]["content"]

    # Apply Guardrails Validation
    try:
        validated_output = guard.parse(raw_reply)
        return validated_output["reply"]
    except Exception:
        return "Sorry, I cannot provide that information."


# Endpoint 1: Simple Chatbot
@app.post("/simple_chat")
def simple_chat(req: ChatRequest):
    if req.session_id not in simple_history:
        simple_history[req.session_id] = [{"role": "system", "content": SIMPLE_PROMPT}]
    
    simple_history[req.session_id].append({"role": "user", "content": req.user_input})
    reply = call_deepseek(simple_history[req.session_id])
    simple_history[req.session_id].append({"role": "assistant", "content": reply})

    return {"reply": reply}


# Endpoint 2: Advanced Research Chatbot

@app.post("/advanced_chat")
def advanced_chat(req: ChatRequest):
        if req.session_id not in advanced_history:
            advanced_history[req.session_id] = [{"role": "system", "content": ADVANCED_PROMPT}]
        
        advanced_history[req.session_id].append({"role": "user", "content": req.user_input})
        reply = call_deepseek(advanced_history[req.session_id])
        advanced_history[req.session_id].append({"role": "assistant", "content": reply})

        return {"reply": reply}
