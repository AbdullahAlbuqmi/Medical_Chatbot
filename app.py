import os
import requests
import logging
import json
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrails import Guard
from guardrails.hub import ValidLength

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Guardrails warnings
import warnings
warnings.filterwarnings("ignore", message="Could not obtain an event loop")
logging.getLogger("guardrails-ai").setLevel(logging.ERROR)

# Initialize FastAPI
app = FastAPI(title="Medical Chatbot API", version="1.0.0")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Environment variables
API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# Create Guardrails programmatically with validators
guard = None
try:
    # Try to load from RAIL file first
    guard = Guard.from_rail("medical_guard.rail")
    logger.info("Guardrails loaded successfully from medical_guard.rail")
except Exception as e:
    logger.warning(f"Could not load RAIL file: {e}")
    try:
        # Fallback: Create guard programmatically
        from guardrails import Guard
        from pydantic import BaseModel, Field
        
        class MedicalResponse(BaseModel):
            reply: str = Field(
                description="AI medical assistant's response",
                min_length=10,
                max_length=2000
            )
        
        guard = Guard.from_pydantic(
            output_class=MedicalResponse,
            prompt="""You are a medical AI assistant. You MUST follow these rules strictly:

CRITICAL: If the user's question is NOT about health, medical topics, wellness, symptoms, or healthcare, you MUST respond with EXACTLY this text:
"I can only help with general health questions. Please consult a healthcare professional for medical advice."

For health-related questions:
1. Provide general health information and education only
2. Do not give personal medical diagnoses or specific treatment advice
3. Do not prescribe medications or suggest specific dosages
4. Always include a reminder to consult with healthcare professionals
5. Use simple, clear language (50-300 words)
6. Never use markdown, JSON, or code formatting - plain text only
7. Be empathetic, supportive, and non-judgmental

User question: {{user_input}}

Your response:"""
        )
        logger.info("Guardrails created programmatically")
    except Exception as e2:
        logger.error(f"Failed to create Guardrails: {e2}")
        guard = None

# Data models
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str

def call_deepseek_with_guard(user_message: str, history: list) -> str:
    """Call DeepSeek using Guardrails __call__ method for proper prompt injection"""
    
    try:
        if guard is None:
            # Fallback if guard not loaded
            return call_deepseek_direct(user_message, history)
        
        logger.info("Using Guardrails to generate and validate response...")
        
        # Use Guardrails __call__ method which handles prompt injection and validation
        result = guard(
            llm_api=lambda prompt: call_llm_with_prompt(prompt, history),
            prompt_params={"user_input": user_message},
            num_reasks=2,
            max_tokens=500,
            temperature=0.7
        )
        
        # Extract validated output
        if result.validated_output:
            if isinstance(result.validated_output, dict):
                reply = result.validated_output.get('reply', str(result.validated_output))
            else:
                reply = str(result.validated_output)
            
            logger.info(f"Guardrails validated response: {reply[:100]}...")
            return clean_medical_response(reply)
        else:
            logger.warning("No validated output, using raw output")
            return clean_medical_response(str(result.raw_llm_output))
            
    except Exception as e:
        logger.error(f"Error in Guardrails processing: {e}")
        logger.exception(e)
        # Fallback to direct call
        return call_deepseek_direct(user_message, history)

def call_llm_with_prompt(prompt: str, history: list) -> str:
    """Call the LLM with a specific prompt (used by Guardrails)"""
    
    try:
        # Guardrails injects the full prompt, so we use it directly
        messages = [
            {"role": "system", "content": "You are a helpful medical AI assistant."}
        ]
        
        # Add conversation history if available
        if history:
            messages.extend(history[:-1])  # Exclude the last user message as it's in the prompt
        
        # Add the Guardrails-generated prompt as the user message
        messages.append({"role": "user", "content": prompt})
        
        payload = {
            "messages": messages,
            "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        if response.status_code != 200:
            raise Exception(f"API error: {response.status_code}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
        
    except Exception as e:
        logger.error(f"Error calling LLM: {e}")
        raise

def call_deepseek_direct(user_message: str, history: list) -> str:
    """Direct call to DeepSeek without Guardrails (fallback)"""
    
    try:
        # Add system message with strict instructions
        messages = [{
            "role": "system",
            "content": """You are a medical AI assistant. CRITICAL RULES:

If the question is NOT about health/medical topics, respond EXACTLY: "I can only help with general health questions. Please consult a healthcare professional for medical advice."

For health questions:
1. Provide general health information only
2. Do NOT diagnose or prescribe
3. ALWAYS include reminder to consult healthcare professionals
4. Use plain text, 50-300 words
5. Be empathetic and supportive"""
        }]
        
        messages.extend(history)
        
        payload = {
            "messages": messages,
            "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        logger.info("Sending direct request to HuggingFace API...")
        
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        if response.status_code == 401:
            return "Authentication error. Please check your API Token."
        elif response.status_code == 429:
            return "Rate limit exceeded. Please wait a moment and try again."
        elif response.status_code == 503:
            return "Model is currently loading. Please try again in 30 seconds."
        elif response.status_code != 200:
            return f"Server error: {response.status_code}"
        
        result = response.json()
        raw_reply = result["choices"][0]["message"]["content"]
        
        return clean_medical_response(raw_reply)
        
    except requests.exceptions.Timeout:
        return "Request timeout. Please try again."
    except requests.exceptions.ConnectionError:
        return "Connection error. Please check your internet."
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return "Sorry, an unexpected error occurred. Please try again later."

def clean_medical_response(text: str) -> str:
    """Clean and format medical response"""
    
    # Remove markdown formatting
    text = text.replace('**', '').replace('*', '').replace('`', '')
    text = text.replace('#', '').strip()
    
    # Remove JSON or code blocks
    if text.startswith('{') and text.endswith('}'):
        try:
            data = json.loads(text)
            if 'reply' in data:
                text = data['reply']
        except:
            pass
    
    text = text.replace('```', '').strip()
    
    # Ensure length limits
    if len(text) < 10:
        text = "I understand your concern. " + text
    
    if len(text) > 2000:
        text = text[:1997] + "..."
    
    # Add disclaimer if missing
    disclaimer_phrases = [
        "consult a doctor", "consult with a healthcare professional", 
        "talk to your doctor", "seek medical advice", "healthcare provider",
        "medical professional"
    ]
    
    has_disclaimer = any(phrase in text.lower() for phrase in disclaimer_phrases)
    if not has_disclaimer and "i can only help with general health questions" not in text.lower():
        text += " Please consult with a healthcare professional for personalized medical advice."
    
    return text

# Root endpoint
@app.get("/")
async def root():
    guard_status = "active" if guard is not None else "inactive"
    return {
        "message": "Medical Chatbot API - DeepSeek Assistant", 
        "status": "active",
        "version": "1.0.0",
        "guardrails": guard_status
    }

# Health endpoint
@app.get("/health")
async def health_check():
    guard_status = "loaded" if guard is not None else "not_loaded"
    return {
        "status": "healthy", 
        "service": "medical_chatbot",
        "guardrails": guard_status
    }

# Main chat endpoint
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # Build conversation history
        history = request.conversation_history.copy()
        history.append({"role": "user", "content": request.message})
        
        logger.info(f"User message: {request.message}")
        
        # Call the model with Guardrails
        reply = call_deepseek_with_guard(request.message, history)
        
        logger.info("Request processed successfully")
        return ChatResponse(reply=reply, status="success")
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )

# Run server
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)