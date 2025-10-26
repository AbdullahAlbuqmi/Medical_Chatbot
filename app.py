import os
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrails import Guard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Load Guardrails from medical_guard.rail file
try:
    guard = Guard.from_rail("medical_guard.rail")
    logger.info("Guardrails loaded successfully from medical_guard.rail")
except Exception as e:
    logger.error(f"Failed to load Guardrails: {e}")
    guard = None

# Data models
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str

# Fixed function to call DeepSeek with Guardrails
def call_deepseek(history: list) -> str:
    """Send conversation history to HF model with Guardrails application"""
    
    try:
        payload = {
            "messages": history,
            "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        logger.info(f"Sending request to HuggingFace API...")
        
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        # Check response status
        if response.status_code == 401:
            return "Authentication error. Please check your API Token."
        elif response.status_code == 429:
            return "Rate limit exceeded. Please wait a moment and try again."
        elif response.status_code == 503:
            return "Model is currently loading. Please try again in 30 seconds."
        elif response.status_code != 200:
            return f"Server error: {response.status_code}"
        
        # Process successful response
        result = response.json()
        raw_reply = result["choices"][0]["message"]["content"]
        
        logger.info(f"Received raw reply: {raw_reply[:100]}...")
        
        # Apply Guardrails if loaded
        if guard is not None:
            try:
                logger.info("Applying Guardrails validation...")
                
                # Correct way: pass raw text to Guardrails for parsing
                validation_result = guard.parse(raw_reply)
                
                # Extract final reply
                if hasattr(validation_result, 'validated_output') and validation_result.validated_output:
                    if hasattr(validation_result.validated_output, 'reply'):
                        final_reply = validation_result.validated_output.reply
                    elif isinstance(validation_result.validated_output, dict) and 'reply' in validation_result.validated_output:
                        final_reply = validation_result.validated_output['reply']
                    else:
                        final_reply = str(validation_result.validated_output)
                    
                    logger.info(f"Reply after validation: {final_reply[:100]}...")
                    return final_reply
                else:
                    # If validation fails, use raw reply with warning
                    logger.warning("Guardrails validation failed, using raw reply")
                    return f"{raw_reply}\n\nNote: This response was not validated according to medical rules"
                    
            except Exception as e:
                logger.error(f"Error in Guardrails: {e}")
                # In case of error, return raw reply
                return raw_reply
        else:
            # If Guardrails not loaded, return raw reply
            return raw_reply
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return "Request timeout. Please try again."
    
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        return "Connection error with server. Please check your internet."
    
    except KeyError as e:
        logger.error(f"Unexpected response structure: {e}")
        return "Sorry, there's an issue with the response format from the model."
    
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return "Sorry, an unexpected error occurred. Please try again later."

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
        
        # Call the model
        reply = call_deepseek(history)
        
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