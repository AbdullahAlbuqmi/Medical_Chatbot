import os
import requests
import logging
import json
import asyncio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrails import Guard

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress Guardrails warnings
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

# Load Guardrails from medical_guard.rail file
guard = None
try:
    guard = Guard.from_rail("medical_guard.rail")
    logger.info("Guardrails loaded successfully from medical_guard.rail")
except Exception as e:
    logger.error(f"Failed to load Guardrails: {e}")
    logger.info("Continuing without Guardrails validation")

# Data models
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str

def validate_with_guardrails(raw_reply: str) -> str:
    """Validate reply using Guardrails with proper error handling"""
    if guard is None:
        return clean_medical_response(raw_reply)
    
    try:
        # Create the structured output
        llm_output = {
            "reply": raw_reply
        }
        
        # Convert to JSON string
        llm_output_str = json.dumps(llm_output)
        
        logger.info("Validating with Guardrails...")
        
        # Use guard.parse with proper parameters
        validation_result = guard.parse(
            llm_output=llm_output_str,
            num_reasks=1
        )
        
        # Check validation result
        if validation_result.validation_passed:
            validated_data = validation_result.validated_output
            if isinstance(validated_data, dict) and 'reply' in validated_data:
                final_reply = validated_data['reply']
            elif isinstance(validated_data, str):
                # Sometimes Guardrails returns a JSON string
                try:
                    parsed = json.loads(validated_data)
                    final_reply = parsed.get('reply', validated_data)
                except:
                    final_reply = validated_data
            else:
                final_reply = str(validated_data)
            
            logger.info("Guardrails validation passed")
            return final_reply
        else:
            logger.warning(f"Validation failed: {validation_result.error}")
            # Fall back to cleaned response
            return clean_medical_response(raw_reply)
            
    except Exception as e:
        logger.error(f"Error in Guardrails validation: {str(e)}")
        # Fall back to cleaned response
        return clean_medical_response(raw_reply)

# Improved function to call DeepSeek with Guardrails
def call_deepseek(history: list) -> str:
    """Send conversation history to HF model with Guardrails application"""
    
    try:
        # Prepare the payload
        payload = {
            "messages": history,
            "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        logger.info("Sending request to HuggingFace API...")
        
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
        
        # Apply Guardrails validation
        final_reply = validate_with_guardrails(raw_reply)
        
        logger.info("Request processed successfully")
        return final_reply
        
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

def clean_medical_response(text: str) -> str:
    """Clean and format medical response to pass Guardrails validation"""
    
    # Remove any markdown formatting
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
    
    # Remove code block markers
    text = text.replace('```', '').strip()
    
    # Ensure the response is within length limits (10-2000 chars)
    if len(text) < 10:
        text = "I understand your health concern. " + text
    
    if len(text) > 2000:
        text = text[:1997] + "..."
    
    # Add medical disclaimer if not present
    disclaimer_phrases = [
        "consult a doctor", "consult with a healthcare professional", 
        "talk to your doctor", "seek medical advice", "healthcare provider",
        "medical professional"
    ]
    
    has_disclaimer = any(phrase in text.lower() for phrase in disclaimer_phrases)
    if not has_disclaimer:
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