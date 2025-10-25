import json
import os
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrails import Guard

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة FastAPI
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# إعداد المتغيرات البيئية
API_URL = "https://router.huggingface.co/v1/chat/completions"
HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here")

headers = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# تعريف Guardrails
try:
    rail_spec = """
<rail version="0.1">
<output>
    <string name="reply" description="AI medical assistant's helpful and informative response to the user"/>
</output>
<instructions>
- You are a friendly and knowledgeable AI medical assistant.
- You may greet users, ask how they feel, and maintain a warm conversational tone.
- You may provide general health information, explanations of symptoms, or advice on when to seek medical care.
- Do NOT give personalized medical diagnoses, prescriptions, or specific dosages.
- Always include a reminder that your guidance does not replace professional medical advice.
- Avoid topics unrelated to health (politics, religion, personal relationships, or entertainment).
- If the user insists on a non-medical topic, reply with: "Sorry, I can only assist with health-related questions."
- Respond in simple plain text (no markdown, JSON, or code formatting).
</instructions>
</rail>
"""
    guard = Guard.from_rail_string(rail_spec)
    logger.info("Guardrails loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Guardrails: {e}")
    guard = None

# نموذج البيانات
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str

# دالة مساعدة لاستخراج الرد
def _extract_reply_from_validated(validated_output):
    """استخراج الرد من المخرجات المصدق عليها"""
    try:
        if isinstance(validated_output, dict) and "reply" in validated_output:
            return validated_output["reply"]
        elif hasattr(validated_output, "reply"):
            return validated_output.reply
        else:
            return str(validated_output)
    except Exception as e:
        logger.error(f"Error extracting reply: {e}")
        return None

# دالة الاتصال بـ DeepSeek المحدثة
def call_deepseek(history: list) -> str:
    """إرسال تاريخ المحادثة إلى نموذج HF باستخدام الـ endpoint الجديد"""
    
    try:
        payload = {
            "messages": history,
            "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        logger.info(f"Sending request to HF API: {payload}")
        
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        logger.info(f"HF API Status: {response.status_code}")
        
        if response.status_code != 200:
            logger.error(f"HF API Error: {response.status_code} - {response.text}")
            if response.status_code == 401:
                return "خطأ في المصادقة. يرجى التحقق من الـ API token."
            elif response.status_code == 429:
                return "تم تجاوز الحد المسموح. يرجى المحاولة لاحقاً."
            else:
                return f"خطأ من الخادم: {response.status_code}"
        
        result = response.json()
        logger.info(f"HF API Response: {result}")
        
        # استخراج الرد من الهيكل الجديد
        raw_reply = result["choices"][0]["message"]["content"]
        logger.info(f"RAW_REPLY: {raw_reply[:200]}...")
        
    except requests.exceptions.Timeout:
        logger.error("Request timeout")
        return "المهلة انتهت. يرجى المحاولة مرة أخرى."
    except requests.exceptions.ConnectionError:
        logger.error("Connection error")
        return "خطأ في الاتصال. يرجى التحقق من الإنترنت والمحاولة مرة أخرى."
    except KeyError as e:
        logger.error(f"Unexpected response structure: {e}")
        return "عذراً، النموذج أعاد تنسيق نتيجة غير متوقع."
    except Exception as e:
        logger.exception(f"Request to HF failed: {e}")
        return "عذراً، خدمة النموذج غير متاحة حالياً."

    # تطبيق Guardrails إذا كان شغال
    if guard is not None:
        try:
            validated_output = guard.parse(raw_reply)
            reply_text = _extract_reply_from_validated(validated_output)
            
            if reply_text:
                return reply_text
            else:
                logger.warning("Guardrails validated but no reply extracted")
                return f"{raw_reply}\n\nملاحظة: تم التحقق ولكن لم يتم استخراج رد. يرجى استشارة متخصص طبي."
                
        except Exception as e:
            logger.warning(f"Guardrails validation failed: {e}")
            return f"{raw_reply}\n\nملاحظة: لم أتمكن من التحقق من هذا الرد، لذا تعامل مع هذه المعلومات كإرشادات عامة واستشر متخصصاً طبياً."
    else:
        return raw_reply

# endpoint الجذر
@app.get("/")
async def root():
    return {"message": "DeepSeek Medical Assistant API", "status": "active"}

# endpoint الدردشة
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # بناء تاريخ المحادثة
        history = request.conversation_history.copy()
        history.append({"role": "user", "content": request.message})
        
        # استدعاء النموذج
        reply = call_deepseek(history)
        
        return ChatResponse(reply=reply, status="success")
        
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)