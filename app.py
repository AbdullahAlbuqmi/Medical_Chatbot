import json
import os
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from guardrails.hub import NSFWText, ToxicLanguage
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
API_URL = os.getenv("HF_API_URL", "https://api-inference.huggingface.co/models/deepseek-ai/DeepSeek-V3.2-Exp")
HF_TOKEN = os.getenv("HF_TOKEN", "your_token_here")

HEADERS = {
    "Authorization": f"Bearer {HF_TOKEN}",
    "Content-Type": "application/json"
}

# تعريف Guardrails
try:
    # الطريقة الأولى: استخدام rail specification كسلسلة نصية
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

# دالة الاتصال بـ DeepSeek
def call_deepseek(history: list) -> str:
    """إرسال تاريخ المحادثة إلى نموذج HF"""
    
    # 1) استدعاء HF API
    try:
        resp = requests.post(
            API_URL,
            headers=HEADERS,
            json={
                "messages": history, 
                "model": "deepseek-ai/DeepSeek-V3.2-Exp",
                "max_tokens": 500,
                "temperature": 0.7
            },
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        logger.exception(f"Request to HF failed: {e}")
        return "عذراً، خدمة النموذج غير متاحة حالياً."

    # 2) تحليل JSON
    try:
        result = resp.json()
    except Exception as e:
        logger.error(f"HF response is not JSON: {e}")
        return "عذراً، النموذج أعاد استجابة غير متوقعة."

    # 3) التحقق من خطأ HF
    if isinstance(result, dict) and "error" in result:
        logger.error(f"HF API returned an error: {result['error']}")
        return "عذراً، خدمة النموذج أعادت خطأ."

    # 4) استخراج الرد الخام
    try:
        raw_reply = result["choices"][0]["message"]["content"]
    except Exception as e:
        logger.error(f"Unexpected HF result structure: {result}, error: {e}")
        return "عذراً، النموذج أعاد تنسيق نتيجة غير متوقع."

    logger.info(f"RAW_REPLY: {raw_reply[:200]}...")

    # 5) إذا كان Guardrails متاحاً، التحقق باستخدام استراتيجية متساهلة
    if guard is not None:
        try:
            # محاولة التحقق باستخدام Guardrails
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
        # إذا لم يتم تحميل Guardrails
        safe_reply = raw_reply[:500] + "..." if len(raw_reply) > 500 else raw_reply
        return f"{safe_reply}\n\nملاحظة: قواعد التحقق غير محملة. يرجى استشارة متخصص طبي للنصيحة النهائية."

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