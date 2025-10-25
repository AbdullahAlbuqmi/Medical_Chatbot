import os
import requests
import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware

# إعداد logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# تهيئة FastAPI
app = FastAPI(title="Medical Chatbot API", version="1.0.0")

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

# نموذج البيانات
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str

# دالة الاتصال بـ DeepSeek - مبسطة وبدون تعقيد
def call_deepseek(history: list) -> str:
    """إرسال تاريخ المحادثة إلى نموذج HF والحصول على الرد"""
    
    try:
        payload = {
            "messages": history,
            "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita",
            "max_tokens": 500,
            "temperature": 0.7
        }
        
        logger.info(f"إرسال طلب إلى HuggingFace API...")
        
        response = requests.post(
            API_URL, 
            headers=headers, 
            json=payload,
            timeout=30
        )
        
        # التحقق من حالة الاستجابة
        if response.status_code == 401:
            return "❌ خطأ في المصادقة. يرجى التحقق من صحة الـ API Token."
        elif response.status_code == 429:
            return "⏳ تم تجاوز الحد المسموح. يرجى الانتظار قليلاً والمحاولة مرة أخرى."
        elif response.status_code == 503:
            return "🔄 النموذج يحمل الآن. يرجى المحاولة مرة أخرى خلال 30 ثانية."
        elif response.status_code != 200:
            return f"⚠️ خطأ من الخادم: {response.status_code}"
        
        # معالجة الرد الناجح
        result = response.json()
        raw_reply = result["choices"][0]["message"]["content"]
        
        logger.info(f"✅ تم استلام رد بنجاح: {raw_reply[:100]}...")
        return raw_reply
        
    except requests.exceptions.Timeout:
        logger.error("⏰ انتهت مهلة الطلب")
        return "⏰ انتهت المهلة. يرجى المحاولة مرة أخرى."
    
    except requests.exceptions.ConnectionError:
        logger.error("🔌 خطأ في الاتصال")
        return "🔌 خطأ في الاتصال بالخادم. يرجى التحقق من الإنترنت."
    
    except KeyError as e:
        logger.error(f"📋 هيكل غير متوقع للرد: {e}")
        return "📋 عذراً، هناك مشكلة في تنسيق الرد من النموذج."
    
    except Exception as e:
        logger.exception(f"❌ خطأ غير متوقع: {e}")
        return "❌ عذراً، حدث خطأ غير متوقع. يرجى المحاولة لاحقاً."

# endpoint الجذر
@app.get("/")
async def root():
    return {
        "message": "Medical Chatbot API - DeepSeek Assistant", 
        "status": "active",
        "version": "1.0.0"
    }

# endpoint الحالة
@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "medical_chatbot"}

# endpoint الدردشة الرئيسي
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # بناء تاريخ المحادثة
        history = request.conversation_history.copy()
        history.append({"role": "user", "content": request.message})
        
        logger.info(f"📨 رسالة المستخدم: {request.message}")
        
        # استدعاء النموذج
        reply = call_deepseek(history)
        
        logger.info("✅ تمت معالجة الطلب بنجاح")
        return ChatResponse(reply=reply, status="success")
        
    except Exception as e:
        logger.error(f"❌ خطأ في endpoint الدردشة: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)