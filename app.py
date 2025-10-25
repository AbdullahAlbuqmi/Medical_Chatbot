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

# تحميل Guardrails من ملف medical_guard.rail
try:
    guard = Guard.from_rail("medical_guard.rail")
    logger.info(" Guardrails loaded successfully from medical_guard.rail")
except Exception as e:
    logger.error(f" Failed to load Guardrails: {e}")
    guard = None

# نموذج البيانات
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str

# دالة مبسطة لاستخراج الرد من Guardrails
def _extract_reply(validation_outcome):
    """استخراج الرد من نتيجة التحقق بشكل مبسط"""
    try:
        # إذا كان هناك raw_llm_output، استخدمه مباشرة
        if hasattr(validation_outcome, 'raw_llm_output') and validation_outcome.raw_llm_output:
            return validation_outcome.raw_llm_output
        # إذا كان هناك validated_output
        elif hasattr(validation_outcome, 'validated_output') and validation_outcome.validated_output:
            if hasattr(validation_outcome.validated_output, 'reply'):
                return validation_outcome.validated_output.reply
            elif isinstance(validation_outcome.validated_output, dict) and 'reply' in validation_outcome.validated_output:
                return validation_outcome.validated_output['reply']
        # إذا فشل التحقق ولكن هناك raw_llm_output
        elif hasattr(validation_outcome, 'raw_llm_output') and validation_outcome.raw_llm_output:
            return validation_outcome.raw_llm_output
    except Exception as e:
        logger.error(f"Error extracting reply: {e}")
    return None

# دالة الاتصال بـ DeepSeek مع Guardrails مبسط
def call_deepseek(history: list) -> str:
    """إرسال تاريخ المحادثة إلى نموذج HF مع تطبيق Guardrails"""
    
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
            return " خطأ في المصادقة. يرجى التحقق من صحة الـ API Token."
        elif response.status_code == 429:
            return " تم تجاوز الحد المسموح. يرجى الانتظار قليلاً والمحاولة مرة أخرى."
        elif response.status_code == 503:
            return " النموذج يحمل الآن. يرجى المحاولة مرة أخرى خلال 30 ثانية."
        elif response.status_code != 200:
            return f" خطأ من الخادم: {response.status_code}"
        
        # معالجة الرد الناجح
        result = response.json()
        raw_reply = result["choices"][0]["message"]["content"]
        
        logger.info(f"تم استلام رد خام: {raw_reply[:100]}...")
        
        # تطبيق Guardrails إذا كان محملاً
        if guard is not None:
            try:
                logger.info(" تطبيق Guardrails للتحقق...")
                
                # الطريقة المبسطة: إنشاء كائن JSON يتوافق مع توقعات الـ rail
                simple_output = {"reply": raw_reply}
                
                # استخدم Guardrails للتحقق
                validation_result = guard.parse(simple_output)
                
                # استخرج الرد النهائي
                final_reply = _extract_reply(validation_result)
                
                if final_reply:
                    logger.info(f" الرد بعد التحقق: {final_reply[:100]}...")
                    return final_reply
                else:
                    logger.warning(" لم يتم استخراج رد من Guardrails")
                    return raw_reply
                    
            except Exception as e:
                logger.error(f" خطأ في Guardrails: {e}")
                # في حالة الخطأ، ارجع الرد الخام مع إضافة تحذير
                return f"{raw_reply}\n\n ملاحظة: لم يتم تطبيق قواعد التحقق الطبي بسبب خطأ تقني"
        else:
            # إذا Guardrails غير محمل، ارجع الرد الخام
            return raw_reply
        
    except requests.exceptions.Timeout:
        logger.error(" انتهت مهلة الطلب")
        return " انتهت المهلة. يرجى المحاولة مرة أخرى."
    
    except requests.exceptions.ConnectionError:
        logger.error(" خطأ في الاتصال")
        return " خطأ في الاتصال بالخادم. يرجى التحقق من الإنترنت."
    
    except KeyError as e:
        logger.error(f" هيكل غير متوقع للرد: {e}")
        return " عذراً، هناك مشكلة في تنسيق الرد من النموذج."
    
    except Exception as e:
        logger.exception(f" خطأ غير متوقع: {e}")
        return "عذراً، حدث خطأ غير متوقع. يرجى المحاولة لاحقاً."

# endpoint الجذر
@app.get("/")
async def root():
    guard_status = "active" if guard is not None else "inactive"
    return {
        "message": "Medical Chatbot API - DeepSeek Assistant", 
        "status": "active",
        "version": "1.0.0",
        "guardrails": guard_status
    }

# endpoint الحالة
@app.get("/health")
async def health_check():
    guard_status = "loaded" if guard is not None else "not_loaded"
    return {
        "status": "healthy", 
        "service": "medical_chatbot",
        "guardrails": guard_status
    }

# endpoint الدردشة الرئيسي
@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        # بناء تاريخ المحادثة
        history = request.conversation_history.copy()
        history.append({"role": "user", "content": request.message})
        
        logger.info(f"رسالة المستخدم: {request.message}")
        
        # استدعاء النموذج
        reply = call_deepseek(history)
        
        logger.info(" تمت معالجة الطلب بنجاح")
        return ChatResponse(reply=reply, status="success")
        
    except Exception as e:
        logger.error(f" خطأ في endpoint الدردشة: {e}")
        raise HTTPException(
            status_code=500, 
            detail="Internal server error"
        )

# تشغيل الخادم
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)