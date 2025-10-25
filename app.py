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
    logger.info("✅ Guardrails loaded successfully from medical_guard.rail")
except Exception as e:
    logger.error(f"❌ Failed to load Guardrails: {e}")
    guard = None

# نموذج البيانات
class ChatRequest(BaseModel):
    message: str
    conversation_history: list = []

class ChatResponse(BaseModel):
    reply: str
    status: str

# دالة مساعدة لاستخراج الرد من Guardrails output
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

# دالة الاتصال بـ DeepSeek مع Guardrails
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
        
        logger.info(f"✅ تم استلام رد خام: {raw_reply[:100]}...")
        
        # تطبيق Guardrails إذا كان محملاً
        if guard is not None:
            try:
                logger.info("🛡️ تطبيق Guardrails للتحقق...")
                
                # محاولات متعددة للتحقق باستخدام Guardrails
                validated_output = None
                
                # المحاولة الأولى: تحقق مباشر من الرد الخام
                try:
                    validated_output = guard.parse(raw_reply)
                    logger.info("✅ التحقق باستخدام Guardrails نجح (المحاولة الأولى)")
                except Exception as e1:
                    logger.warning(f"المحاولة الأولى فشلت: {e1}")
                    
                    # المحاولة الثانية: تغليف الرد في كائن JSON
                    try:
                        validated_output = guard.parse({"reply": raw_reply})
                        logger.info("✅ التحقق باستخدام Guardrails نجح (المحاولة الثانية)")
                    except Exception as e2:
                        logger.warning(f"المحاولة الثانية فشلت: {e2}")
                        
                        # المحاولة الثالثة: استخدام prompt مخصص
                        try:
                            last_message = history[-1]["content"] if history else ""
                            prompt_with_context = f"""
User Question: {last_message}

Please provide a helpful medical response following these rules:
- Provide general health information only
- Do not give personal diagnoses
- Remind to consult a doctor
- Use simple plain text

Response to validate: {raw_reply}
"""
                            validated_output = guard.parse(prompt_with_context)
                            logger.info("✅ التحقق باستخدام Guardrails نجح (المحاولة الثالثة)")
                        except Exception as e3:
                            logger.error(f"جميع محاولات Guardrails فشلت: {e3}")
                            validated_output = None
                
                # استخراج الرد النهائي
                if validated_output is not None:
                    final_reply = _extract_reply_from_validated(validated_output)
                    if final_reply:
                        logger.info(f"🔄 الرد بعد التحقق: {final_reply[:100]}...")
                        return final_reply
                    else:
                        logger.warning("⚠️ Guardrails نجح لكن لم يتم استخراج رد")
                        return f"{raw_reply}\n\nملاحظة: هذه معلومات عامة - يرجى استشارة طبيب"
                else:
                    logger.warning("🛡️ Guardrails فشل في التحقق، استخدام الرد الخام مع تحذير")
                    return f"{raw_reply}\n\n⚠️ ملاحظة: لم يتم التحقق من هذا الرد طبياً، يرجى استشارة متخصص"
                    
            except Exception as e:
                logger.error(f"❌ خطأ غير متوقع في Guardrails: {e}")
                return f"{raw_reply}\n\n⚠️ ملاحظة: حدث خطأ في التحقق، يرجى استشارة طبيب"
        else:
            # إذا Guardrails غير محمل، ارجع الرد الخام مع تحذير
            logger.warning("🛡️ Guardrails غير محمل، استخدام الرد الخام")
            return f"{raw_reply}\n\n⚠️ ملاحظة: لم يتم تطبيق قواعد التحقق الطبي على هذا الرد"
        
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