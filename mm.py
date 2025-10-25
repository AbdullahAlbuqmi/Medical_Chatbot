import requests

try:
    # أولاً جرب تستعلم عن السيرفر الأساسي
    print("testing server...")
    base_response = requests.get("https://medical-chatbot-gtdl.onrender.com/")
    print(f"الحالة: {base_response.status_code}")
    print(f"الرد: {base_response.json()}")
    
    # بعدين جرب الدردشة
    print("\ntesting chat...")
    chat_response = requests.post(
        "https://medical-chatbot-gtdl.onrender.com/chat",
        json={
            "message": "I'm seek, high pressure",
            "conversation_history": []
        },
        timeout=30
    )
    print(f"conversation status: {chat_response.status_code}")
    print(f"reply: {chat_response.json()}")
    
except requests.exceptions.Timeout:
    print("❌ المهلة انتهت - السيرفر ياخذ وقت طويل")
except requests.exceptions.ConnectionError:
    print("❌ لا يمكن الاتصال بالسيرفر")
except Exception as e:
    print(f"❌ خطأ: {e}")