import json
# ... باقي imports كما هي ...

def call_deepseek(history: list) -> str:
    """
    Send conversation history to HF model, validate with Guardrails if available,
    and always return a safe string (never return raw objects).
    """
    # 1) Call HF API
    try:
        resp = requests.post(
            API_URL,
            headers=HEADERS,
            json={"messages": history, "model": "deepseek-ai/DeepSeek-V3.2-Exp:novita"},
            timeout=20,
        )
    except Exception:
        logger.exception("Request to HF failed")
        return "Sorry, the model service is currently unavailable."

    # 2) Parse JSON
    try:
        result = resp.json()
    except Exception:
        logger.error("HF response is not JSON. status=%s text=%s", getattr(resp, "status_code", None), getattr(resp, "text", None))
        return "Sorry, the model returned an unexpected response."

    # 3) Check HF error shape
    if isinstance(result, dict) and "error" in result:
        logger.error("HF API returned an error: %s", result["error"])
        return "Sorry, the model service returned an error."

    # 4) Extract raw reply robustly
    try:
        raw_reply = result["choices"][0]["message"]["content"]
    except Exception:
        logger.error("Unexpected HF result structure: %s", result)
        return "Sorry, the model returned an unexpected result format."

    logger.info("RAW_REPLY (truncated): %s", (raw_reply[:400] if isinstance(raw_reply, str) else str(raw_reply)) )

    # 5) If Guardrails available, validate with tolerant strategy
    if guard is not None:
        validated_output = None
        parse_attempts = []

        # Attempt A: if raw_reply is JSON string, try to load and parse that
        try:
            parsed_candidate = json.loads(raw_reply)
            parse_attempts.append("json.loads -> guard.parse(parsed_candidate)")
            try:
                validated_output = guard.parse(parsed_candidate)
            except Exception:
                # second chance: wrap parsed_candidate under "reply"
                try:
                    validated_output = guard.parse({"reply": parsed_candidate})
                    parse_attempts.append("guard.parse({'reply': parsed_candidate})")
                except Exception:
                    pass
        except Exception:
            parse_attempts.append("raw_reply not JSON")

        # Attempt B: parse as {"reply": raw_reply}
        if validated_output is None:
            try:
                parse_attempts.append("guard.parse({'reply': raw_reply})")
                validated_output = guard.parse({"reply": raw_reply})
            except Exception:
                pass

        # Attempt C: parse raw string directly
        if validated_output is None:
            try:
                parse_attempts.append("guard.parse(raw_reply)")
                validated_output = guard.parse(raw_reply)
            except Exception:
                pass

        # Log parse attempts
        logger.debug("Guard parse attempts: %s", parse_attempts)

        # If validated_output found -> extract reply
        if validated_output is not None:
            try:
                reply_text = _extract_reply_from_validated(validated_output)
                if reply_text:
                    return reply_text
                else:
                    logger.warning("Guardrails validated output but no 'reply' extracted. validated_output type=%s value=%s", type(validated_output), getattr(validated_output, "__dict__", str(validated_output)))
                    # safe fallback: return raw reply with note
                    safe = raw_reply.strip() if isinstance(raw_reply, str) else str(raw_reply)
                    if len(safe) > 1000:
                        safe = safe[:1000] + "..."
                    return f"{safe}\n\nNote: Validation succeeded but no 'reply' could be extracted. Treat this as general information and consult a healthcare professional."
            except Exception:
                logger.exception("Error while extracting reply from validated output")
                # fallthrough to safe return below

        # If we get here: validation attempts all failed
        logger.warning("Guardrails validation failed. Returning raw reply with disclaimer. attempts=%s", parse_attempts)
        safe = raw_reply.strip() if isinstance(raw_reply, str) else str(raw_reply)
        if len(safe) > 1000:
            safe = safe[:1000] + "..."
        return f"{safe}\n\nNote: I couldn't run validation rules on this reply, so treat this as general information and consult a healthcare professional."

    else:
        # If guard not loaded: return a safe truncated raw reply and append reminder
        safe = (raw_reply.strip()[:200] + "...") if isinstance(raw_reply, str) and len(raw_reply) > 200 else (raw_reply.strip() if isinstance(raw_reply, str) else str(raw_reply))
        return f"{safe}\n\n Note: Validation rules are not loaded. Please consult a healthcare professional for final advice."