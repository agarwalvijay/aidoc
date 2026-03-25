import asyncio
import io
import json as _json
import logging
import time

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel

from .config import settings
from .llm import get_clinician_llm, get_llm_status
from .models import (
    ChatMessage,
    ConfidenceLevel,
    RedFlagHit,
    Role,
    StartSessionRequest,
    StartSessionResponse,
    TriageAssessment,
    TurnRequest,
    TurnResponse,
    UrgencyLevel,
)
from .pipeline import run_assessment_pipeline
from .red_flags import detect_red_flags
from .session_store import session_store
from .triage import apply_safety_policy, build_red_flag_assessment

logger = logging.getLogger("uvicorn.error")
MAX_REPORT_BYTES = 8 * 1024 * 1024
ALLOWED_REPORT_EXTENSIONS = (".pdf", ".txt", ".csv")


async def _session_cleanup_loop() -> None:
    """Purge expired sessions every hour."""
    while True:
        await asyncio.sleep(3600)
        removed = session_store.cleanup_expired()
        if removed:
            logger.info("[SESSION_CLEANUP] Removed %d expired sessions", removed)


app = FastAPI(title="AI Doctor API", version="0.3.0")


@app.on_event("startup")
async def startup() -> None:
    asyncio.create_task(_session_cleanup_loop())

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _check_critical_vitals(profile) -> tuple:
    """
    Deterministic gate for emergency-level vitals recorded at intake.
    Returns (is_critical: bool, assessment: TriageAssessment | None, message: str | None).
    Thresholds: BP ≥ 180/120, HR > 150, SpO2 < 92%, Temp ≥ 105°F.
    """
    flags = []
    care = []

    s = profile.systolic_bp
    d = profile.diastolic_bp
    if (s is not None and s >= 180) or (d is not None and d >= 120):
        flags.append(f"BP {s}/{d} mmHg — hypertensive crisis range")
        care.append("Call 911 or go to the ER immediately for hypertensive crisis evaluation.")
        care.append("Do not drive yourself.")

    hr = profile.heart_rate
    if hr is not None and hr > 150:
        flags.append(f"HR {hr} bpm — extreme tachycardia")
        care.append("Heart rate above 150 at rest requires immediate emergency evaluation.")

    o2 = profile.spo2
    if o2 is not None and o2 < 92:
        flags.append(f"SpO2 {o2}% — critical hypoxia")
        care.append("Oxygen saturation is critically low. Call 911 immediately.")

    t = profile.temperature_f
    if t is not None and t >= 105.0:
        flags.append(f"Temp {t}°F — dangerously high fever")
        care.append("Temperature above 105°F is a medical emergency. Go to the ER now.")

    if not flags:
        return False, None, None

    flag_text = "; ".join(flags)
    assessment = TriageAssessment(
        urgency=UrgencyLevel.emergency_now,
        confidence_level=ConfidenceLevel.high,
        likely_conditions=["Potential emergency — critical vital signs recorded at intake"],
        confidence_note=f"Critical vital sign(s) detected: {flag_text}.",
        reasoning=f"Deterministic escalation: {flag_text}.",
        recommended_next_step="Call 911 or go to the nearest emergency room immediately.",
        safety_rationale="Emergency-level vitals recorded at intake require immediate evaluation.",
        care_instructions=care or ["Call 911 or go to the ER immediately."],
        red_flag_hits=[RedFlagHit(code="critical_vitals", evidence=flag_text)],
    )
    message = (
        f"I need to flag something important right away. The vitals you recorded — {flag_text} — "
        "are in an emergency range. Please call 911 or have someone take you to the nearest "
        "emergency room immediately. Do not wait and do not drive yourself."
    )
    return True, assessment, message


# ---------------------------------------------------------------------------
# SSE helper
# ---------------------------------------------------------------------------

def _sse(obj: dict) -> str:
    return f"data: {_json.dumps(obj)}\n\n"


def _extract_pdf_text(content: bytes) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:
        raise HTTPException(
            status_code=501,
            detail="PDF extraction requires pypdf. Install with: pip install pypdf",
        ) from exc

    reader = PdfReader(io.BytesIO(content))
    texts = []
    for page in reader.pages:
        page_text = page.extract_text() or ""
        if page_text.strip():
            texts.append(page_text)
    return "\n".join(texts).strip()


def _clean_report_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def _parse_json_object_loose(text: str) -> dict:
    txt = (text or "").strip()
    if txt.startswith("```"):
        txt = txt.replace("```json", "").replace("```JSON", "").replace("```", "").strip()
    try:
        data = _json.loads(txt)
        return data if isinstance(data, dict) else {}
    except Exception:
        start = txt.find("{")
        end = txt.rfind("}")
        if start != -1 and end > start:
            try:
                data = _json.loads(txt[start:end + 1])
                return data if isinstance(data, dict) else {}
            except Exception:
                pass
    return {}


async def _answer_post_assessment_question(session, user_question: str) -> str:
    llm = get_clinician_llm()
    system = """You are a warm, clear clinician answering follow-up questions after triage.

Rules:
- Use plain language and a calm tone.
- Keep the answer practical and concise (<= 170 words).
- Do not invent new diagnoses beyond the completed triage summary.
- If the user reports new emergency symptoms (chest pain, severe shortness of breath, stroke signs, active self-harm intent), tell them to seek emergency care now.
- If the user asks about a specific lab value and it is not present in the uploaded reports, say you do not see it rather than guessing.
- End with: "Do you have any other questions I can help with?"

Return valid JSON only:
{"reply": "your response here"}
"""
    lab_block = ""
    if session.patient_profile.lab_reports:
        parts = []
        for idx, report in enumerate(session.patient_profile.lab_reports[:3], start=1):
            snippet = " ".join(report.content.split())[:1200]
            parts.append(f"{idx}. {report.filename}: {snippet}")
        lab_block = "\n\nUPLOADED LAB REPORTS:\n" + "\n".join(parts)

    context = (
        f"TRIAGE SUMMARY:\n{session.post_assessment_summary}"
        f"{lab_block}\n\n"
        f"PATIENT QUESTION:\n{user_question}"
    )
    try:
        raw = await asyncio.to_thread(
            llm.invoke_raw,
            system,
            [{"role": "user", "content": context}],
            450,
        )
        data = _parse_json_object_loose(raw)
        reply = str(data.get("reply", "")).strip() if isinstance(data, dict) else ""
        if reply:
            return reply
    except Exception as exc:
        logger.error("[POST_QA_ERROR] session=%s error=%s", session.session_id, exc)

    return (
        "I understand. Based on your current triage summary, I recommend following the same next-step plan, "
        "and seeking urgent care sooner if symptoms worsen. Do you have any other questions I can help with?"
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.get("/api/health")
async def health() -> dict:
    return {
        "api": "healthy",
        "service": settings.app_name,
        "env": settings.app_env,
        "llm": get_llm_status(),
        "tts": settings.tts_provider if settings.tts_enabled else "web_speech",
    }


@app.post("/api/reports/extract")
async def extract_report(file: UploadFile = File(...)) -> dict:
    name = file.filename or "uploaded_report"
    name_lower = name.lower()
    if not name_lower.endswith(ALLOWED_REPORT_EXTENSIONS):
        raise HTTPException(
            status_code=415,
            detail="Unsupported file type. Please upload PDF, TXT, or CSV.",
        )

    content = await file.read()
    if not content:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")
    if len(content) > MAX_REPORT_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Please keep each report under 8 MB.",
        )

    is_pdf = name_lower.endswith(".pdf") or (file.content_type or "").lower() == "application/pdf"

    if is_pdf:
        raw_text = _extract_pdf_text(content)
    else:
        try:
            raw_text = content.decode("utf-8", errors="ignore")
        except Exception:
            raw_text = ""

    cleaned = _clean_report_text(raw_text)
    if not cleaned:
        raise HTTPException(status_code=422, detail="Could not extract readable text from the report.")

    return {
        "filename": name,
        "content": cleaned[:8000],
    }


@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session(request: StartSessionRequest) -> StartSessionResponse:
    profile = request.patient_profile
    session = session_store.create_with_profile(profile)

    name_part = f", {profile.name}" if profile.name else ""
    welcome = (
        f"Hello{name_part}. I've reviewed the information you provided and I'm here to help "
        "assess your symptoms today. I'll ask a few focused questions — please answer as clearly "
        "as you can, and I'll guide you toward the right next steps."
    )

    # ── Critical vitals gate — flag emergency readings before intake begins ──
    is_critical, vital_assessment, vital_message = _check_critical_vitals(profile)
    if is_critical:
        session.messages.append(ChatMessage(role=Role.assistant, content=f"{welcome} {vital_message}"))
        logger.warning("[CRITICAL_VITALS_AT_START] session=%s", session.session_id)
        return StartSessionResponse(
            session_id=session.session_id,
            welcome_message=welcome,
            first_question=vital_message,
        )

    # ── LLM generates the opening question from full patient context ──
    llm = get_clinician_llm()
    cc = (profile.chief_complaint or "").strip()
    first_q = (
        f"I see you're here about {cc}. Can you tell me more about when this started and what it feels like?"
        if cc else
        "What's brought you in today? Please describe what's been bothering you."
    )
    try:
        result = await asyncio.to_thread(
            llm.process_turn,
            profile=profile,
            conversation=[],
            force_assess=False,
        )
        if result.get("action") == "ask_question":
            q = str(result.get("question", "")).strip()
            if len(q) > 8:
                first_q = q if q.endswith("?") else q + "?"
    except Exception as exc:
        logger.error("[LLM_START_ERROR] session=%s error=%s", session.session_id, exc)

    session.messages.append(ChatMessage(role=Role.assistant, content=f"{welcome} {first_q}"))

    return StartSessionResponse(
        session_id=session.session_id,
        welcome_message=welcome,
        first_question=first_q,
    )


@app.post("/api/session/{session_id}/turn")
async def process_turn(session_id: str, request: TurnRequest) -> StreamingResponse:
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.turn_count += 1
    session.messages.append(ChatMessage(role=Role.user, content=request.transcript))

    # After assessment is complete, only scan the current message for new red flags.
    # Scanning accumulated history would permanently block post-assessment Q&A for any
    # patient whose intake triggered a red flag (e.g. chest pain → every follow-up hijacked).
    if session.assessment_complete:
        red_flag_text = request.transcript
    else:
        red_flag_text = " ".join(m.content for m in session.messages if m.role == Role.user)
    red_flags = detect_red_flags(red_flag_text)

    async def stream():
      try:
        # ── Hard gate: deterministic red-flag check ──
        if red_flags:
            assessment, assistant_message = build_red_flag_assessment(red_flags)
            session.messages.append(ChatMessage(role=Role.assistant, content=assistant_message))
            logger.warning("[RED_FLAG] session=%s flags=%s", session_id, [rf.code for rf in red_flags])
            resp = TurnResponse(
                session_id=session_id,
                assistant_message=assistant_message,
                ask_follow_up=False,
                assessment=assessment,
            )
            yield _sse({"type": "result", "data": resp.model_dump()})
            return

        # ── Post-assessment follow-up Q&A mode ──
        if session.assessment_complete:
            yield _sse({"type": "progress", "stage": "thinking", "label": "Answering your follow-up question..."})
            follow_up_answer = await _answer_post_assessment_question(session, request.transcript)
            session.messages.append(ChatMessage(role=Role.assistant, content=follow_up_answer))
            resp = TurnResponse(
                session_id=session_id,
                assistant_message=follow_up_answer,
                ask_follow_up=True,
                follow_up_question="Do you have any other questions I can help with?",
            )
            yield _sse({"type": "result", "data": resp.model_dump()})
            return

        # ── LLM intake turn ──
        yield _sse({"type": "progress", "stage": "thinking", "label": "Reviewing your responses..."})

        min_turns = max(1, settings.min_turns_before_assessment)
        max_turns = max(min_turns, settings.max_turns_before_assessment)
        force_assess = session.turn_count >= max_turns

        llm = get_clinician_llm()
        llm_failed = False
        _t0 = time.monotonic()

        async def _call_intake_llm():
            return await asyncio.wait_for(
                asyncio.to_thread(
                    llm.process_turn,
                    profile=session.patient_profile,
                    conversation=session.messages,
                    force_assess=force_assess,
                ),
                timeout=35,
            )

        try:
            result = await _call_intake_llm()
        except Exception as exc1:
            elapsed1 = time.monotonic() - _t0
            logger.warning(
                "[LLM_RETRY] session=%s turn=%d elapsed=%.2fs first attempt failed: %s — retrying",
                session_id, session.turn_count, elapsed1, exc1,
            )
            await asyncio.sleep(1)
            try:
                result = await _call_intake_llm()
            except Exception as exc2:
                logger.error(
                    "[LLM_FAIL] session=%s turn=%d elapsed=%.2fs both attempts failed: %s",
                    session_id, session.turn_count, time.monotonic() - _t0, exc2,
                )
                result = {}
                llm_failed = True

        if not llm_failed:
            logger.info(
                "[LLM_LATENCY] session=%s turn=%d elapsed=%.2fs action=%s",
                session_id, session.turn_count, time.monotonic() - _t0, result.get("action", "?"),
            )

        # ── LLM failure mid-conversation: keep the conversation alive ──
        # If the LLM errored/timed-out and we haven't hit the forced-assess
        # ceiling yet, return a canned follow-up question rather than immediately
        # producing an "Undetermined" assessment, which is jarring and useless.
        if llm_failed and not force_assess:
            user_turns = sum(1 for m in session.messages if m.role == Role.user)
            fallback_qs = [
                "Can you tell me a bit more about how severe the symptoms are — are they interfering with your normal activities?",
                "Have you noticed anything that makes the symptoms better or worse?",
                "Is there anything else going on that you think might be related?",
            ]
            fallback_q = fallback_qs[min(user_turns - 1, len(fallback_qs) - 1)]
            session.messages.append(ChatMessage(role=Role.assistant, content=fallback_q))
            resp = TurnResponse(
                session_id=session_id,
                assistant_message=fallback_q,
                ask_follow_up=True,
                follow_up_question=fallback_q,
            )
            yield _sse({"type": "result", "data": resp.model_dump()})
            return

        action = result.get("action", "")

        # ── Follow-up question path ──
        if action == "ask_question" and not force_assess:
            question = str(result.get("question", "")).strip()
            if question and not question.endswith("?"):
                question += "?"
            if question and len(question) > 8:
                session.messages.append(ChatMessage(role=Role.assistant, content=question))
                resp = TurnResponse(
                    session_id=session_id,
                    assistant_message=question,
                    ask_follow_up=True,
                    follow_up_question=question,
                )
                yield _sse({"type": "result", "data": resp.model_dump()})
                return

        # ── Assessment path ──
        # Trigger pipeline whenever action == "assess" (intake_summary is optional —
        # the pipeline works from the conversation transcript directly).
        if action == "assess" or force_assess:
            # Pipeline path: progress events emitted per stage via callback
            queue: asyncio.Queue = asyncio.Queue()
            loop = asyncio.get_running_loop()

            def progress_cb(stage: str, label: str) -> None:
                asyncio.run_coroutine_threadsafe(
                    queue.put({"type": "progress", "stage": stage, "label": label}),
                    loop,
                )

            async def run_pipeline() -> None:
                try:
                    fa, pm = await asyncio.wait_for(
                        asyncio.to_thread(
                            run_assessment_pipeline,
                            profile=session.patient_profile,
                            conversation=session.messages,
                            intake_summary=result.get("intake_summary"),
                            invoke_fn=llm.invoke_raw,
                            prescribing_enabled=settings.prescribing_enabled,
                            progress_callback=progress_cb,
                            specialty=session.patient_profile.specialty,
                        ),
                        timeout=100,  # 3 pipeline stages × 30s each + buffer
                    )
                except asyncio.TimeoutError:
                    logger.error("[PIPELINE_TIMEOUT] session=%s pipeline exceeded 120s", session_id)
                    fa, pm = {}, ""
                except Exception as exc:
                    logger.error("[PIPELINE_ERROR] session=%s error=%s", session_id, exc)
                    fa, pm = {}, ""
                await queue.put({"__done__": True, "result": (fa, pm)})

            asyncio.create_task(run_pipeline())

            while True:
                event = await queue.get()
                if "__done__" in event:
                    final_assessment, patient_message = event["result"]
                    break
                yield _sse(event)

            merged = {**final_assessment, "assistant_message": patient_message}
            assessment, assistant_message = apply_safety_policy(
                {"assessment": merged},
                conservative_mode=settings.conservative_mode,
                prescribing_enabled=settings.prescribing_enabled,
            )
        else:
            # Fallback path: NoopLLM or intake LLM returned old full-assessment format
            assessment, assistant_message = apply_safety_policy(
                result,
                conservative_mode=settings.conservative_mode,
                prescribing_enabled=settings.prescribing_enabled,
            )

        logger.info(
            "[TRIAGE] session=%s urgency=%s confidence=%s conditions=%s",
            session_id,
            assessment.urgency,
            assessment.confidence_level,
            assessment.likely_conditions,
        )

        session.assessment_complete = True
        session.post_assessment_summary = (
            f"Urgency: {assessment.urgency}. "
            f"Possible conditions: {', '.join(assessment.likely_conditions)}. "
            f"Next step: {assessment.recommended_next_step}. "
            f"Care instructions: {'; '.join(assessment.care_instructions[:3])}."
        )
        handoff = (
            f"{assistant_message}\n\n"
            "If you have any additional questions about this plan, feel free to ask and I can walk you through them."
        )
        session.messages.append(ChatMessage(role=Role.assistant, content=handoff))
        resp = TurnResponse(
            session_id=session_id,
            assistant_message=handoff,
            ask_follow_up=True,
            follow_up_question="Do you have any additional questions?",
            assessment=assessment,
        )
        yield _sse({"type": "result", "data": resp.model_dump()})

      except Exception as exc:
        logger.error("[STREAM_ERROR] session=%s error=%s", session_id, exc, exc_info=True)
        yield _sse({"type": "error", "message": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------------
# TTS endpoint — provider selected by TTS_PROVIDER env var (openai | google).
# Frontend probes this once on session start; falls back to Web Speech if 501/503.
# ---------------------------------------------------------------------------

class _TTSRequest(BaseModel):
    text: str


async def _tts_openai(text: str) -> bytes:
    if not settings.openai_api_key:
        raise HTTPException(status_code=501, detail="TTS_PROVIDER=openai but OPENAI_API_KEY not set")
    from openai import OpenAI
    client = OpenAI(api_key=settings.openai_api_key)
    resp = await asyncio.to_thread(
        lambda: client.audio.speech.create(
            model="tts-1",
            voice="nova",
            input=text[:4096],
        )
    )
    return resp.content


async def _tts_google(text: str) -> bytes:
    try:
        from google.cloud import texttospeech as tts
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="google-cloud-texttospeech not installed — run: pip install google-cloud-texttospeech",
        )
    def _call() -> bytes:
        client = tts.TextToSpeechClient()
        response = client.synthesize_speech(
            input=tts.SynthesisInput(text=text[:5000]),
            voice=tts.VoiceSelectionParams(
                language_code="en-US",
                name=settings.google_tts_voice,
            ),
            audio_config=tts.AudioConfig(
                audio_encoding=tts.AudioEncoding.MP3,
                speaking_rate=0.95,   # slightly slower — clearer for medical context
                pitch=0.0,
            ),
        )
        return response.audio_content

    return await asyncio.to_thread(_call)


@app.post("/api/tts")
async def text_to_speech(request: _TTSRequest) -> Response:
    if not settings.tts_enabled:
        raise HTTPException(status_code=501, detail="TTS disabled: set TTS_ENABLED=true")
    text = request.text.strip()
    if not text:
        return Response(content=b"", media_type="audio/mpeg")
    try:
        if settings.tts_provider == "google":
            audio = await _tts_google(text)
        else:
            audio = await _tts_openai(text)
        return Response(content=audio, media_type="audio/mpeg")
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("[TTS_ERROR] provider=%s error=%s", settings.tts_provider, exc)
        raise HTTPException(status_code=503, detail=f"TTS error: {exc}")
