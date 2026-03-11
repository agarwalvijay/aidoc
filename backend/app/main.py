import asyncio
import json as _json
import logging

from fastapi import FastAPI, HTTPException
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

app = FastAPI(title="AI Doctor API", version="0.3.0")

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

    all_user_text = " ".join(m.content for m in session.messages if m.role == Role.user)
    red_flags = detect_red_flags(all_user_text)

    async def stream():
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

        # ── LLM intake turn ──
        yield _sse({"type": "progress", "stage": "thinking", "label": "Reviewing your responses..."})

        min_turns = max(1, settings.min_turns_before_assessment)
        max_turns = max(min_turns, settings.max_turns_before_assessment)
        force_assess = session.turn_count >= max_turns

        llm = get_clinician_llm()
        try:
            result = await asyncio.to_thread(
                llm.process_turn,
                profile=session.patient_profile,
                conversation=session.messages,
                force_assess=force_assess,
            )
        except Exception as exc:
            logger.error("[LLM_ERROR] session=%s error=%s", session_id, exc)
            result = {}

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
        intake_summary = result.get("intake_summary")

        if intake_summary is not None:
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
                    fa, pm = await asyncio.to_thread(
                        run_assessment_pipeline,
                        profile=session.patient_profile,
                        conversation=session.messages,
                        intake_summary=intake_summary,
                        invoke_fn=llm.invoke_raw,
                        prescribing_enabled=settings.prescribing_enabled,
                        progress_callback=progress_cb,
                        specialty=session.patient_profile.specialty,
                    )
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

        session.messages.append(ChatMessage(role=Role.assistant, content=assistant_message))
        resp = TurnResponse(
            session_id=session_id,
            assistant_message=assistant_message,
            ask_follow_up=False,
            assessment=assessment,
        )
        yield _sse({"type": "result", "data": resp.model_dump()})

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
