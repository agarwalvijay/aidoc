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

_CONDITION_OPENERS: dict[str, str] = {
    "uri": "How long have you had these symptoms, and do you have a fever?",
    "gi": "When did this start, and have you had any nausea, vomiting, or diarrhea?",
    "uti": "How long have you had the burning sensation, and are you also experiencing increased frequency or urgency to urinate?",
    "rash": "Where on your body is the rash, and how would you describe it — is it raised, flat, blistering, or scaly?",
    "back": "Did this start suddenly or gradually, and does the pain radiate anywhere down your legs?",
    "headache": "How quickly did the headache come on, and on a scale of 0 to 10 how severe is it right now?",
    "mental": "Over the past two weeks, have you had little interest or pleasure in doing things you usually enjoy?",
    "htn": "What have your recent blood pressure readings been at home, and are you taking your medications as prescribed?",
    "dm": "How have your blood glucose readings been lately, and are you taking your diabetes medications as prescribed?",
}

_CONDITION_KEYWORDS: list[tuple[str, str]] = [
    ("uri", ["cough", "cold", "sore throat", "runny nose", "congestion", "uri", "flu", "respiratory", "nasal"]),
    ("gi", ["stomach", "nausea", "vomit", "diarrhea", "abdominal", "gi", "bowel", "belly", "cramp", "indigestion", "heartburn"]),
    ("uti", ["urine", "urination", "burning", "uti", "bladder", "dysuria", "frequency", "urgency urinating"]),
    ("rash", ["rash", "skin", "itch", "hives", "redness", "blister", "bumps on skin"]),
    ("back", ["back pain", "back ache", "lower back", "spine", "lumbar", "back is hurting"]),
    ("headache", ["headache", "migraine", "head pain", "head is pounding", "head hurts"]),
    ("mental", ["anxious", "anxiety", "depressed", "depression", "mood", "mental health", "panic", "sad", "hopeless"]),
    ("htn", ["blood pressure", "hypertension", "bp check", "htn"]),
    ("dm", ["diabetes", "blood sugar", "glucose", "diabetic", "dm", "insulin"]),
]


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


def _classify_complaint(complaint: str) -> str:
    lower = complaint.lower()
    for condition, keywords in _CONDITION_KEYWORDS:
        if any(kw in lower for kw in keywords):
            return condition
    return ""


def _first_question(profile) -> str:
    cc = (profile.chief_complaint or "").strip()
    if not cc:
        return "What symptoms are bringing you in today? Please describe what's been bothering you."

    condition = _classify_complaint(cc)
    opener = _CONDITION_OPENERS.get(condition)
    if opener:
        return f"You mentioned {cc}. {opener}"

    return (
        f"You mentioned {cc}. "
        "How long have you been experiencing this, and how would you describe the severity on a scale of 0 to 10?"
    )


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
    first_q = _first_question(profile)  # deterministic fallback
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
# TTS endpoint — uses OpenAI TTS if OPENAI_API_KEY is configured.
# Frontend probes this once; falls back to Web Speech API if unavailable.
# ---------------------------------------------------------------------------

class _TTSRequest(BaseModel):
    text: str


@app.post("/api/tts")
async def text_to_speech(request: _TTSRequest) -> Response:
    if not settings.openai_api_key:
        raise HTTPException(status_code=501, detail="TTS not configured: OPENAI_API_KEY not set")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=settings.openai_api_key)
        tts_response = await asyncio.to_thread(
            lambda: client.audio.speech.create(
                model="tts-1",
                voice="nova",
                input=request.text[:4096],
            )
        )
        return Response(content=tts_response.content, media_type="audio/mpeg")
    except Exception as exc:
        logger.error("[TTS_ERROR] %s", exc)
        raise HTTPException(status_code=503, detail="TTS service error")
