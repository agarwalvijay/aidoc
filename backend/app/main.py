import json
import logging

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .config import settings
from .intake import next_follow_up
from .llm import get_clinician_llm, get_llm_status
from .models import (
    ChatMessage,
    ConfidenceLevel,
    Role,
    StartSessionResponse,
    TurnRequest,
    TurnResponse,
    UrgencyLevel,
)
from .red_flags import detect_red_flags
from .session_store import session_store
from .triage import build_assessment

logger = logging.getLogger("aidoc")


app = FastAPI(title="AI Doctor API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _joined_user_transcript(session_messages: list[ChatMessage], latest_transcript: str) -> str:
    """Build a conversation-level user symptom narrative for triage and LLM prompts."""
    user_turns = [m.content.strip() for m in session_messages if m.role == Role.user and m.content.strip()]
    if not user_turns:
        user_turns = [latest_transcript]
    return " | ".join(user_turns)


def _infer_sensitive_specialist(conditions: list[str]) -> str:
    joined = " ".join([c.lower() for c in conditions])
    mapping = [
        ("cancer", "Oncologist"),
        ("tumor", "Oncologist"),
        ("breast lump", "Breast Surgeon or Oncologist"),
        ("blood in stool", "Gastroenterologist"),
        ("blood in urine", "Urologist"),
        ("suicidal", "Psychiatrist"),
        ("depression", "Psychiatrist"),
        ("pregnancy", "Obstetrician-Gynecologist"),
        ("pelvic pain", "Obstetrician-Gynecologist"),
        ("stroke", "Neurologist"),
        ("seizure", "Neurologist"),
        ("chest pain", "Cardiologist"),
        ("heart", "Cardiologist"),
    ]
    for key, specialist in mapping:
        if key in joined:
            return specialist
    return "Relevant Specialist"


@app.get("/api/health")
async def health() -> dict:
    return {
        "api": "healthy",
        "service": settings.app_name,
        "env": settings.app_env,
        "llm": get_llm_status(),
    }


@app.post("/api/session/start", response_model=StartSessionResponse)
async def start_session() -> StartSessionResponse:
    session = session_store.create()

    branding = "AI Doctor: Voice-first clinical triage assistant"
    welcome = "Welcome. I can help triage symptoms and recommend next steps safely."
    first_question = "What symptoms or conditions are you experiencing today?"

    session.messages.append(ChatMessage(role=Role.assistant, content=welcome))
    session.asked_questions.append(first_question)

    return StartSessionResponse(
        session_id=session.session_id,
        branding_message=branding,
        welcome_message=welcome,
        first_question=first_question,
        play_chime=True,
    )


@app.post("/api/session/{session_id}/turn", response_model=TurnResponse)
async def process_turn(session_id: str, request: TurnRequest) -> TurnResponse:
    session = session_store.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    session.turn_count += 1
    session.messages.append(ChatMessage(role=Role.user, content=request.transcript))
    llm = get_clinician_llm()
    transcript_context = _joined_user_transcript(session.messages, request.transcript)

    red_flags = detect_red_flags(transcript_context)

    # Ask up to 3 follow-up questions before final assessment, unless red flags trigger immediate escalation.
    should_finalize = bool(red_flags) or session.turn_count >= 4

    if not should_finalize:
        question = None
        try:
            question = llm.generate_follow_up(transcript_context, session.asked_questions)
        except Exception:
            question = None
        if not question:
            question = next_follow_up(session.asked_questions)
        if question:
            session.asked_questions.append(question)
            session.messages.append(ChatMessage(role=Role.assistant, content=question))
            return TurnResponse(
                session_id=session_id,
                assistant_message=question,
                ask_follow_up=True,
                follow_up_question=question,
            )

    assessment = build_assessment(
        transcript=transcript_context,
        red_flags=red_flags,
        conservative_mode=settings.conservative_mode,
    )

    enhancement = {}
    if not red_flags:
        try:
            enhancement = llm.generate_assessment_enhancement(
                transcript_context,
                assessment.model_dump(),
            )
        except Exception:
            enhancement = {}

        if isinstance(enhancement.get("confidence_level"), str):
            value = enhancement.get("confidence_level", "").strip().lower()
            if value in ("high", "medium", "low"):
                assessment.confidence_level = ConfidenceLevel(value)

        if isinstance(enhancement.get("likely_conditions"), list):
            safe_conditions = [str(c).strip() for c in enhancement["likely_conditions"] if str(c).strip()]
            if safe_conditions:
                assessment.likely_conditions = safe_conditions[:3]

        if isinstance(enhancement.get("confidence_note"), str) and enhancement["confidence_note"].strip():
            assessment.confidence_note = enhancement["confidence_note"].strip()

        if isinstance(enhancement.get("care_instructions"), list):
            care = [str(c).strip() for c in enhancement["care_instructions"] if str(c).strip()]
            if care:
                assessment.care_instructions = care[:6]

        if isinstance(enhancement.get("prescription_guidance"), list):
            rx = [str(c).strip() for c in enhancement["prescription_guidance"] if str(c).strip()]
            if rx:
                assessment.prescription_guidance = rx[:4]

        if isinstance(enhancement.get("recommended_next_step"), str) and enhancement["recommended_next_step"].strip():
            assessment.recommended_next_step = enhancement["recommended_next_step"].strip()

        if isinstance(enhancement.get("sensitive_condition"), bool):
            assessment.sensitive_condition = enhancement["sensitive_condition"]

        if isinstance(enhancement.get("specialist_type"), str) and enhancement["specialist_type"].strip():
            assessment.specialist_type = enhancement["specialist_type"].strip()

    if assessment.sensitive_condition and not assessment.specialist_type:
        assessment.specialist_type = _infer_sensitive_specialist(assessment.likely_conditions)

    if assessment.confidence_level == ConfidenceLevel.low and assessment.urgency == UrgencyLevel.self_care_monitor:
        assessment.recommended_next_step = "Please follow up with a clinician for in-person evaluation in 24-72 hours."
    if assessment.sensitive_condition and assessment.specialist_type:
        assessment.recommended_next_step = (
            "Please arrange prompt follow-up with a {0}. If symptoms worsen, seek urgent care immediately."
        ).format(assessment.specialist_type)

    assistant_message = enhancement.get("assistant_message")
    if not isinstance(assistant_message, str) or not assistant_message.strip():
        condition_text = ", ".join(assessment.likely_conditions) if assessment.likely_conditions else "an unclear condition"
        assistant_message = "From what you shared, you may have {0}. ".format(condition_text)
        if assessment.confidence_level == ConfidenceLevel.high:
            assistant_message += "I have relatively high confidence in this triage direction. "
        elif assessment.confidence_level == ConfidenceLevel.medium:
            assistant_message += "I have moderate confidence, so monitor symptoms closely. "
        else:
            assistant_message += "My confidence is limited from remote information alone. "
        assistant_message += "Next step: {0}".format(assessment.recommended_next_step)

    if assessment.care_instructions:
        assistant_message += "\n\nCare instructions:\n- " + "\n- ".join(assessment.care_instructions)
    if assessment.prescription_guidance:
        assistant_message += "\n\nMedication guidance (confirm with a licensed clinician):\n- " + "\n- ".join(
            assessment.prescription_guidance
        )
    if assessment.sensitive_condition and assessment.specialist_type:
        assistant_message += "\n\nSpecialist follow-up: {0}".format(assessment.specialist_type)

    logger.info(
        "[LLM_TRIAGE_DEBUG] %s",
        json.dumps(
            {
                "session_id": session_id,
                "transcript_context": transcript_context,
                "deterministic_assessment": assessment.model_dump(),
                "llm_enhancement": enhancement,
                "final_assistant_message": assistant_message,
            },
            ensure_ascii=True,
        ),
    )

    session.messages.append(ChatMessage(role=Role.assistant, content=assistant_message))

    return TurnResponse(
        session_id=session_id,
        assistant_message=assistant_message,
        ask_follow_up=False,
        assessment=assessment,
    )
