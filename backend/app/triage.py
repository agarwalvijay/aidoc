"""
Safety policy layer.

This module does NOT make clinical decisions — the LLM does.
Its job is to:
  1. Build deterministic emergency assessments when red flags are detected
     (bypassing the LLM entirely).
  2. Convert raw LLM JSON into a validated TriageAssessment, enforcing
     output safety rules (no direct prescribing, urgency floors, etc.).
"""

import logging
from typing import Any, Dict, List, Tuple

from .models import ConfidenceLevel, RedFlagHit, TriageAssessment, UrgencyLevel

logger = logging.getLogger("uvicorn.error")


def build_red_flag_assessment(red_flags: List[RedFlagHit]) -> Tuple[TriageAssessment, str]:
    """Deterministic emergency response for detected red-flag conditions."""
    codes = {rf.code for rf in red_flags}

    if "suicidality" in codes:
        assessment = TriageAssessment(
            urgency=UrgencyLevel.emergency_now,
            confidence_level=ConfidenceLevel.high,
            likely_conditions=["Mental health crisis — suicidal ideation"],
            confidence_note="Immediate mental health emergency response required.",
            reasoning="Patient expressed suicidal ideation.",
            recommended_next_step="Call or text 988 (Suicide & Crisis Lifeline) or go to the nearest ER now.",
            safety_rationale="Suicidal ideation is a psychiatric emergency.",
            care_instructions=[
                "Call or text 988 (Suicide & Crisis Lifeline) — available 24/7.",
                "If you have a plan or means, call 911 or go to the nearest ER immediately.",
                "Do not be alone right now — stay with a trusted person.",
            ],
            sensitive_condition=True,
            specialist_type="Psychiatry / Emergency Mental Health",
            red_flag_hits=red_flags,
        )
        message = (
            "I'm very concerned about what you've shared, and I want you to know you're not alone. "
            "Please reach out for help right now. Call or text 988 — that's the Suicide and Crisis "
            "Lifeline, available 24/7. If you're in immediate danger, call 911 or go to your nearest "
            "emergency room. Please don't be alone right now."
        )
        return assessment, message

    # Specific messaging for cardiac events
    if "cardiac" in codes:
        assessment = TriageAssessment(
            urgency=UrgencyLevel.emergency_now,
            confidence_level=ConfidenceLevel.high,
            likely_conditions=["Possible cardiac emergency — requires immediate evaluation"],
            confidence_note="Chest pain / cardiac symptoms require emergency evaluation.",
            reasoning=f"Red flag(s) detected: {', '.join(codes)}.",
            recommended_next_step="Call 911 immediately — do not drive yourself.",
            safety_rationale="Cardiac red flags detected.",
            care_instructions=[
                "Call 911 immediately.",
                "Chew an aspirin (325mg) if available and you are not allergic to aspirin.",
                "Sit or lie down and try to stay calm while help arrives.",
                "Do not drive yourself to the hospital.",
            ],
            red_flag_hits=red_flags,
        )
        message = (
            "What you're describing sounds like it could be a cardiac emergency. "
            "Please call 911 right now — do not drive yourself. "
            "If you have aspirin and are not allergic, chew one 325mg tablet while waiting for help. "
            "Stay as calm as possible and remain seated or lying down."
        )
        return assessment, message

    # Generic emergency
    assessment = TriageAssessment(
        urgency=UrgencyLevel.emergency_now,
        confidence_level=ConfidenceLevel.high,
        likely_conditions=["Potential emergency condition — requires immediate evaluation"],
        confidence_note="Safety-first escalation: high-risk symptom(s) detected.",
        reasoning=f"Red flag(s) detected: {', '.join(codes)}.",
        recommended_next_step="Call 911 or go to the nearest emergency room immediately.",
        safety_rationale="One or more emergency red-flag signals detected.",
        care_instructions=[
            "Call 911 or have someone take you to the ER immediately.",
            "Do not drive yourself.",
            "Stay as calm as possible while waiting for help.",
        ],
        red_flag_hits=red_flags,
    )
    message = (
        "Based on what you've described, I'm detecting symptoms that need immediate emergency attention. "
        "Please call 911 or have someone take you to the nearest emergency room right away. "
        "Do not drive yourself."
    )
    return assessment, message


def apply_safety_policy(
    llm_result: Dict[str, Any],
    conservative_mode: bool = True,
    prescribing_enabled: bool = False,
) -> Tuple[TriageAssessment, str]:
    """
    Convert a raw LLM assessment JSON into a validated TriageAssessment.
    Enforces safety floors: urgency minimums, output sanitization, emergency appends.
    Returns (TriageAssessment, assistant_message_str).
    """
    raw = llm_result.get("assessment", {})

    # --- Urgency ---
    try:
        urgency = UrgencyLevel(raw.get("urgency", "specialist_soon"))
    except ValueError:
        urgency = UrgencyLevel.specialist_soon  # conservative fallback

    # --- Confidence ---
    try:
        confidence = ConfidenceLevel(raw.get("confidence_level", "medium"))
    except ValueError:
        confidence = ConfidenceLevel.medium

    # Deterministic urgency sanity check:
    # emergency_now without high confidence is almost certainly an LLM over-escalation.
    # True emergencies (MI, stroke, PE, suicidality) are assessed with high confidence;
    # routine presentations incorrectly flagged emergency come through as medium/low.
    # Note: genuine life-threatening emergencies are caught by the deterministic red_flag
    # gate in main.py BEFORE this code runs — so reaching here means no hard red flag fired.
    if urgency == UrgencyLevel.emergency_now and confidence != ConfidenceLevel.high:
        logger.warning(
            "[URGENCY_SANITY] Downgrading emergency_now (confidence=%s) → urgent_today. "
            "Reasoning: %s",
            confidence,
            str(raw.get("reasoning", ""))[:200],
        )
        urgency = UrgencyLevel.urgent_today

    # Conservative mode: never allow self_care_monitor when confidence is low
    if conservative_mode and confidence == ConfidenceLevel.low:
        if urgency == UrgencyLevel.self_care_monitor:
            urgency = UrgencyLevel.specialist_soon

    # --- Conditions ---
    likely = [str(c).strip() for c in raw.get("likely_conditions", []) if str(c).strip()][:3]
    if not likely:
        likely = ["Undetermined — in-person evaluation recommended"]

    # --- Care instructions ---
    care = [str(c).strip() for c in raw.get("care_instructions", []) if str(c).strip()][:8]
    if not care:
        care = [
            "Monitor your symptoms closely.",
            "Seek care sooner if symptoms worsen significantly.",
        ]

    # --- Prescription guidance ---
    def _rx_item_to_str(item) -> str:
        """Convert a prescription item to a readable string regardless of shape."""
        if isinstance(item, dict):
            parts = [
                item.get("medication", ""),
                item.get("dose", ""),
                item.get("route", ""),
                item.get("frequency", ""),
                f"for {item['duration']}" if item.get("duration") else "",
                f"— {item['instruction']}" if item.get("instruction") else "",
            ]
            return " ".join(p for p in parts if p).strip()
        return str(item).strip()

    rx_safe = []
    for item in raw.get("prescription_guidance", []):
        text = _rx_item_to_str(item)
        if not text:
            continue
        if not prescribing_enabled:
            lower = text.lower()
            if any(kw in lower for kw in ("you should take", "take ", "prescribed", "mg twice", "mg once", "mg daily", "mg by mouth")):
                text = f"Ask a licensed clinician about: {text}"
        rx_safe.append(text)

    # --- Specialist ---
    specialist = raw.get("specialist_type") or None
    if isinstance(specialist, str):
        specialist = specialist.strip() or None

    sensitive = bool(raw.get("sensitive_condition", False))

    # --- Recommended next step ---
    next_step = str(raw.get("recommended_next_step", "")).strip()
    if not next_step:
        next_step = "Schedule a clinician follow-up for proper evaluation."

    # --- Assistant message ---
    assistant_message = str(raw.get("assistant_message", "")).strip()
    if not assistant_message:
        condition_text = ", ".join(likely)
        assistant_message = (
            f"Based on what you've shared, your symptoms may be consistent with {condition_text}. "
            f"Next step: {next_step}"
        )

    # Safety appends
    if urgency == UrgencyLevel.emergency_now and "911" not in assistant_message and "emergency" not in assistant_message.lower():
        assistant_message += "\n\nThis requires emergency care — please call 911 or go to the ER immediately."

    return (
        TriageAssessment(
            urgency=urgency,
            confidence_level=confidence,
            likely_conditions=likely,
            confidence_note=str(raw.get("confidence_note", "")).strip(),
            reasoning=str(raw.get("reasoning", "")).strip(),
            reasoning_summary=str(raw.get("patient_reasoning", "")).strip(),
            recommended_next_step=next_step,
            safety_rationale="LLM-driven triage with deterministic safety policy applied.",
            care_instructions=care,
            prescription_guidance=rx_safe[:4],
            sensitive_condition=sensitive,
            specialist_type=specialist,
            red_flag_hits=[],
        ),
        assistant_message,
    )
