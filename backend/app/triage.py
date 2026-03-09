from .models import TriageAssessment, UrgencyLevel, RedFlagHit, ConfidenceLevel


def build_assessment(
    transcript: str,
    red_flags: list[RedFlagHit],
    conservative_mode: bool = True,
) -> TriageAssessment:
    text = transcript.lower()

    if red_flags:
        return TriageAssessment(
            urgency=UrgencyLevel.emergency_now,
            confidence_level=ConfidenceLevel.high,
            likely_conditions=["Potential high-risk condition requiring immediate care"],
            confidence_note="Safety-first escalation due to red-flag symptom(s).",
            recommended_next_step="Seek emergency care now or call emergency services.",
            safety_rationale="Detected one or more high-risk red-flag signals.",
            red_flag_hits=red_flags,
        )

    if any(k in text for k in ["severe", "very bad", "worsening", "fainting"]):
        return TriageAssessment(
            urgency=UrgencyLevel.urgent_today,
            confidence_level=ConfidenceLevel.medium,
            likely_conditions=["Acute condition needing same-day evaluation"],
            confidence_note="Moderate confidence; symptoms may require urgent in-person exam.",
            recommended_next_step="Visit urgent care today or telemedicine urgent evaluation.",
            safety_rationale="Severity/worsening signal detected.",
        )

    if conservative_mode:
        return TriageAssessment(
            urgency=UrgencyLevel.specialist_soon,
            confidence_level=ConfidenceLevel.low,
            likely_conditions=["Non-emergent but unresolved condition"],
            confidence_note="Conservative routing due to limited remote exam data.",
            recommended_next_step="Schedule a clinician or specialist follow-up within 3-7 days.",
            safety_rationale="Conservative mode favors escalation over false reassurance.",
        )

    return TriageAssessment(
        urgency=UrgencyLevel.self_care_monitor,
        confidence_level=ConfidenceLevel.medium,
        likely_conditions=["Likely minor self-limited illness"],
        confidence_note="Low-to-moderate confidence.",
        recommended_next_step="Self-care and monitor symptoms for 24-48 hours.",
        safety_rationale="No red flags and no severe progression signs reported.",
    )
