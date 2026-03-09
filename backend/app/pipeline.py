"""
3-stage clinical assessment pipeline: assess → critic → writer

Runs once at the end of intake when the intake_llm decides it has gathered
sufficient information (or when force_assess is triggered).

Stage 1 — assessment_llm
  Input:  patient profile + structured intake summary + full conversation
  Job:    clinical reasoning — differential, urgency, care plan
  Output: structured assessment JSON (no patient-facing language)

Stage 2 — critic_llm
  Input:  patient profile + conversation + proposed assessment
  Job:    safety review — catches missed red flags, wrong urgency, drug
          contraindications, overconfidence
  Output: corrected/approved assessment + list of issues found

Stage 3 — writer_llm
  Input:  approved assessment + patient profile
  Job:    plain-language patient communication only
  Output: patient_message string (≤180 words, empathetic, jargon-free)

All three stages call `invoke_fn(system_prompt, messages) -> str`, which is
the raw LLM caller exposed by whichever provider is active.  This keeps the
pipeline provider-agnostic.
"""

import json
import logging
import re
from typing import Any, Callable, Dict, List, Optional, Tuple

from .models import ChatMessage, PatientProfile, Role

logger = logging.getLogger("uvicorn.error")

# ---------------------------------------------------------------------------
# Stage prompts
# ---------------------------------------------------------------------------

ASSESSMENT_SYSTEM_PROMPT = """\
You are a clinical reasoning engine performing evidence-based triage assessment.
Your sole job is structured clinical reasoning — do NOT write patient-facing language.

Given the patient profile, the completed intake summary, and the conversation
transcript, produce a structured triage assessment.

Return valid JSON only:
{
  "likely_conditions": ["most likely", "second", "third"],
  "primary_impression": "brief clinical impression with key supporting evidence",
  "urgency": "emergency_now|urgent_today|specialist_soon|self_care_monitor",
  "confidence_level": "high|medium|low",
  "confidence_note": "what supports or limits confidence",
  "recommended_next_step": "specific, actionable instruction",
  "care_instructions": ["...", "..."],
  "prescription_guidance": ["Drug name dose route frequency duration. Key instruction. When to stop/seek care."],
  "sensitive_condition": false,
  "specialist_type": null,
  "reasoning": "brief differential reasoning"
}

Rules:
- Base the assessment strictly on gathered information; do not assume
- Abnormal vitals must be addressed in the assessment even if the chief complaint seems unrelated
- When information is incomplete, reflect that as lower confidence
- When uncertain between urgency levels, choose the higher one
- If prescribing is not authorized, leave prescription_guidance as []
- If prescribing is authorized, check allergies and drug interactions before listing any medication

Urgency thresholds:
  emergency_now    — red flags, suspected MI/stroke/PE/sepsis/suicidality, SpO2 < 92%, BP ≥ 180/120
  urgent_today     — severe/worsening symptoms, suspected pyelonephritis, uncontrolled BP > 160/100,
                     HR > 120, significant mood symptoms, new neuro symptoms
  specialist_soon  — needs in-person evaluation within 2-5 days
  self_care_monitor — high confidence benign self-limited condition with clear return precautions

Every assessment must include at least 2 specific return-precaution items in care_instructions.
"""

CRITIC_SYSTEM_PROMPT = """\
You are a senior emergency physician performing a mandatory safety review of a
triage assessment before it is delivered to the patient. Your role is quality
control — catch errors before they cause harm.

Review the proposed assessment for:
1. Missed red flags — does any symptom in the conversation warrant emergency escalation?
2. Vital sign alignment — does the assessment adequately address abnormal vitals?
   If BP ≥ 160/100 or HR > 120 were recorded, the assessment must address this.
3. Urgency appropriateness — is the urgency level justified by the clinical data?
   You may UPGRADE urgency but require strong evidence to DOWNGRADE.
4. Prescription safety — for every medication in prescription_guidance, verify:
   - Not on the patient's allergy list
   - No significant interaction with current medications
   - No contraindication from PMH (NSAIDs + peptic ulcer, nitrofurantoin + renal failure, etc.)
5. Confidence calibration — is the stated confidence honest given what was gathered?
6. Sensitive condition detection — cancer alarm symptoms, mental health crisis, obstetric
   emergencies should be flagged with sensitive_condition=true and specialist_type set.

Return valid JSON only:
{
  "issues_found": ["description of each issue you corrected"],
  "assessment": { ...final corrected assessment, identical structure to input (same field names: likely_conditions, confidence_level, confidence_note, reasoning, etc.)... }
}

If no issues: return issues_found: [] and assessment identical to the input.
If issues found: correct them in the assessment and explain each in issues_found.
"""

WRITER_SYSTEM_PROMPT = """\
You write patient-facing communications based on clinical triage assessments.
Your only job is clear, empathetic, plain-language writing — no clinical reasoning.

Requirements:
- Plain language: no medical jargon ("infection" not "infectious etiology",
  "blood pressure" not "BP", "heart rhythm problem" not "SVT")
- Warm, calm tone — patients are anxious
- Open by briefly acknowledging their main symptom
- Use hedged language: "may suggest", "consistent with", "appears to be"
- State the next step clearly and specifically (not vaguely)
- Mention 2-3 key care points inline, naturally (not as a bulleted list in the message itself)
- Maximum 180 words
- For emergencies: lead with the emergency action first, not the diagnosis
- For sensitive conditions: be compassionate and clear about specialist referral

Return valid JSON only:
{
  "patient_message": "..."
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_stage_json(text: str) -> Dict[str, Any]:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start : end + 1])
                return result if isinstance(result, dict) else {}
            except json.JSONDecodeError:
                pass
    return {}


def _format_profile_block(profile: PatientProfile) -> str:
    lines = []
    lines.append(f"Age: {profile.age or 'not provided'}, Sex: {profile.sex.value if profile.sex else 'not provided'}")
    if profile.pregnant is not None:
        lines.append(f"Pregnant: {'yes' if profile.pregnant else 'no'}")
    lines.append(f"PMH: {', '.join(profile.pmh_conditions) or 'none reported'}")
    lines.append(f"Medications: {', '.join(profile.current_medications) or 'none reported'}")
    lines.append(f"Allergies: {', '.join(profile.allergies) or 'none reported'}")

    vitals = []
    if profile.temperature_f is not None:
        vitals.append(f"Temp {profile.temperature_f}°F{'  ⚠ FEVER' if profile.temperature_f >= 100.4 else ''}")
    if profile.systolic_bp is not None and profile.diastolic_bp is not None:
        s, d = profile.systolic_bp, profile.diastolic_bp
        flag = "  ⚠ ELEVATED" if s >= 140 or d >= 90 else ""
        vitals.append(f"BP {s}/{d}{flag}")
    if profile.heart_rate is not None:
        flag = "  ⚠ TACHYCARDIA" if profile.heart_rate > 100 else ""
        vitals.append(f"HR {profile.heart_rate}{flag}")
    if profile.spo2 is not None:
        flag = "  ⚠ LOW" if profile.spo2 < 95 else ""
        vitals.append(f"SpO2 {profile.spo2}%{flag}")
    lines.append(f"Vitals: {', '.join(vitals) or 'not measured'}")
    return "\n".join(lines)


def _format_conversation_block(conversation: List[ChatMessage]) -> str:
    lines = []
    for msg in conversation:
        speaker = "Patient" if msg.role == Role.user else "Clinician"
        lines.append(f"{speaker}: {msg.content}")
    return "\n".join(lines)


def _call_stage(
    invoke_fn: Callable[[str, List[Dict]], str],
    system: str,
    user_content: str,
    stage_name: str,
    max_tokens: int = 1024,
) -> Dict[str, Any]:
    messages = [{"role": "user", "content": user_content}]
    try:
        raw = invoke_fn(system, messages, max_tokens=max_tokens)
        result = _parse_stage_json(raw)
        if result:
            return result
        logger.warning("[PIPELINE_%s] Could not parse JSON from response: %s", stage_name.upper(), raw[:200])
    except Exception as exc:
        logger.error("[PIPELINE_%s] Error: %s", stage_name.upper(), exc)
    return {}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_assessment_pipeline(
    profile: PatientProfile,
    conversation: List[ChatMessage],
    intake_summary: Optional[Dict],
    invoke_fn: Callable[[str, List[Dict]], str],
    prescribing_enabled: bool = False,
    progress_callback: Optional[Callable[[str, str], None]] = None,
) -> Tuple[Dict[str, Any], str]:
    """
    Run the 3-stage assessment pipeline and return (assessment_dict, patient_message).

    assessment_dict has the same structure expected by triage.apply_safety_policy.
    patient_message is the writer-generated patient-facing string (may be empty on failure,
    in which case apply_safety_policy will generate a fallback).
    """
    profile_block = _format_profile_block(profile)
    conversation_block = _format_conversation_block(conversation)
    intake_block = (
        json.dumps(intake_summary, indent=2)
        if intake_summary
        else "(not structured — see conversation transcript)"
    )
    prescribing_note = (
        "PRESCRIBING IS AUTHORIZED: include specific medications with dose/route/frequency/duration."
        if prescribing_enabled
        else "PRESCRIBING IS NOT AUTHORIZED: leave prescription_guidance as []."
    )

    # ── Stage 1: Clinical assessment ──────────────────────────────────────
    if progress_callback:
        progress_callback("assessing", "Analyzing your symptoms...")
    assess_input = (
        f"PATIENT PROFILE:\n{profile_block}\n\n"
        f"INTAKE SUMMARY:\n{intake_block}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_block}\n\n"
        f"PRESCRIBING POLICY: {prescribing_note}"
    )
    assessment_raw = _call_stage(invoke_fn, ASSESSMENT_SYSTEM_PROMPT, assess_input, "assess")
    logger.info("[PIPELINE_ASSESS] urgency=%s confidence=%s",
                assessment_raw.get("urgency"), assessment_raw.get("confidence"))

    # ── Stage 2: Safety critic ────────────────────────────────────────────
    if progress_callback:
        progress_callback("critic", "Running safety review...")
    critic_input = (
        f"PATIENT PROFILE:\n{profile_block}\n\n"
        f"CONVERSATION TRANSCRIPT:\n{conversation_block}\n\n"
        f"PROPOSED ASSESSMENT:\n{json.dumps(assessment_raw, indent=2)}"
    )
    critic_result = _call_stage(invoke_fn, CRITIC_SYSTEM_PROMPT, critic_input, "critic")

    issues = critic_result.get("issues_found", [])
    if issues:
        logger.info("[PIPELINE_CRITIC] Issues found: %s", issues)

    # Use critic-corrected assessment if available and valid, else fall back
    final_assessment = critic_result.get("assessment") or assessment_raw

    # ── Stage 3: Patient-facing writer ───────────────────────────────────
    if progress_callback:
        progress_callback("writing", "Drafting your summary...")
    writer_input = (
        f"PATIENT PROFILE (for context):\n{profile_block}\n\n"
        f"APPROVED CLINICAL ASSESSMENT:\n{json.dumps(final_assessment, indent=2)}"
    )
    writer_result = _call_stage(
        invoke_fn, WRITER_SYSTEM_PROMPT, writer_input, "writer", max_tokens=400
    )
    patient_message = str(writer_result.get("patient_message", "")).strip()

    return final_assessment, patient_message
