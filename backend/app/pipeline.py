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

def build_assessment_prompt(specialty: str) -> str:
    specialty_label = specialty.replace("_", " ").title()
    return f"""\
You are a {specialty_label} specialist performing a clinical triage assessment.

Use your medical knowledge and specialty expertise to analyze the patient profile, intake
summary, and conversation transcript. Your sole job is structured clinical reasoning —
do NOT write patient-facing language.

Return valid JSON only:
{{
  "likely_conditions": ["most likely", "second most likely", "third if applicable"],
  "primary_impression": "brief clinical impression with key supporting evidence",
  "urgency": "emergency_now|urgent_today|specialist_soon|self_care_monitor",
  "confidence_level": "high|medium|low",
  "confidence_note": "what supports or limits your confidence",
  "recommended_next_step": "specific, actionable instruction",
  "care_instructions": ["instruction 1", "instruction 2", "at least 2 return precautions"],
  "prescription_guidance": [],
  "sensitive_condition": false,
  "specialist_type": null,
  "reasoning": "your differential reasoning"
}}

URGENCY RULES — read carefully before assigning urgency:

  emergency_now     Reserved for immediately life-threatening conditions where calling 911
                    RIGHT NOW is the correct action. Use ONLY for: active MI/ACS, stroke,
                    PE, anaphylaxis with airway involvement, respiratory arrest, SpO2 < 92%,
                    BP ≥ 180/120 with end-organ symptoms, suicidal ideation WITH a plan,
                    cauda equina syndrome, active obstetric emergency.
                    NEVER use for: UTI, URI, headache, rash, stomach bug, back pain,
                    anxiety, or any condition where "see a doctor soon" is the right advice.

  urgent_today      Not life-threatening, but needs same-day evaluation. Examples:
                    pyelonephritis (UTI + flank pain + fever ≥ 101°F), uncontrolled
                    BP > 160/100 with symptoms, SpO2 92–95%, fever ≥ 103°F, significant
                    psychiatric crisis without active plan, rapidly spreading rash with fever.
                    NOT for: uncomplicated UTI without fever or flank pain.

  specialist_soon   Needs in-person evaluation within 2–5 days — the correct level for
                    most acute primary care presentations. Examples: uncomplicated UTI
                    (dysuria + frequency, no fever, no flank pain, no systemic symptoms),
                    URI/pharyngitis, moderate headache without red flags, skin rash of
                    unclear cause, chronic condition with recent change.

  self_care_monitor High confidence benign self-limited condition with clear return
                    precautions. Examples: mild viral URI, mild tension headache,
                    mild contact dermatitis with known trigger.

BEFORE assigning emergency_now, ask yourself: "Would I call 911 for this patient right
now, or would I tell them to see a doctor?" If the answer is "see a doctor," use
urgent_today or specialist_soon instead.

Assessment principles:
- Base the assessment strictly on gathered information — do not assume.
- Abnormal vitals must be addressed even if the chief complaint seems unrelated.
- When information is incomplete, reflect that as lower confidence.
- Include at least 2 specific return-precaution items in care_instructions.
- If prescribing is authorized, apply standard contraindications and allergy checks.
"""


CRITIC_SYSTEM_PROMPT = """\
You are a senior emergency physician performing a mandatory independent safety review of a
triage assessment before it reaches the patient. Your job is to catch errors that could harm.

SAFETY REVIEW — check every item:

1. MISSED EMERGENCIES — does ANY part of the conversation suggest:
   - Cardiac: chest pain/pressure/tightness, arm/jaw pain, diaphoresis with chest symptoms
   - Stroke: sudden facial drooping, arm weakness, speech difficulty, worst-ever headache
   - Pulmonary embolism: sudden dyspnea + pleuritic chest pain + unilateral leg swelling
   - Sepsis: fever + altered mental status, hypotension, or extreme tachycardia
   - Suicidality: thoughts of self-harm or ending life with plan or intent
   - Cauda equina: back pain + saddle numbness + bowel/bladder dysfunction
   - Obstetric emergency: pregnant + severe abdominal pain or significant bleeding
   - Anaphylaxis: allergic reaction + throat swelling or breathing difficulty
   → Any of the above must be urgency: emergency_now

2. VITAL SIGN ALIGNMENT — abnormal vitals must be reflected in urgency:
   - BP ≥ 180/120: emergency_now
   - BP ≥ 160/100 or HR > 120: at minimum urgent_today; explain in assessment
   - SpO2 < 92%: emergency_now
   - Fever ≥ 103°F: source must be identified; consider urgent_today

3. URGENCY CALIBRATION:
   - UPGRADE urgency only for CURRENT active symptoms indicating an emergency RIGHT NOW.
   - Do NOT upgrade based on theoretical future progression — "UTI could become sepsis",
     "cold could progress to pneumonia", "headache might be a tumor" are NOT grounds for
     upgrading urgency. The patient's CURRENT presentation must have active emergency features.
   - DOWNGRADE emergency_now if the patient's current symptoms do not include: active chest
     pain/pressure, current respiratory distress, current facial drooping or arm weakness,
     SpO2 < 92%, active suicidal ideation with a plan, current hemodynamic instability,
     cauda equina signs, or active obstetric emergency.
   - A UTI, URI, headache, rash, GI illness, anxiety, or chronic condition follow-up is
     NEVER emergency_now based on current presentation alone.
   - Missing or unmeasured vitals lower confidence — they do NOT raise urgency. If vitals
     were not provided, note it as a confidence limitation, not an escalation reason.

4. DRUG SAFETY (if prescription_guidance is non-empty):
   - Allergy check: no medication on the patient's allergy list
   - Drug interactions: flag significant interactions with current medications
   - PMH contraindications: NSAIDs + peptic ulcer/renal failure, nitrofurantoin + CKD,
     quinolones + pregnancy, aspirin + children, TMP-SMX + sulfa allergy/first trimester
   - No controlled substances

5. CONFIDENCE HONESTY — is confidence_level honest given what was gathered?

6. SENSITIVE CONDITIONS — mental health crisis, cancer alarm symptoms, obstetric concerns,
   HIV/STI disclosures → set sensitive_condition: true and specialist_type appropriately.

Return valid JSON only:
{
  "issues_found": ["description of each issue corrected — empty list if none"],
  "assessment": { ...complete corrected assessment using identical field names to input... }
}

If no issues: issues_found: [], assessment identical to input.
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

You must also write a patient_reasoning: 2-3 plain-language sentences that explain
WHY this assessment was reached. It should:
- Name the 2-3 specific findings from the conversation that pointed toward this conclusion
- Briefly explain why the urgency level was chosen (not too high, not too low)
- Mention any key negative findings that influenced the assessment (e.g. "the absence of fever makes a serious infection less likely")
- Never repeat the recommendation — that is already in patient_message
- Never use clinical jargon
- Be specific to THIS patient, not generic ("Your 3-day fever and productive cough..."
  not "Based on your symptoms...")

Return valid JSON only:
{
  "patient_message": "...",
  "patient_reasoning": "..."
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
    specialty: str = "primary_care",
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
    assessment_raw = _call_stage(invoke_fn, build_assessment_prompt(specialty), assess_input, "assess")
    logger.info("[PIPELINE_ASSESS] urgency=%s confidence=%s conditions=%s",
                assessment_raw.get("urgency"), assessment_raw.get("confidence_level"),
                assessment_raw.get("likely_conditions"))

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
    critic_assessment = critic_result.get("assessment", {})
    logger.info("[PIPELINE_CRITIC] urgency=%s confidence=%s issues=%s",
                critic_assessment.get("urgency"), critic_assessment.get("confidence_level"),
                issues)

    # Use critic-corrected assessment if available and valid, else fall back
    final_assessment = critic_result.get("assessment") or assessment_raw

    # Flag if critic upgraded urgency to emergency_now from a lower level.
    # apply_safety_policy uses this to apply extra skepticism to critic-driven escalations.
    if (assessment_raw.get("urgency") != "emergency_now"
            and final_assessment.get("urgency") == "emergency_now"):
        final_assessment["_critic_escalated_to_emergency"] = True
        logger.warning(
            "[PIPELINE_CRITIC] Critic escalated to emergency_now from %s. Issues: %s",
            assessment_raw.get("urgency"), issues,
        )

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
    patient_message  = str(writer_result.get("patient_message",  "")).strip()
    patient_reasoning = str(writer_result.get("patient_reasoning", "")).strip()

    # Attach reasoning to the assessment dict so apply_safety_policy can thread it through
    final_assessment["patient_reasoning"] = patient_reasoning

    return final_assessment, patient_message
