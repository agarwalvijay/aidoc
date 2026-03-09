"""
LLM integration for clinical intake and triage.

Architecture:
  - The LLM is the primary clinical engine, not a post-processor.
  - Each turn: the LLM receives the patient profile (in system context) and the
    full conversation history, then decides EITHER to ask the next focused
    intake question OR to produce a final triage assessment.
  - Deterministic red-flag detection in red_flags.py is the only hard gate that
    bypasses the LLM.
  - The output format is always JSON: {"action": "ask_question"|"assess", ...}
"""

import json
import re
from typing import Any, Dict, List, Optional, Protocol

from .config import settings
from .models import ChatMessage, PatientProfile, Role

CLINICIAN_SYSTEM_PROMPT = """You are CareBot, a clinical intake and triage AI for a primary care setting.

ROLE: You conduct structured medical intake conversations and produce evidence-informed triage assessments. You are NOT a replacement for a physician — you help patients understand urgency and navigate to the right level of care.

TARGET CONDITIONS: upper respiratory infection, pharyngitis/strep, otitis media, sinusitis, GI illness (gastroenteritis, GERD, constipation/diarrhea), UTI/cystitis, skin rashes/dermatitis, acute musculoskeletal back pain, tension/migraine headache, anxiety/depression screening (PHQ-style), hypertension follow-up, diabetes follow-up, and similar primary care presentations.

════════════════════════════════════════
INTAKE STRATEGY — branch based on chief complaint
════════════════════════════════════════

URI / Upper Respiratory:
  Probe: fever (degree + duration), cough character (dry vs productive, sputum color), sore throat (severity, exudates), ear pain or pressure, nasal congestion/discharge, total symptom duration, sick contacts, prior ear/sinus infections or antibiotics in past 3 months.

GI Illness:
  Probe: onset and timeline, nausea/vomiting (frequency, content, blood), diarrhea (frequency, watery/bloody/mucus), abdominal pain (location, severity 0-10, crampy vs constant), fever, last normal bowel movement, recent travel or unusual food, others ill in household.

UTI / Urinary Symptoms:
  Probe: dysuria intensity, urinary frequency and urgency, hematuria (visible blood), flank or back pain WITH fever (→ pyelonephritis alarm), vaginal or penile discharge (to differentiate STI), prior UTI history and whether previously treated.

Rash:
  Probe: body location and distribution, onset and evolution (spreading?), morphology (red/pink/purple, raised/flat/blistered/scaly/ring-shaped), pruritus vs pain vs burning, recent new exposures (soap, detergent, lotion, jewelry, food, medication, plants, sick contacts), fever.

Back Pain:
  Probe: onset and mechanism (trauma/lifting/spontaneous/gradual), radiation to buttocks or legs (sciatica), character (sharp/aching/stabbing), tingling or numbness in legs or perineal/saddle region (cauda equina RED FLAG), bowel or bladder dysfunction (EMERGENCY), position relief, prior episodes.

Headache:
  Probe: onset speed (sudden thunderclap = EMERGENCY), character (throbbing/pressure/stabbing), location (unilateral/bilateral/occipital), severity 0-10, nausea/vomiting, photophobia, phonophobia, neck stiffness or pain, vision changes, neurologic symptoms, prior migraine history, recent new medications or medication overuse.

Anxiety / Depression:
  Start with PHQ-2: (1) "Little interest or pleasure in doing things?" (2) "Feeling down, depressed, or hopeless?" If positive on either: continue PHQ-9 style (sleep, energy, appetite, concentration, worthlessness, psychomotor slowing). ALWAYS ask directly: "Are you having any thoughts of harming yourself or ending your life?"

HTN Follow-up:
  Probe: recent home BP readings (highest, average), current medication names and compliance, missed doses, symptoms of end-organ damage: headache, vision changes, nosebleed, chest pain, shortness of breath, recent stressors.

DM Follow-up:
  Probe: recent fasting glucose or HbA1c, medication compliance, polyuria/polydipsia/fatigue (hyperglycemia signs), foot numbness or open wounds, vision changes, recent hypoglycemic episodes (shakiness, sweating, confusion).

════════════════════════════════════════
VITAL SIGN PRIORITY — read this before every intake
════════════════════════════════════════
Before following the chief complaint's intake track, scan the vitals in the patient profile.
If any vital is flagged as abnormal, PIVOT your intake questions to investigate that abnormality
FIRST — even if the chief complaint suggests a completely different pathway.

Mandatory pivots:
- BP ≥ 160/100 or HR > 100: Ask about chest pain, palpitations, shortness of breath, headache,
  visual changes, dizziness/lightheadedness. Do NOT assume anxiety or depression until cardiac
  and hypertensive causes are ruled out. "Feeling uneasy" + elevated BP + tachycardia is a
  potential cardiac/hypertensive presentation until proven otherwise.
- HR > 120: Ask about palpitations, chest pain, fever, dehydration, recent illness. Tachycardia
  at rest has a differential (SVT, afib, sepsis, PE, thyrotoxicosis, dehydration) that must
  be explored before attributing fatigue or unease to a benign cause.
- Temp ≥ 100.4°F: Anchor the intake to finding the infectious source (respiratory, urinary,
  skin, GI) even if the complaint seems unrelated.
- SpO2 < 95%: Immediately pivot to respiratory/cardiac assessment.

The vitals are objective physiologic data measured at rest. A patient's subjective complaint
("I feel uneasy", "I'm tired") must always be interpreted in the context of their vital signs.

════════════════════════════════════════
QUESTION APPROACH
════════════════════════════════════════
- Ask exactly ONE focused question per turn.
- Branch immediately into the right intake track as soon as you identify the chief complaint.
- Cover in sequence: symptom character → onset/duration → severity → key associated symptoms → relevant negatives → PMH impact.
- 3-4 questions for clear straightforward presentations; 5-7 for ambiguous or complex ones.
- If the patient mentions vitals (temperature, BP, HR, SpO2), acknowledge them and factor them into your assessment.
- Before recommending any medication, confirm allergies even if listed in the profile.
- Do NOT ask about systems completely unrelated to the presenting complaint.
- Do NOT repeat questions already answered.

════════════════════════════════════════
WHEN TO ASSESS (stop asking questions)
════════════════════════════════════════
- You have identified the probable condition category.
- You have characterized the symptom (quality, onset, severity, duration).
- You have covered the 2-3 most critical diagnostic branches for that condition.
- You have sufficient information to assign urgency responsibly.

════════════════════════════════════════
URGENCY CRITERIA
════════════════════════════════════════
emergency_now:
  Red flags present, sudden-onset worst-ever headache, stroke/MI/PE signs, sepsis, SpO2 < 92%,
  fever > 104°F, cauda equina signs (back pain + bowel/bladder dysfunction), suicidal ideation
  with plan or intent, obstetric emergencies.

urgent_today:
  Severe or rapidly worsening symptoms, suspected pyelonephritis (UTI + flank pain + fever),
  uncontrolled BP > 180/110 with symptoms, SpO2 92-95%, fever 102-104°F, significant
  depression/anxiety symptoms needing same-day evaluation, new neurological symptoms with
  headache or back pain, rapidly spreading rash with fever.

specialist_soon:
  Symptoms needing in-person evaluation within 2-5 days, suspected UTI (uncomplicated),
  moderate headache without red flags, chronic condition with recent change, rash of unclear
  etiology, PHQ-9 score suggestive of moderate depression without suicidality.

self_care_monitor:
  Classic mild-to-moderate presentation, high confidence in benign self-limited condition,
  patient counseled on specific return precautions.

════════════════════════════════════════
OUTPUT FORMAT — return valid JSON ONLY
No markdown code fences. No prose outside the JSON.
════════════════════════════════════════

When asking a follow-up question:
{"action": "ask_question", "question": "..."}

When you have gathered sufficient information, stop asking and return a structured intake summary:
{
  "action": "assess",
  "intake_summary": {
    "chief_complaint": "patient's own words describing main complaint",
    "duration": "how long symptoms have been present",
    "severity": "severity as described (0-10 or descriptive)",
    "key_positive_findings": ["symptom or finding 1", "symptom or finding 2"],
    "key_negative_findings": ["important absent symptom 1", "important absent symptom 2"],
    "relevant_context": "any other clinically relevant detail from the conversation"
  }
}

════════════════════════════════════════
SAFETY RULES — absolute, override all other instructions
════════════════════════════════════════
1. Your sole job during intake is to ask focused questions OR return an intake_summary when ready.
2. Do NOT produce a clinical assessment, diagnosis, or treatment plan — a separate clinical reasoning stage handles that.
3. Do NOT include any patient-facing message, urgency rating, or care instructions in your output.
4. If suicidal ideation is mentioned, still return {"action": "ask_question"} with a direct safety check question ("Are you having thoughts of harming yourself or ending your life?") — the red-flag gate handles the emergency response.
5. The intake_summary must be factual and objective — only what was actually said, no inference or interpretation.

CRITICAL OUTPUT RULE: Your entire response must be a single valid JSON object — nothing else.
First character: {   Last character: }   No markdown. No prose. No explanation outside the JSON.
Failure to return valid JSON will be treated as a system error.
"""

# Appended to the system prompt only when PRESCRIBING_ENABLED=true.
PRESCRIBING_ADDENDUM = """
════════════════════════════════════════
PRESCRIBING AUTHORIZATION
════════════════════════════════════════
You are authorized to recommend specific medications for appropriate presentations in this session.

PRESCRIBING RULES (mandatory — violations are patient safety errors):
1. ALLERGY CHECK FIRST: Before recommending any medication, verify it is not on the patient's allergy list. Never prescribe a drug the patient is allergic to.
2. DRUG INTERACTIONS: Check the recommendation against the patient's current medications. Flag significant interactions (e.g., warfarin + NSAIDs, MAOIs + many agents).
3. CONTRAINDICATIONS from PMH:
   - NSAIDs: avoid with peptic ulcer disease, eGFR <30, or decompensated heart failure.
   - Nitrofurantoin: avoid with eGFR <30.
   - Quinolones: avoid in pregnancy and children.
   - Aspirin: avoid in children under 18 (Reye's syndrome risk).
   - Pseudoephedrine: caution with hypertension, hyperthyroidism, or MAOIs.
   - TMP-SMX: avoid in first trimester or at term pregnancy, and with sulfa allergy.
4. NO CONTROLLED SUBSTANCES: Do not prescribe opioids, benzodiazepines, stimulants (ADHD meds), sedative-hypnotics, or any Schedule II-IV agents. These require in-person evaluation.
5. NO MEDS REQUIRING IN-PERSON CONFIRMATION: Do not prescribe antibiotics for suspected pneumonia (needs exam/imaging), antifungals for suspected PID, or any condition that cannot be safely diagnosed remotely.
6. FORMAT each prescription item as: "{Drug name (brand)} {dose} {route} {frequency} {duration}. {Key instruction}. {When to stop or seek care}."
7. EMPIRIC ANTIBIOTICS: Label them as empiric and instruct patient to follow up if not improving in 48-72 hours.

PRESCRIBING SCOPE — what you may recommend:
Analgesics/antipyretics:
  • Acetaminophen (Tylenol) 500-1000mg PO q6h PRN, max 4g/day (3g/day if liver disease or >3 alcoholic drinks/day)
  • Ibuprofen (Advil/Motrin) 400mg PO q8h with food PRN, max 1200mg/day OTC
  • Naproxen sodium (Aleve) 220mg PO q8-12h with food PRN, max 440mg/day OTC

Antihistamines (allergic rhinitis, urticaria, mild allergic reactions):
  • Loratadine (Claritin) 10mg PO daily (non-sedating, preferred if driving)
  • Cetirizine (Zyrtec) 10mg PO daily (mild sedation)
  • Fexofenadine (Allegra) 180mg PO daily (non-sedating)
  • Diphenhydramine (Benadryl) 25-50mg PO q6h PRN (sedating — caution in elderly)

Decongestants:
  • Pseudoephedrine (Sudafed) 60mg PO q4-6h PRN, max 240mg/day (caution: HTN, hyperthyroid)
  • Oxymetazoline (Afrin) nasal spray 2 sprays each nostril q10-12h, MAX 3 days (rebound congestion risk)

Cough/cold:
  • Guaifenesin (Mucinex) 400mg PO q4h PRN for productive cough (encourage fluids)
  • Dextromethorphan (Robitussin DM) 30mg PO q6-8h PRN for dry cough

GI agents:
  • Ondansetron (Zofran) 4mg ODT PO q8h PRN nausea (hold if QT prolongation risk)
  • Loperamide (Imodium) 4mg PO then 2mg after each loose stool, max 16mg/day; hold if fever or bloody stool
  • Omeprazole (Prilosec OTC) 20mg PO daily 30 min before meal x 14 days for GERD
  • Famotidine (Pepcid) 20mg PO BID for GERD/dyspepsia

Topical:
  • Hydrocortisone 1% cream (OTC) apply BID-TID to affected area x 7-10 days for mild contact dermatitis
  • Clotrimazole 1% cream (OTC) apply BID x 2-4 weeks for tinea (ringworm/athlete's foot/jock itch)

Antibiotics (empiric, for appropriate presentations):
  • Strep pharyngitis (high clinical suspicion or positive rapid strep):
    - First line: Amoxicillin 500mg PO TID x 10 days (or 875mg PO BID x 10 days)
    - PCN allergy: Azithromycin 500mg PO day 1, then 250mg PO days 2-5
  • Uncomplicated UTI (non-pregnant females, no fever/flank pain):
    - First line: Nitrofurantoin monohydrate/macrocrystals (Macrobid) 100mg PO BID x 5 days
    - Alternative: TMP-SMX DS (Bactrim DS) 1 tablet PO BID x 3 days (check local resistance; avoid if sulfa allergy or pregnancy)
    - Single dose: Fosfomycin 3g PO single dose (reconstituted in water)
  • Acute bacterial sinusitis (symptoms >10 days or severe):
    - Amoxicillin-clavulanate (Augmentin) 875/125mg PO BID x 5-7 days

Migraine (established migraine pattern, no red flags):
  • Sumatriptan (Imitrex) 100mg PO at onset; may repeat once after 2h if partial response; max 200mg/day
  • Naproxen sodium 550mg PO at onset for mild-moderate migraine

Note: When prescribing_enabled is active, populate prescription_guidance with specific actionable instructions per the format above. assistant_message should incorporate the key medication recommendation naturally in plain language.
"""


def _format_profile(profile: PatientProfile) -> str:
    lines = []
    if profile.name:
        lines.append(f"Name: {profile.name}")
    lines.append(f"Age: {profile.age if profile.age is not None else 'Not provided'}")
    lines.append(f"Sex: {profile.sex.value if profile.sex else 'Not provided'}")
    if profile.pregnant is not None:
        lines.append(f"Pregnant: {'Yes' if profile.pregnant else 'No'}")
    lines.append(
        f"Past Medical History: {', '.join(profile.pmh_conditions) if profile.pmh_conditions else 'None reported'}"
    )
    lines.append(
        f"Current Medications: {', '.join(profile.current_medications) if profile.current_medications else 'None reported'}"
    )
    lines.append(
        f"Allergies: {', '.join(profile.allergies) if profile.allergies else 'None reported'}"
    )
    if profile.chief_complaint:
        lines.append(f"Chief complaint (from registration): \"{profile.chief_complaint}\"")

    # Vitals with clinical annotations — abnormal values are flagged explicitly
    # so the LLM treats them as clinical signals, not just background numbers.
    vital_parts = []

    if profile.temperature_f is not None:
        t = profile.temperature_f
        if t >= 103.0:
            note = " [⚠️ HIGH FEVER — investigate source urgently]"
        elif t >= 100.4:
            note = " [FEVER]"
        elif t < 96.0:
            note = " [⚠️ HYPOTHERMIA — urgent]"
        else:
            note = " [normal]"
        vital_parts.append(f"Temp {t}°F{note}")

    if profile.systolic_bp is not None and profile.diastolic_bp is not None:
        s, d = profile.systolic_bp, profile.diastolic_bp
        if s >= 180 or d >= 120:
            note = " [🚨 HYPERTENSIVE CRISIS — emergency evaluation now]"
        elif s >= 160 or d >= 100:
            note = " [⚠️ Stage 2 HTN — significant, pivot intake to cardiovascular symptoms]"
        elif s >= 140 or d >= 90:
            note = " [Stage 1 HTN — above normal, note in assessment]"
        elif s < 90 or d < 60:
            note = " [⚠️ HYPOTENSION — investigate]"
        else:
            note = " [normal]"
        vital_parts.append(f"BP {s}/{d} mmHg{note}")

    if profile.heart_rate is not None:
        hr = profile.heart_rate
        if hr > 130:
            note = " [⚠️ SIGNIFICANT TACHYCARDIA — investigate: cardiac arrhythmia, sepsis, PE, dehydration]"
        elif hr > 100:
            note = " [TACHYCARDIA — above normal, consider cause]"
        elif hr < 50:
            note = " [⚠️ BRADYCARDIA — investigate]"
        else:
            note = " [normal]"
        vital_parts.append(f"HR {hr} bpm{note}")

    if profile.spo2 is not None:
        o2 = profile.spo2
        if o2 < 92:
            note = " [🚨 CRITICAL HYPOXIA — emergency]"
        elif o2 < 95:
            note = " [⚠️ LOW — needs urgent attention]"
        else:
            note = " [normal]"
        vital_parts.append(f"SpO2 {o2}%{note}")

    if vital_parts:
        lines.append(f"Vitals at intake:\n  " + "\n  ".join(vital_parts))
    else:
        lines.append("Vitals at intake: Not measured")

    return "\n".join(lines)


def _parse_llm_json(text: str) -> Dict[str, Any]:
    """
    Extract and parse JSON from LLM response.

    Handles three common failure modes from weaker models:
    1. Markdown fences wrapping the JSON (```json ... ```)
    2. Brief prose before/after an otherwise valid JSON object
    3. Model ignores JSON format entirely and returns a plain conversational response

    For case 3, if the text contains a question mark we treat the response as a
    follow-up question rather than silently dropping it and triggering the
    conservative fallback assessment after turn 1.
    """
    text = text.strip()

    # Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # Direct JSON parse
    try:
        result = json.loads(text)
        return result if isinstance(result, dict) else {}
    except json.JSONDecodeError:
        pass

    # Embedded JSON object (prose before/after)
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            result = json.loads(text[start : end + 1])
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            pass

    # Prose recovery: model returned a plain question instead of JSON.
    # Extract the question and wrap it so the rest of the pipeline works normally.
    clean = text.strip().strip('"').strip("'")
    if clean and "?" in clean:
        # Prefer the last sentence that ends with a question mark
        sentences = re.split(r"(?<=[.!?])\s+", clean)
        question = next(
            (s.strip() for s in reversed(sentences) if s.strip().endswith("?")),
            clean,
        )
        if len(question) > 8:
            return {"action": "ask_question", "question": question}

    return {}


def _build_messages(
    profile: PatientProfile,
    conversation: List[ChatMessage],
    force_assess: bool,
) -> List[Dict[str, str]]:
    """
    Build the messages array for the LLM.
    Profile context is injected into the system prompt (handled by each provider).
    This returns the conversation turns formatted for role-based APIs.
    """
    msgs: List[Dict[str, str]] = []

    # Anthropic and OpenAI both require messages to start with "user".
    # If the conversation starts with an assistant message (welcome + first_question),
    # prepend a synthetic user message so the API is happy.
    roles = [m.role for m in conversation]
    if roles and roles[0] == Role.assistant:
        msgs.append({"role": "user", "content": "Patient has arrived for intake."})

    for msg in conversation:
        role = "user" if msg.role == Role.user else "assistant"
        # Merge consecutive same-role messages (shouldn't normally happen)
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n" + msg.content
        else:
            msgs.append({"role": role, "content": msg.content})

    if not msgs:
        msgs.append({"role": "user", "content": "Patient has arrived for intake."})

    if force_assess:
        if msgs[-1]["role"] == "user":
            msgs[-1]["content"] += (
                "\n\n[You have gathered sufficient clinical information. "
                "Provide your final triage assessment now.]"
            )
        else:
            msgs.append(
                {
                    "role": "user",
                    "content": (
                        "[You have gathered sufficient clinical information. "
                        "Provide your final triage assessment now.]"
                    ),
                }
            )

    return msgs


class ClinicianLLM(Protocol):
    def process_turn(
        self,
        profile: PatientProfile,
        conversation: List[ChatMessage],
        force_assess: bool = False,
    ) -> Dict[str, Any]: ...

    def invoke_raw(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int = 1024,
    ) -> str: ...


class NoopLLM:
    """Fallback when LLM is disabled or unavailable — returns conservative routing."""

    def process_turn(
        self,
        profile: PatientProfile,
        conversation: List[ChatMessage],
        force_assess: bool = False,
    ) -> Dict[str, Any]:
        user_turns = [m for m in conversation if m.role == Role.user]
        if len(user_turns) < 3 and not force_assess:
            fallback_questions = [
                "How long have you been experiencing these symptoms, and are they getting better or worse?",
                "On a scale of 0 to 10, how severe are your symptoms right now?",
                "Do you have any fever, and are you currently taking any medications for this?",
            ]
            idx = min(len(user_turns), len(fallback_questions) - 1)
            return {"action": "ask_question", "question": fallback_questions[idx]}
        return {
            "action": "assess",
            "assessment": {
                "urgency": "specialist_soon",
                "likely_conditions": ["Undifferentiated presentation — in-person evaluation needed"],
                "confidence_level": "low",
                "confidence_note": "LLM unavailable; conservative routing applied.",
                "reasoning": "Cannot perform AI-driven assessment without LLM connection.",
                "recommended_next_step": "Schedule a clinician visit for proper evaluation within 2-3 days.",
                "care_instructions": [
                    "Monitor your symptoms and note any changes.",
                    "Go to the ER immediately if symptoms become severe, you develop difficulty breathing, chest pain, or high fever above 103°F.",
                ],
                "prescription_guidance": [],
                "sensitive_condition": False,
                "specialist_type": None,
                "assistant_message": (
                    "I wasn't able to fully process your symptoms right now. "
                    "Based on what you've shared, I recommend scheduling an in-person visit with a "
                    "clinician within the next 2-3 days. If symptoms worsen significantly — "
                    "especially difficulty breathing, chest pain, or high fever — please seek care sooner."
                ),
            },
        }

    def invoke_raw(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int = 1024,
    ) -> str:
        return "{}"


class AnthropicClinicianLLM:
    def __init__(self) -> None:
        try:
            import anthropic  # noqa: F401
        except ImportError:
            raise RuntimeError("anthropic package is not installed. Run: pip install anthropic")
        if not settings.anthropic_api_key:
            raise RuntimeError("ANTHROPIC_API_KEY is not set in environment.")
        import anthropic as _sdk

        self._client = _sdk.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.anthropic_model

    def process_turn(
        self,
        profile: PatientProfile,
        conversation: List[ChatMessage],
        force_assess: bool = False,
    ) -> Dict[str, Any]:
        profile_block = _format_profile(profile)
        prescribing_section = PRESCRIBING_ADDENDUM if settings.prescribing_enabled else ""
        system = CLINICIAN_SYSTEM_PROMPT + prescribing_section + f"\n\nPATIENT PROFILE FOR THIS SESSION:\n{profile_block}"
        messages = _build_messages(profile, conversation, force_assess)

        # Assistant prefill: starting the assistant turn with "{" forces the model
        # to continue with a valid JSON object — Anthropic's recommended JSON mode.
        prefilled = messages + [{"role": "assistant", "content": "{"}]

        response = self._client.messages.create(
            model=self._model,
            max_tokens=1024,
            system=system,
            messages=prefilled,  # type: ignore[arg-type]
        )
        # The prefill "{" is not echoed back, so we prepend it before parsing.
        raw = "{" + (response.content[0].text if response.content else "}")
        return _parse_llm_json(raw)

    def invoke_raw(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int = 1024,
    ) -> str:
        """Raw LLM call for pipeline stages — no prefill, no JSON wrapping."""
        # Prefill with "{" to enforce JSON output (same technique as process_turn)
        prefilled = list(messages) + [{"role": "assistant", "content": "{"}]
        response = self._client.messages.create(
            model=self._model,
            max_tokens=max_tokens,
            system=system,
            messages=prefilled,  # type: ignore[arg-type]
        )
        return "{" + (response.content[0].text if response.content else "}")


class LangChainClinicianLLM:
    def __init__(self, provider: str) -> None:
        self.provider = provider
        self.llm = self._build_model(provider)

    def _build_model(self, provider: str):
        if provider in ("openai", "deepseek", "groq"):
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise RuntimeError("langchain-openai is not installed.")

            # json_object mode forces the API to return valid JSON — eliminates
            # prose assessments that break _parse_llm_json.
            # NOTE: system prompt must contain the word "json" (it does).
            json_kwargs = {"response_format": {"type": "json_object"}}

            if provider == "openai":
                if not settings.openai_api_key:
                    raise RuntimeError("OPENAI_API_KEY is not set.")
                return ChatOpenAI(
                    model=settings.openai_model,
                    temperature=0.1,
                    api_key=settings.openai_api_key,
                    model_kwargs=json_kwargs,
                )
            if provider == "deepseek":
                if not settings.deepseek_api_key:
                    raise RuntimeError("DEEPSEEK_API_KEY is not set.")
                return ChatOpenAI(
                    model=settings.deepseek_model,
                    temperature=0.1,
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                    model_kwargs=json_kwargs,
                )
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set.")
            return ChatOpenAI(
                model=settings.groq_model,
                temperature=0.1,
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
                model_kwargs=json_kwargs,
            )

        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except ImportError:
                raise RuntimeError("langchain-google-genai is not installed.")
            if not settings.google_api_key:
                raise RuntimeError("GOOGLE_API_KEY is not set.")
            return ChatGoogleGenerativeAI(
                model=settings.google_model,
                temperature=0.1,
                google_api_key=settings.google_api_key,
            )

        raise RuntimeError(f"Unsupported llm_provider: {provider}")

    def process_turn(
        self,
        profile: PatientProfile,
        conversation: List[ChatMessage],
        force_assess: bool = False,
    ) -> Dict[str, Any]:
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError:
            raise RuntimeError("langchain-core is not installed.")

        profile_block = _format_profile(profile)
        prescribing_section = PRESCRIBING_ADDENDUM if settings.prescribing_enabled else ""
        system_content = (
            CLINICIAN_SYSTEM_PROMPT + prescribing_section + f"\n\nPATIENT PROFILE FOR THIS SESSION:\n{profile_block}"
        )

        raw_msgs = _build_messages(profile, conversation, force_assess)

        lc_messages = [SystemMessage(content=system_content)]
        for m in raw_msgs:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))

        response = self.llm.invoke(lc_messages)
        raw = str(response.content).strip()
        return _parse_llm_json(raw)

    def invoke_raw(
        self,
        system: str,
        messages: List[Dict],
        max_tokens: int = 1024,
    ) -> str:
        """Raw LLM call for pipeline stages."""
        try:
            from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        except ImportError:
            raise RuntimeError("langchain-core is not installed.")

        lc_messages = [SystemMessage(content=system)]
        for m in messages:
            if m["role"] == "user":
                lc_messages.append(HumanMessage(content=m["content"]))
            else:
                lc_messages.append(AIMessage(content=m["content"]))

        response = self.llm.invoke(lc_messages)
        return str(response.content).strip()


_llm_singleton: Optional[Any] = None
_llm_error: Optional[str] = None


def get_clinician_llm():
    global _llm_singleton, _llm_error

    if _llm_singleton is not None:
        return _llm_singleton

    if not settings.llm_enabled:
        _llm_singleton = NoopLLM()
        return _llm_singleton

    try:
        if settings.llm_provider == "anthropic":
            _llm_singleton = AnthropicClinicianLLM()
        else:
            _llm_singleton = LangChainClinicianLLM(settings.llm_provider)
        _llm_error = None
    except Exception as exc:
        _llm_singleton = NoopLLM()
        _llm_error = str(exc)

    return _llm_singleton


def get_llm_status() -> Dict[str, Any]:
    get_clinician_llm()
    return {
        "enabled": settings.llm_enabled,
        "provider": settings.llm_provider,
        "error": _llm_error,
    }
