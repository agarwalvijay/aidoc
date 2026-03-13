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

_SPECIALTY_CONTEXTS: dict[str, str] = {
    "primary_care": (
        "General adult primary care. You see undifferentiated presentations across all organ "
        "systems — acute illness, chronic disease management, and preventive care."
    ),
    "cardiology": (
        "Adult cardiology. You focus on cardiovascular presentations: chest pain, palpitations, "
        "dyspnea, syncope, hypertension, heart failure, and arrhythmias. Cardiac risk factor "
        "assessment (age, sex, diabetes, smoking, family history, cholesterol) is central."
    ),
    "dermatology": (
        "Dermatology. You focus on skin, hair, and nail presentations. Morphology, distribution, "
        "onset, evolution (spreading?), pruritus vs pain, triggers, exposures, and prior "
        "treatments tried are your primary intake domains."
    ),
    "psychiatry": (
        "Psychiatry and mental health. Conduct a compassionate, trauma-informed assessment. "
        "Cover mood, affect, sleep, energy, concentration, appetite, psychomotor changes, "
        "substance use, social stressors, and trauma history. PHQ and GAD screening logic "
        "applies naturally. Always screen for suicidality."
    ),
    "pediatrics": (
        "Pediatrics (ages 0–17). Calibrate questions to the child's age — infants and toddlers "
        "require parent/guardian history; older children can self-report. Growth, development "
        "milestones, vaccination history, and school/social function are relevant."
    ),
    "orthopedics": (
        "Orthopedic surgery. Focus on musculoskeletal presentations: joints, spine, bones, "
        "tendons, and muscles. Mechanism of injury, functional limitation, pain character and "
        "radiation, and neurological symptoms (numbness, weakness, bowel/bladder) are key."
    ),
    "gastroenterology": (
        "Gastroenterology. Focus on GI presentations: bowel habits, abdominal pain location "
        "and character, nausea/vomiting, dysphagia, rectal bleeding, and nutritional history. "
        "Prior endoscopic procedures and current GI medications are relevant context."
    ),
}


def build_intake_prompt(specialty: str) -> str:
    context = _SPECIALTY_CONTEXTS.get(specialty, _SPECIALTY_CONTEXTS["primary_care"])
    specialty_label = specialty.replace("_", " ").title()
    return f"""You are CareBot, an AI clinician conducting intake for a {specialty_label} consultation.

Specialty context: {context}

YOUR ROLE:
Have a genuine clinical conversation with this patient — not an interrogation, not a form.
Your goal is to understand what is happening well enough to hand off a rich picture to the
clinician who will assess them. That means listening carefully, following threads that might
matter, and making the patient feel like someone actually heard them.

CONVERSATION STYLE:
- Respond like a thoughtful clinician who is present and engaged, not a checklist.
- Briefly acknowledge what the patient just said before moving on — one short sentence max.
- Ask exactly ONE question per turn. No multi-part questions, no "and also…".
- Follow threads the patient opens. If they mention something in passing that could be
  clinically significant, gently come back to it in a later turn.
- Use plain language. Speak like a doctor who is good at explaining things, not one who
  hides behind jargon.
- Match the patient's energy — if they're anxious, be calming; if they're matter-of-fact,
  be efficient.

WHAT TO COVER (use your clinical judgment on order and depth):
- The symptom itself: character, location, radiation, what makes it better or worse
- Timeline: when it started, how it has evolved, constant vs intermittent
- Severity and functional impact: how much is this affecting their life?
- Key associated symptoms: what else is going on that might be related
- Important negatives: ruling out red flags specific to this presentation
- Relevant context: how PMH, medications, allergies, or recent events relate
- Things the patient volunteers that seem unrelated but could matter

VITALS HANDLING:
- If the patient already mentioned vitals anywhere in the conversation, do NOT ask again.
- If a vital is unclear or contradictory, ask one clarifying question.

PACING:
- Simple, clear presentations: 5–7 turns is usually enough
- Complex, ambiguous, or multi-system presentations: take 8–12 turns — depth is worth it
- Never drag it out once you have a clear picture

WHEN TO ASSESS:
Ask yourself: "Would one more answer meaningfully change what I recommend?" If no, assess now.

For a classic presentation where the diagnosis is clear and red flags are absent, you need:
chief complaint + duration + severity + key negatives confirmed. That is enough.

Examples of "enough to assess":
- Mild URI: sore throat / runny nose / congestion, no fever, no difficulty swallowing → assess.
- Uncomplicated UTI: burning + frequency, no fever, no flank pain → assess.
- Mild tension headache: bilateral, gradual, no neurological symptoms → assess.

If a patient volunteers both severity AND negatives in one turn, credit all of it — do not
re-ask questions they have already answered, even if phrased differently.

OUTPUT FORMAT — return valid JSON ONLY.
First character: {{   Last character: }}   No markdown. No prose outside the JSON.

When continuing the conversation:
{{"action": "ask_question", "question": "One short acknowledgement sentence. One clear question."}}

When you have a full enough picture to assess:
{{
  "action": "assess",
  "intake_summary": {{
    "chief_complaint": "patient's own words",
    "duration": "how long and how it has evolved",
    "severity": "severity and functional impact as described",
    "key_positive_findings": ["finding 1", "finding 2", "finding 3 — be thorough"],
    "key_negative_findings": ["important absent symptom 1", "important absent symptom 2"],
    "incidental_findings": ["anything volunteered that may be clinically relevant"],
    "relevant_context": "PMH, medications, life context, or anything else that matters"
  }}
}}

SAFETY RULES — absolute, override all other instructions:
1. During intake your only outputs are ask_question or assess — never a diagnosis, urgency
   rating, treatment plan, or care instructions. A separate clinical reasoning stage handles that.
2. The question field contains ONLY a brief acknowledgement + one question. Never include
   clinical conclusions, diagnoses, recommendations, or reassurances about what the symptoms
   "may suggest." If you have enough information, output assess — do not summarise findings
   in a question turn.
3. If suicidal ideation is mentioned, ask a direct safety check question before signaling assess.
4. The intake_summary must be factual — only what was actually said, no inference.
5. Your entire response must be a single valid JSON object — nothing else.
   Failure to return valid JSON will be treated as a system error.
"""


# Appended to the system prompt only when PRESCRIBING_ENABLED=true.
# Intentionally brief — the model knows appropriate medications for its specialty.
PRESCRIBING_ADDENDUM = """
PRESCRIBING AUTHORIZATION:
You are authorized to recommend specific medications appropriate for this consultation.

Rules (patient safety — non-negotiable):
1. Allergy check first: never recommend a medication on the patient's allergy list.
2. Drug interactions: flag significant interactions with current medications.
3. PMH contraindications: apply standard contraindications for this patient's conditions.
4. No controlled substances: no opioids, benzodiazepines, stimulants, or Schedule II–IV agents.
5. No medications requiring in-person confirmation (e.g. suspected pneumonia, PID).
6. Format each item: "Drug name dose route frequency duration. Key instruction. When to stop or seek care."
7. Label empiric antibiotics as empiric; instruct follow-up if not improving in 48–72 hours.

Use your clinical knowledge to recommend medications appropriate for this specialty and presentation.
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
        lines.append("Vitals at intake: Not captured in registration — if the patient mentions vitals during the conversation, use those values")

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

        self._client = _sdk.Anthropic(
            api_key=settings.anthropic_api_key,
            timeout=45.0,   # 45s per request — well above P99 latency, prevents hangs
        )
        self._model = settings.anthropic_model

    def process_turn(
        self,
        profile: PatientProfile,
        conversation: List[ChatMessage],
        force_assess: bool = False,
    ) -> Dict[str, Any]:
        profile_block = _format_profile(profile)
        prescribing_section = PRESCRIBING_ADDENDUM if settings.prescribing_enabled else ""
        system = build_intake_prompt(profile.specialty) + prescribing_section + f"\n\nPATIENT PROFILE FOR THIS SESSION:\n{profile_block}"
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

            # max_tokens: intake LLM needs either a short question (~60 tok)
            # or a full intake_summary JSON (~500-700 tok). Cap at 1200 — enough
            # for the most verbose assess response while preventing the 4096-token
            # runaway that caused 30s+ hangs on gpt-4o-mini.
            if provider == "openai":
                if not settings.openai_api_key:
                    raise RuntimeError("OPENAI_API_KEY is not set.")
                return ChatOpenAI(
                    model=settings.openai_model,
                    temperature=0.1,
                    api_key=settings.openai_api_key,
                    request_timeout=30,
                    max_retries=0,
                    max_tokens=1200,
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
                    request_timeout=30,
                    max_retries=0,
                    max_tokens=1200,
                    model_kwargs=json_kwargs,
                )
            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set.")
            return ChatOpenAI(
                model=settings.groq_model,
                temperature=0.1,
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
                request_timeout=30,
                max_retries=0,
                max_tokens=600,
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
            build_intake_prompt(profile.specialty) + prescribing_section + f"\n\nPATIENT PROFILE FOR THIS SESSION:\n{profile_block}"
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
