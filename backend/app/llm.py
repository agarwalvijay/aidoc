import json
from typing import Any, Dict, List, Optional, Protocol

from .config import settings

FOLLOW_UP_SYSTEM_PROMPT = """You are a safety-focused clinical intake assistant.
Act like a calm, experienced triage nurse.
Generate exactly one concise follow-up question (max 18 words).
Prioritize triage-critical gaps in this order:
1) danger signs, 2) symptom severity/duration, 3) major risk factors (age/comorbidities/pregnancy), 4) current meds/allergies.
Do not diagnose. Do not ask more than one question."""


ASSESSMENT_SYSTEM_PROMPT = """You are a safety-focused triage assistant.
Act like a careful triage nurse.
You must not provide a definitive diagnosis.
Given the user transcript and deterministic triage output, return JSON only:
{
  "confidence_level": "high|medium|low",
  "likely_conditions": ["...","...","..."],
  "confidence_note": "...",
  "care_instructions": ["..."],
  "prescription_guidance": ["..."],
  "sensitive_condition": true,
  "specialist_type": "...",
  "recommended_next_step": "...",
  "assistant_message": "..."
}
Rules:
- likely_conditions max 3, broad and cautious (e.g., viral illness, inflammatory condition).
- Confidence policy:
  - high: likely condition appears clear from history; include concrete care instructions and optional prescription guidance.
  - medium: provide likely condition + practical home-care instructions + watch-outs.
  - low: explicitly state uncertainty and advise in-person follow-up.
- If sensitive_condition is true (e.g., suspected cancer, severe mental health, reproductive emergencies),
  set specialist_type to a real specialty and recommend prompt specialist follow-up.
- prescription_guidance must be safe and non-final:
  - Prefer OTC classes for self-care.
  - For prescription meds, phrase as "ask a licensed clinician about ...", never direct prescribing.
- assistant_message must be empathetic, clear, and aligned with recommended_next_step and urgency.
- Never contradict deterministic red-flag escalation."""


class ClinicianLLM(Protocol):
    def generate_follow_up(self, transcript: str, asked_questions: List[str]) -> Optional[str]:
        ...

    def generate_assessment_enhancement(self, transcript: str, assessment_payload: Dict[str, Any]) -> Dict[str, Any]:
        ...


class NoopLLM:
    """Fallback implementation when LLM is disabled or unavailable."""

    def generate_follow_up(self, transcript: str, asked_questions: List[str]) -> Optional[str]:
        return None

    def generate_assessment_enhancement(self, transcript: str, assessment_payload: Dict[str, Any]) -> Dict[str, Any]:
        return {}


class LangChainClinicianLLM:
    def __init__(self, provider: str):
        self.provider = provider
        self.llm = self._build_model(provider)

    def _build_model(self, provider: str):
        if provider in ("openai", "deepseek", "groq"):
            try:
                from langchain_openai import ChatOpenAI
            except Exception:
                raise RuntimeError("langchain-openai is not installed.")

            if provider == "openai":
                if not settings.openai_api_key:
                    raise RuntimeError("OPENAI_API_KEY is not set.")
                return ChatOpenAI(
                    model=settings.openai_model,
                    temperature=0.1,
                    api_key=settings.openai_api_key,
                )

            if provider == "deepseek":
                if not settings.deepseek_api_key:
                    raise RuntimeError("DEEPSEEK_API_KEY is not set.")
                return ChatOpenAI(
                    model=settings.deepseek_model,
                    temperature=0.1,
                    api_key=settings.deepseek_api_key,
                    base_url=settings.deepseek_base_url,
                )

            if not settings.groq_api_key:
                raise RuntimeError("GROQ_API_KEY is not set.")
            return ChatOpenAI(
                model=settings.groq_model,
                temperature=0.1,
                api_key=settings.groq_api_key,
                base_url=settings.groq_base_url,
            )

        if provider == "google":
            try:
                from langchain_google_genai import ChatGoogleGenerativeAI
            except Exception:
                raise RuntimeError("langchain-google-genai is not installed.")
            if not settings.google_api_key:
                raise RuntimeError("GOOGLE_API_KEY is not set.")
            return ChatGoogleGenerativeAI(
                model=settings.google_model,
                temperature=0.1,
                google_api_key=settings.google_api_key,
            )

        raise RuntimeError("Unsupported llm_provider: {0}".format(provider))

    def _invoke_text(self, system_prompt: str, user_prompt: str) -> str:
        try:
            from langchain_core.messages import HumanMessage, SystemMessage
        except Exception:
            raise RuntimeError("langchain-core is not installed.")

        response = self.llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        return str(response.content).strip()

    def generate_follow_up(self, transcript: str, asked_questions: List[str]) -> Optional[str]:
        asked_json = json.dumps(asked_questions)
        user_prompt = (
            "Current transcript:\n{0}\n\nAlready asked:\n{1}\n\n"
            "Return only the next single follow-up question."
        ).format(transcript, asked_json)

        text = self._invoke_text(FOLLOW_UP_SYSTEM_PROMPT, user_prompt)
        if not text:
            return None
        cleaned = text.strip().strip('"')
        cleaned = cleaned.replace("\n", " ").strip()
        if len(cleaned) < 8:
            return None
        if not cleaned.endswith("?"):
            cleaned = cleaned + "?"
        return cleaned

    def generate_assessment_enhancement(self, transcript: str, assessment_payload: Dict[str, Any]) -> Dict[str, Any]:
        payload_json = json.dumps(assessment_payload)
        user_prompt = (
            "User transcript:\n{0}\n\nDeterministic triage output:\n{1}\n\n"
            "Return JSON only."
        ).format(transcript, payload_json)

        text = self._invoke_text(ASSESSMENT_SYSTEM_PROMPT, user_prompt)
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else {}
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                try:
                    result = json.loads(text[start:end + 1])
                    return result if isinstance(result, dict) else {}
                except json.JSONDecodeError:
                    return {}
            return {}


_llm_singleton: Optional[ClinicianLLM] = None
_llm_error: Optional[str] = None


def get_clinician_llm() -> ClinicianLLM:
    global _llm_singleton, _llm_error

    if _llm_singleton is not None:
        return _llm_singleton

    if not settings.llm_enabled:
        _llm_singleton = NoopLLM()
        return _llm_singleton

    try:
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
