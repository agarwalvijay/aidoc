from typing import List
from .models import RedFlagHit

# Deterministic safety gate — absolute minimum only.
#
# Philosophy: keep only phrases that are:
#   1. Virtually never said in denial or past-tense context mid-conversation
#   2. Always mean the same emergency regardless of surrounding words
#   3. Require immediate action before any LLM processing
#
# Everything else — chest pain, stroke symptoms, vital sign concerns,
# sepsis, bleeding, psychiatric nuance — is handled by the LLM which
# understands context, negation, and clinical significance far better
# than substring matching ever can.
#
# When in doubt, leave it out. A false positive here sends a patient
# to the ER unnecessarily and destroys trust. The LLM is the safety net.

RED_FLAG_PATTERNS = {
    "cardiac": [
        "heart attack",
    ],
    "suicidality": [
        "want to kill myself",
        "suicidal",
        "thinking about suicide",
    ],
    "respiratory_emergency": [
        "cannot breathe",
        "can't breathe",
        "gasping for air",
        "choking",
        "throat is closing",
        "throat is swelling",
    ],
    "neurological_emergency": [
        "having a seizure",
        "convulsing",
    ],
    "obstetric_emergency": [
        "water broke",
        "think i am in labor",
    ],
}


def detect_red_flags(text: str) -> List[RedFlagHit]:
    lowered = text.lower()
    hits: List[RedFlagHit] = []
    for code, patterns in RED_FLAG_PATTERNS.items():
        for pattern in patterns:
            if pattern in lowered:
                hits.append(RedFlagHit(code=code, evidence=pattern))
                break  # one hit per category is enough
    return hits
