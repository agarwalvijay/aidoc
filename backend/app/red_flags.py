from typing import List
from .models import RedFlagHit


# Deterministic high-sensitivity safety checks.
RED_FLAG_PATTERNS = {
    "cardiac_chest_pain": [
        "chest pain",
        "pressure in chest",
        "crushing chest pain",
        "pain to left arm",
    ],
    "stroke_signs": [
        "face drooping",
        "slurred speech",
        "one side weakness",
        "sudden numbness",
        "sudden confusion",
    ],
    "resp_distress": [
        "shortness of breath",
        "cannot breathe",
        "struggling to breathe",
        "blue lips",
    ],
    "severe_bleeding": [
        "heavy bleeding",
        "won't stop bleeding",
        "vomiting blood",
        "blood in stool",
    ],
    "sepsis_signs": [
        "high fever and confusion",
        "fever with low blood pressure",
        "rapid breathing and fever",
    ],
    "cancer_red_flags": [
        "unexplained weight loss",
        "blood in urine",
        "lump getting bigger",
        "persistent night sweats",
        "coughing blood",
    ],
    "suicidality": [
        "want to kill myself",
        "suicidal",
        "end my life",
    ],
}


def detect_red_flags(text: str) -> List[RedFlagHit]:
    lowered = text.lower()
    hits: list[RedFlagHit] = []

    for code, patterns in RED_FLAG_PATTERNS.items():
        for pattern in patterns:
            if pattern in lowered:
                hits.append(RedFlagHit(code=code, evidence=pattern))
                break

    return hits
