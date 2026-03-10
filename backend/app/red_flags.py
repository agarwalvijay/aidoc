from typing import List
from .models import RedFlagHit

# Deterministic safety gate — LIFE-THREATENING patterns only.
#
# Philosophy: this gate exists for the narrow set of emergencies where
# immediate action is needed BEFORE any LLM processing (MI, stroke, active
# suicidality, resp arrest, obstetric emergency). It intentionally bypasses
# the LLM entirely.
#
# Everything else — cancer alarm, sepsis, neurological nuance, drug reactions —
# is handled by the LLM intake + independent critic stage, which has the full
# context needed to reason correctly.
#
# False positives here are acceptable. False negatives are not.
RED_FLAG_PATTERNS = {
    "cardiac": [
        "chest pain",
        "chest pressure",
        "chest tightness",
        "chest heaviness",
        "crushing chest",
        "squeezing in my chest",
        "pain radiating to my arm",
        "pain in my left arm",
        "jaw pain",
        "heart attack",
    ],
    "stroke": [
        "face drooping",
        "face is drooping",
        "face numb",
        "sudden slurred speech",
        "slurring my words",
        "arm weakness suddenly",
        "one side weakness",
        "one sided weakness",
        "sudden numbness",
        "sudden confusion",
        "worst headache of my life",
        "thunderclap headache",
        "sudden vision loss",
    ],
    "respiratory_emergency": [
        "cannot breathe",
        "can't breathe",
        "struggling to breathe",
        "gasping for air",
        "choking",
        "throat is closing",
        "throat is swelling",
        "blue lips",
        "blue fingertips",
    ],
    "severe_bleeding": [
        "vomiting blood",
        "throwing up blood",
        "black tarry stool",
        "coughing up blood",
        "heavy uncontrolled bleeding",
        "won't stop bleeding",
    ],
    "suicidality": [
        "want to kill myself",
        "suicidal",
        "end my life",
        "don't want to live",
        "thinking about suicide",
        "want to die",
        "thoughts of ending my life",
        "harming myself",
        "no reason to live",
    ],
    "obstetric_emergency": [
        "pregnant and bleeding",
        "pregnant and severe abdominal pain",
        "think i am in labor",
        "water broke",
        "baby not moving",
        "pregnant and chest pain",
    ],
    "neurological_emergency": [
        "having a seizure",
        "convulsing",
        "loss of consciousness",
        "unresponsive",
        "saddle numbness",
        "loss of bladder control with back pain",
        "loss of bowel control with back pain",
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
