from typing import List
from .models import RedFlagHit

# Deterministic high-sensitivity safety gate.
# Patterns are intentionally broad — false positives are acceptable here;
# false negatives are not.
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
        "palpitations and chest pain",
    ],
    "stroke": [
        "face drooping",
        "face is drooping",
        "face numb",
        "sudden slurred speech",
        "slurring my words",
        "slurring words",
        "arm weakness suddenly",
        "one side weakness",
        "one sided weakness",
        "sudden numbness",
        "sudden confusion",
        "sudden severe headache",
        "worst headache of my life",
        "thunderclap headache",
        "sudden vision loss",
        "vision suddenly went",
    ],
    "respiratory_emergency": [
        "cannot breathe",
        "can't breathe",
        "struggling to breathe",
        "shortness of breath at rest",
        "blue lips",
        "blue fingertips",
        "turning blue",
        "gasping for air",
        "choking",
        "throat is closing",
        "throat is swelling",
        "allergic reaction and breathing",
    ],
    "severe_bleeding": [
        "vomiting blood",
        "throwing up blood",
        "blood in stool",
        "black tarry stool",
        "bright red blood from rectum",
        "coughing up blood",
        "coughing blood",
        "heavy uncontrolled bleeding",
        "won't stop bleeding",
    ],
    "sepsis": [
        "high fever and confusion",
        "fever with confusion",
        "fever and very low blood pressure",
        "rapid breathing and high fever",
        "cold clammy skin and fever",
        "sepsis",
    ],
    "neurological_emergency": [
        "seizure",
        "having a seizure",
        "convulsing",
        "loss of consciousness",
        "passed out and",
        "unresponsive",
        "sudden weakness in both legs",
        "cannot move legs",
        "loss of bladder control with back pain",
        "loss of bowel control with back pain",
        "saddle numbness",
        "numbness in groin with back pain",
    ],
    "cancer_alarm": [
        "unexplained weight loss",
        "unintentional weight loss",
        "blood in urine and back pain",
        "persistent night sweats and weight loss",
        "lump getting bigger",
        "lump that is growing",
        "painless lump",
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
        "hurting myself",
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
