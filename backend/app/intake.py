from typing import Optional


FOLLOW_UP_QUESTIONS = [
    "When did these symptoms start, and are they getting better or worse?",
    "On a scale of 0 to 10, how severe are the symptoms right now?",
    "Do you have fever, weight loss, breathing trouble, or chest pain?",
    "Do you have major medical conditions or take regular medications?",
]


def next_follow_up(asked_questions: list[str]) -> Optional[str]:
    for question in FOLLOW_UP_QUESTIONS:
        if question not in asked_questions:
            return question
    return None
