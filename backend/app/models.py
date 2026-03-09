from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional


class UrgencyLevel(str, Enum):
    emergency_now = "emergency_now"
    urgent_today = "urgent_today"
    specialist_soon = "specialist_soon"
    self_care_monitor = "self_care_monitor"


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class Role(str, Enum):
    user = "user"
    assistant = "assistant"


class ChatMessage(BaseModel):
    role: Role
    content: str


class StartSessionResponse(BaseModel):
    session_id: str
    branding_message: str
    welcome_message: str
    first_question: str
    play_chime: bool = True


class TurnRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="User speech converted to text")


class RedFlagHit(BaseModel):
    code: str
    evidence: str


class TriageAssessment(BaseModel):
    urgency: UrgencyLevel
    confidence_level: ConfidenceLevel = ConfidenceLevel.low
    likely_conditions: List[str]
    confidence_note: str
    recommended_next_step: str
    safety_rationale: str
    care_instructions: List[str] = []
    prescription_guidance: List[str] = []
    specialist_type: Optional[str] = None
    sensitive_condition: bool = False
    red_flag_hits: List[RedFlagHit] = []


class TurnResponse(BaseModel):
    session_id: str
    assistant_message: str
    ask_follow_up: bool
    follow_up_question: Optional[str] = None
    assessment: Optional[TriageAssessment] = None


class SessionState(BaseModel):
    session_id: str
    messages: List[ChatMessage] = []
    facts: dict = {}
    asked_questions: List[str] = []
    turn_count: int = 0
