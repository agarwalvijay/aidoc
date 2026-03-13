from enum import Enum
from typing import List, Optional
from pydantic import BaseModel, Field, field_validator


class Sex(str, Enum):
    male = "male"
    female = "female"
    other = "other"
    prefer_not_to_say = "prefer_not_to_say"


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


class PatientProfile(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = Field(None, ge=0, le=120)
    sex: Optional[Sex] = None
    pregnant: Optional[bool] = None
    specialty: str = "primary_care"
    chief_complaint: Optional[str] = None
    pmh_conditions: List[str] = Field(default_factory=list)
    current_medications: List[str] = Field(default_factory=list)
    allergies: List[str] = Field(default_factory=list)
    # Vitals recorded at intake (all optional — patient may not have measured)
    temperature_f: Optional[float] = Field(None, ge=88.0, le=110.0)
    systolic_bp: Optional[int] = Field(None, ge=60, le=280)
    diastolic_bp: Optional[int] = Field(None, ge=30, le=160)
    heart_rate: Optional[int] = Field(None, ge=20, le=300)
    spo2: Optional[int] = Field(None, ge=50, le=100)


class StartSessionRequest(BaseModel):
    patient_profile: PatientProfile


class StartSessionResponse(BaseModel):
    session_id: str
    welcome_message: str
    first_question: str


class TurnRequest(BaseModel):
    transcript: str = Field(..., min_length=1, description="User speech or typed text")

    @field_validator("transcript")
    @classmethod
    def transcript_not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("transcript must not be blank or whitespace only")
        return v


class RedFlagHit(BaseModel):
    code: str
    evidence: str


class ClinicalVitals(BaseModel):
    temperature_f: Optional[float] = None
    heart_rate: Optional[int] = None
    systolic_bp: Optional[int] = None
    diastolic_bp: Optional[int] = None
    spo2: Optional[int] = None


class TriageAssessment(BaseModel):
    urgency: UrgencyLevel
    confidence_level: ConfidenceLevel = ConfidenceLevel.medium
    likely_conditions: List[str] = Field(default_factory=list)
    confidence_note: str = ""
    reasoning: str = ""
    reasoning_summary: str = ""   # plain-language explanation shown to the patient
    recommended_next_step: str = ""
    safety_rationale: str = ""
    care_instructions: List[str] = Field(default_factory=list)
    prescription_guidance: List[str] = Field(default_factory=list)
    specialist_type: Optional[str] = None
    sensitive_condition: bool = False
    red_flag_hits: List[RedFlagHit] = Field(default_factory=list)


class TurnResponse(BaseModel):
    session_id: str
    assistant_message: str
    ask_follow_up: bool
    follow_up_question: Optional[str] = None
    assessment: Optional[TriageAssessment] = None


class SessionState(BaseModel):
    session_id: str
    patient_profile: PatientProfile = Field(default_factory=PatientProfile)
    messages: List[ChatMessage] = Field(default_factory=list)
    turn_count: int = 0
    vitals: ClinicalVitals = Field(default_factory=ClinicalVitals)
