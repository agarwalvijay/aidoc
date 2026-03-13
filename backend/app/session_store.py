import time
from uuid import uuid4
from typing import Optional

from .models import PatientProfile, SessionState

# Sessions expire after 4 hours of inactivity — enough for any clinical encounter.
SESSION_TTL_SECONDS = 4 * 60 * 60


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}
        self._created_at: dict[str, float] = {}

    def create_with_profile(self, profile: PatientProfile) -> SessionState:
        session_id = str(uuid4())
        session = SessionState(session_id=session_id, patient_profile=profile)
        self._sessions[session_id] = session
        self._created_at[session_id] = time.monotonic()
        return session

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)

    def cleanup_expired(self) -> int:
        """Remove sessions older than SESSION_TTL_SECONDS. Returns count removed."""
        now = time.monotonic()
        expired = [
            sid for sid, t in self._created_at.items()
            if now - t > SESSION_TTL_SECONDS
        ]
        for sid in expired:
            self._sessions.pop(sid, None)
            self._created_at.pop(sid, None)
        return len(expired)


session_store = SessionStore()
