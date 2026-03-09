from uuid import uuid4
from typing import Optional
from .models import SessionState


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, SessionState] = {}

    def create(self) -> SessionState:
        session_id = str(uuid4())
        session = SessionState(session_id=session_id)
        self._sessions[session_id] = session
        return session

    def get(self, session_id: str) -> Optional[SessionState]:
        return self._sessions.get(session_id)


session_store = SessionStore()
