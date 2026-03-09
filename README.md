# AI Doctor (Voice-First Triage Scaffold)

Fresh, ground-up starter for a safety-first AI Doctor workflow. This is intentionally **not** TI-project code reuse.

## Product Positioning
- Focus: triage and routing, not definitive diagnosis
- Output: likely conditions, urgency level, and safe next step
- Safety posture: avoid false non-follow-up on high-risk conditions

## Current Architecture
- `backend/app/red_flags.py`
  - Deterministic red-flag detection for emergency/cancer-risk signals
- `backend/app/triage.py`
  - Conservative risk adjudication policy
- `backend/app/intake.py`
  - Follow-up question sequencer
- `backend/app/main.py`
  - FastAPI endpoints for start session + voice turn processing
- `frontend/index.html`
  - Full-screen voice-first shell with chime, TTS, browser STT

## API Endpoints
- `POST /api/session/start`
- `POST /api/session/{session_id}/turn`
- `GET /api/health`

## Local Run

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
# Create env config from template
cp .env.example .env
# Edit .env and set LLM_PROVIDER + matching API key
uvicorn app.main:app --reload
```

### Frontend
Serve `frontend/index.html` with any static server, e.g.:
```bash
cd frontend
python3 -m http.server 3000
```
Then open `http://localhost:3000`.

## Important Safety Note
This scaffold is a technical starting point and not a medical device. It must be validated clinically, legally, and operationally before real patient use.

## LLM Provider Configuration
Set these in `backend/.env`:

- `LLM_ENABLED=true|false`
- `LLM_PROVIDER=openai|deepseek|groq|google`

Provider-specific keys/models:
- OpenAI: `OPENAI_API_KEY`, `OPENAI_MODEL`
- DeepSeek: `DEEPSEEK_API_KEY`, `DEEPSEEK_MODEL`, `DEEPSEEK_BASE_URL`
- Groq: `GROQ_API_KEY`, `GROQ_MODEL`, `GROQ_BASE_URL`
- Google: `GOOGLE_API_KEY`, `GOOGLE_MODEL`

The API health endpoint reports LLM status:
- `GET /api/health`

## Debugging LLM Triage
- Finalized turns now emit `LLM_TRIAGE_DEBUG` logs in the backend console.
- The log includes:
  - full transcript context,
  - parsed LLM enhancement JSON,
  - final assessment payload,
  - final assistant message.
