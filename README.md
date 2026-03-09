# AI Doctor — Primary Care Triage Assistant

A safety-first AI Doctor that identifies common primary care presentations and escalates high-risk situations, reducing load on the medical system.

**This is a technical scaffold — not a medical device. Clinical, legal, and operational validation is required before any patient use.**

## Architecture

```
Patient Onboarding (demographics, PMH, meds, allergies, chief complaint)
    │
    ▼
POST /api/session/start  →  personalized welcome + condition-specific first question
    │
    ▼
POST /api/session/{id}/turn  (repeats until assessment)
    │
    ├── Deterministic red-flag gate (red_flags.py) — bypasses LLM, immediate emergency response
    │
    └── LLM-driven intake + triage (llm.py)
            │
            ├── action: ask_question  →  condition-specific follow-up (3-7 turns)
            │
            └── action: assess  →  safety policy layer (triage.py)  →  TriageAssessment
```

**LLM is the primary clinical engine.** It receives the full patient profile and conversation history and decides what to ask next or when to assess. The deterministic layer only handles red flags and output policy enforcement (no direct prescribing, urgency floors, etc.).

## Target Conditions
URI, pharyngitis/strep, otitis media, sinusitis, GI illness, UTI/cystitis, skin rashes, acute back pain, tension/migraine headache, anxiety/depression (PHQ-style), HTN follow-up, DM follow-up.

## Key Files
- `backend/app/llm.py` — comprehensive clinical system prompt + `process_turn()` interface
- `backend/app/red_flags.py` — deterministic safety gate
- `backend/app/triage.py` — safety policy: builds emergency assessments, sanitizes LLM output
- `backend/app/main.py` — FastAPI routes (simplified — minimal business logic)
- `backend/app/models.py` — `PatientProfile`, `TriageAssessment`, session models
- `frontend/index.html` — 3-screen app: Welcome → Onboarding form → Voice/text consultation

## API Endpoints
- `POST /api/session/start` — body: `{ "patient_profile": { ... } }`
- `POST /api/session/{session_id}/turn` — body: `{ "transcript": "..." }`
- `GET /api/health`

## Local Run

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Set LLM_PROVIDER and the matching API key in .env
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
python3 -m http.server 3000
```
Then open `http://localhost:3000`.

## LLM Provider Configuration (`backend/.env`)

Recommended: **Anthropic Claude** (best clinical reasoning, reliable JSON output)
```
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Other supported providers: `openai`, `deepseek`, `groq`, `google` — see `.env.example` for keys.

## Safety Configuration
- `CONSERVATIVE_MODE=true` — floors uncertain (low-confidence) cases at `specialist_soon` instead of `self_care_monitor`
- `MIN_TURNS_BEFORE_ASSESSMENT=3` — LLM won't be forced to assess before this many turns
- `MAX_TURNS_BEFORE_ASSESSMENT=7` — hard cap; LLM is instructed to assess at this point
