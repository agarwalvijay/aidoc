# AI Doctor — Primary Care Triage Assistant

A safety-first AI triage assistant that identifies common primary care presentations, escalates high-risk situations, and delivers plain-language guidance — reducing load on the medical system.

**This is a technical scaffold — not a medical device. Clinical, legal, and operational validation is required before any patient use.**

---

## Architecture

```
Patient Onboarding (demographics, PMH, vitals, meds, allergies, chief complaint)
    │
    ▼
POST /api/session/start  →  personalized welcome + LLM-generated opening question
    │
    ▼
POST /api/session/{id}/turn  →  Server-Sent Events stream
    │
    ├── Deterministic red-flag gate (red_flags.py)
    │       Bypasses LLM entirely — immediate emergency response
    │
    ├── Deterministic critical-vitals gate (main.py)
    │       BP ≥ 180/120, HR > 150, SpO2 < 92%, Temp ≥ 105°F → emergency_now
    │
    └── LLM intake engine (llm.py)
            │
            ├── action: ask_question  →  condition-specific follow-up (up to 12 turns)
            │
            └── action: assess  →  4-stage clinical pipeline (pipeline.py)
                    │
                    ├── Stage 1: assessment_llm  — differential, urgency, care plan
                    ├── Stage 2: critic_llm      — safety review, red flag check, drug safety
                    ├── Stage 3: writer_llm      — plain-language patient message + reasoning summary
                    └── Safety policy (triage.py) — urgency floors, Rx sanitization
```

**The LLM is the primary clinical engine.** It receives the full patient profile (including annotated vitals) and conversation history, and decides what to ask next or when to assess. The 4-stage pipeline separates clinical reasoning from patient communication, with an independent safety review between them.

---

## Target Conditions

URI, pharyngitis/strep, otitis media, sinusitis, GI illness (gastroenteritis/GERD), UTI/cystitis, skin rashes/dermatitis, acute musculoskeletal back pain, tension/migraine headache, anxiety/depression (PHQ-9 style), HTN follow-up, DM follow-up.

---

## Key Files

| File | Purpose |
|---|---|
| `backend/app/llm.py` | Intake LLM — clinical system prompt, condition-specific branches, vital sign priority pivots, `process_turn()` |
| `backend/app/pipeline.py` | 4-stage assessment pipeline: assess → critic → writer, with progress callbacks for SSE |
| `backend/app/red_flags.py` | Deterministic safety gate — 70+ patterns across 9 categories |
| `backend/app/triage.py` | Safety policy layer — builds emergency assessments, enforces urgency floors, sanitizes Rx output |
| `backend/app/main.py` | FastAPI routes — SSE streaming turn endpoint, TTS endpoint |
| `backend/app/models.py` | `PatientProfile`, `TriageAssessment`, session models |
| `backend/app/config.py` | All configuration via environment variables |
| `frontend/index.html` | Single-file 3-screen app: Welcome → Onboarding → Voice/text consultation |

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `POST` | `/api/session/start` | Start session; returns welcome + first question |
| `POST` | `/api/session/{id}/turn` | Submit turn; returns SSE stream of progress events then result |
| `POST` | `/api/tts` | Neural TTS via OpenAI (returns mp3); 501 if disabled |
| `GET`  | `/api/health` | Health check; shows LLM provider and TTS mode |

### SSE event format (`/api/session/{id}/turn`)
```json
{"type": "progress", "stage": "assessing", "label": "Analyzing your symptoms..."}
{"type": "result",   "data": { ...TurnResponse... }}
```

Progress stages: `thinking` → `assessing` → `critic` → `writing`

---

## Local Development

### Backend
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — set LLM_PROVIDER and the matching API key
uvicorn app.main:app --reload
```

### Frontend
```bash
cd frontend
python3 -m http.server 3000
```
Open `http://localhost:3000`. The frontend auto-detects local vs production and points API calls at `localhost:8000` when running locally.

---

## LLM Provider Configuration

Recommended: **Anthropic Claude** — best clinical reasoning, reliable JSON output, supports assistant prefill for JSON enforcement.

```env
LLM_PROVIDER=anthropic
ANTHROPIC_API_KEY=sk-ant-...
ANTHROPIC_MODEL=claude-sonnet-4-6
```

Other supported providers (set in `.env`):

| Provider | Key | Notes |
|---|---|---|
| `openai` | `OPENAI_API_KEY` | GPT-4o; uses `response_format: json_object` |
| `groq` | `GROQ_API_KEY` | llama-3.3-70b; fast, good for dev/testing |
| `deepseek` | `DEEPSEEK_API_KEY` | OpenAI-compatible |
| `google` | `GOOGLE_API_KEY` | Gemini 1.5 Pro |

---

## Configuration Reference (`backend/.env`)

```env
# Safety
CONSERVATIVE_MODE=true          # low-confidence cases floored at specialist_soon
PRESCRIBING_ENABLED=false       # true = LLM may recommend specific medications (OTC + limited Rx)
MIN_TURNS_BEFORE_ASSESSMENT=3
MAX_TURNS_BEFORE_ASSESSMENT=12  # 12 allows full PHQ-9; simpler cases self-terminate earlier

# TTS
TTS_ENABLED=true                # false = force Web Speech API (browser built-in)
                                # true + OPENAI_API_KEY = OpenAI neural TTS (nova voice)
```

---

## Production Deployment (GCP / nginx + PM2)

### PM2 start script (`backend/start.sh`)
```bash
#!/bin/bash
cd /home/vagarwal/aidoc/backend
set -a; source .env; set +a
exec .venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001 --workers 1
```
> **Important:** Use `--workers 1`. The session store is in-memory; multiple workers will cause 404s when requests hit different workers.

### nginx — critical SSE setting
```nginx
location /api/ {
    proxy_pass         http://127.0.0.1:8001;
    proxy_buffering    off;   # required — SSE progress events must not be buffered
    proxy_cache        off;
    proxy_read_timeout 120s;
}
```

### Deploy updates
```bash
cd /home/vagarwal/aidoc && git pull && pm2 restart aidoctor
```

---

## Frontend Features

- **3-screen flow:** Welcome (cinematic chime + stethoscope splash) → Onboarding form → Consultation
- **Voice-first:** Web Speech API with neural voice preference (Online/Enhanced voices prioritised); falls back to text input
- **OpenAI TTS:** If `TTS_ENABLED=true` and `OPENAI_API_KEY` is set, audio is served from the backend using OpenAI's `nova` voice — significantly better quality than browser TTS
- **Real-time progress panel:** SSE stream shows each pipeline stage as it runs (Reviewing → Analyzing → Safety review → Drafting)
- **Structured assessment panel:** Urgency banner (colour-coded), possible conditions, plain-language reasoning summary ("Why this assessment?"), recommended next step, care instructions, medication guidance

---

## Safety Design Principles

1. **Deterministic gates first** — red flags and critical vitals bypass the LLM entirely; no model can override these
2. **Independent safety critic** — a separate LLM pass reviews every assessment before it reaches the patient
3. **Urgency bias** — when uncertain between two urgency levels, always choose the higher one
4. **Prescribing off by default** — `PRESCRIBING_ENABLED=false`; when enabled, the critic checks allergies, drug interactions, and PMH contraindications
5. **Conservative mode** — low-confidence assessments are floored at `specialist_soon`, never `self_care_monitor`
6. **Plain-language output** — the writer stage is separate from clinical reasoning; no jargon reaches the patient
