# ClinicalTriage OpenEnv

**Built by VisionVerse** for the Meta × PyTorch × Hugging Face OpenEnv Hackathon

> ⚠️ Synthetic simulation only. All patient data is procedurally generated. Not for real medical use.

---

## Project Overview

ClinicalTriage OpenEnv is an emergency room triage simulation where an RL agent learns to act as a triage nurse — classifying patient severity, allocating scarce hospital resources, and making safety-critical decisions under pressure.

**Why this is a hard RL problem:**
- Classify severity under time pressure with incomplete information
- Allocate scarce resources — only 1 ICU bed may be available
- Avoid safety-critical mistakes — missing a cardiac arrest is catastrophic
- Balance competing priorities across 4–12 patients per episode
- Reward signal varies meaningfully across every decision

---

## Team

**VisionVerse** — OpenEnv Hackathon 2025

---

## Environment Design

### Agent makes two decisions per patient

**Decision 1 — Triage Level** (how severe is this patient?)

| Level | Meaning | Clinical Signal |
|---|---|---|
| `IMMEDIATE` | Life-threatening — act right now | SpO2 < 88%, HR > 150, severe bleeding |
| `URGENT` | Serious — within 30 minutes | SpO2 < 95% + dyspnea, cardiac history |
| `STANDARD` | Stable — within 2 hours | Temp > 38.5, moderate pain |
| `LOW` | Minor — can wait | Normal vitals, minor symptoms |

**Decision 2 — Disposition** (where does this patient go?)

| Disposition | Meaning | Paired with |
|---|---|---|
| `ICU` | Intensive care unit | IMMEDIATE |
| `GENERAL` | General ward admission | URGENT |
| `OBSERVATION` | Monitor and reassess | STANDARD |
| `DISCHARGE` | Safe to send home | LOW |

---

## Tasks

| Task | Difficulty | Patients | ICU Beds | Doctors | Key Challenge |
|---|---|---|---|---|---|
| Single Patient Triage | Easy | 4 | 5 | 4 | Learn basic severity from clear signals |
| Mixed Severity ER | Medium | 8 | 3 | 2 | Balance accuracy with resource constraints |
| Mass Casualty Event | Hard | 12 | 1 | 1 | Subtle signals + extreme scarcity + safety stakes |

---

## Reward Function

Reward is computed **per patient** (per-step, not sparse end-of-episode):

| Component | Score | Condition |
|---|---|---|
| Triage level — exact match | +0.50 | Agent level == correct level |
| Triage level — off by one | +0.25 | e.g. URGENT when correct is IMMEDIATE |
| Triage level — off by 2+ | 0.00 | Badly wrong |
| Disposition — exact match | +0.30 | Agent disposition == correct disposition |
| Disposition — partial | +0.10–0.15 | Related but wrong |
| Disposition — unsafe discharge | 0.00 | DISCHARGE when care needed |
| Safety penalty — IMMEDIATE→LOW/STANDARD | **−0.50** | Life-threatening miss |
| Safety penalty — IMMEDIATE→URGENT | −0.10 | Close but risky |
| Safety penalty — URGENT→LOW | −0.25 | Dangerous delay |
| Resource penalty — ICU when 0 beds | −0.15 | Overuse of scarce ICU |
| Resource penalty — GENERAL when 0 beds | −0.10 | No ward space |
| Time bonus — critical seen fast | +0.10 | IMMEDIATE, waiting < 5 min, correct |

**Final score = clip(sum, 0.0, 1.0)**

### Off-by-one level calculation

Levels are ordered: IMMEDIATE(4) > URGENT(3) > STANDARD(2) > LOW(1)

`diff = abs(agent_rank - true_rank)`
- diff == 0 → 0.50
- diff == 1 → 0.25
- diff >= 2 → 0.00

---

## Observation Space

```json
{
  "patient": {
    "patient_id": "P003",
    "age": 72,
    "symptoms": ["chest_pain", "shortness_of_breath"],
    "oxygen_saturation": 88,
    "heart_rate": 142,
    "blood_pressure": "low",
    "temperature": 37.2,
    "pain_level": 8,
    "risk_factors": ["cardiac_history", "diabetes"],
    "waiting_time_min": 3
  },
  "hospital": {
    "icu_beds": 2,
    "general_beds": 8,
    "doctors": 2,
    "nurses": 5
  },
  "queue_length": 5,
  "step_num": 3,
  "total_patients": 8,
  "patients_handled": 3,
  "episode_done": false
}
```

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check — returns 200 + status ok |
| POST | `/reset` | Start new episode, returns first observation |
| POST | `/step` | Submit triage decision, get reward |
| GET | `/state` | Complete environment state |
| GET | `/tasks` | All 3 tasks with full descriptions |
| POST | `/grade/{task_id}` | Oracle grader — returns score 0.0–1.0 |

### Example usage

```bash
# Health check
curl http://localhost:7860/health

# Start episode (medium difficulty)
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "medium"}'

# Submit decision
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"triage_level": "IMMEDIATE", "disposition": "ICU"}'

# Check state
curl http://localhost:7860/state

# Run grader
curl -X POST http://localhost:7860/grade/task_easy
```

---

## Edge Case Handling

| Situation | Behaviour |
|---|---|
| Invalid triage_level or disposition | 422 — rejected by Pydantic before processing |
| step() before reset() | 400 — "Not initialised. Call reset() first." |
| step() after done=True | 400 — "Episode finished. Call reset() to start a new one." |
| ICU beds exhausted | Resource penalty applied, no crash |
| General beds exhausted | Resource penalty applied, no crash |
| Missing fields in request | 422 — FastAPI schema validation |

---

## Local Setup

```bash
git clone https://github.com/YOUR_USERNAME/clinical-triage-env
cd clinical-triage-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

Open **http://localhost:7860** for the landing page.
Open **http://localhost:7860/docs** for interactive API docs.

---

## Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

---

## Baseline Inference

```bash
# Windows
set API_BASE_URL=https://api.openai.com/v1
set MODEL_NAME=gpt-4o-mini
set HF_TOKEN=your_hf_token_here
python inference.py

# Mac/Linux
export API_BASE_URL=https://api.openai.com/v1
export MODEL_NAME=gpt-4o-mini
export HF_TOKEN=your_hf_token_here
python inference.py
```

The script emits structured JSON logs ([START], [STEP], [END]) to stdout and saves `baseline_scores.json`.

---

## Baseline Scores

Produced by a simple heuristic policy (classify_severity oracle):

| Task | Score | Policy |
|---|---|---|
| task_easy | ~0.81 | Oracle (classify_severity) |
| task_medium | ~0.65 | Oracle (classify_severity) |
| task_hard | ~0.55 | Oracle (classify_severity) |
| Overall | ~0.67 | — |

Baseline policy: uses deterministic clinical rules to map vitals → correct triage level and disposition.

---

## Synthetic Data Notice

All patient data in this environment is procedurally generated using randomised templates and rules. No real patient data is used. This environment is not intended for clinical use, medical decision support, or real triage assistance.