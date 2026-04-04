---
title: ClinicalTriage OpenEnv
emoji: 🏥
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
license: mit
tags:
  - openenv
  - clinical
  - triage
  - healthcare
---

# ClinicalTriage OpenEnv

**Built by VisionVerse** for the Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026

> ⚠️ Synthetic simulation only. All patient data is procedurally generated. Not for real medical use.

---

## Repository Contents

```
clinical-triage-env/
├── environment.py       ← Core OpenEnv environment (offline, no external APIs)
├── app.py               ← FastAPI server wrapping the environment
├── inference.py         ← Baseline demo script (optional local evaluation)
├── Dockerfile           ← Container definition for HF Spaces
├── requirements.txt     ← Python dependencies
├── openenv.yaml         ← OpenEnv metadata and spec
├── README.md            ← This file
└── static/
    └── index.html       ← Landing page
```

**Live demo:** https://tanukalla09-clinical-triage-env.hf.space

---

## Project Overview

ClinicalTriage OpenEnv is an emergency room triage simulation where an RL agent learns to act as a triage nurse — classifying patient severity, allocating scarce hospital resources, and making safety-critical decisions under pressure.

**Why this is a hard RL problem:**
- Classify severity under time pressure with incomplete information
- Allocate scarce resources — only 1 ICU bed may be available in hard mode
- Avoid safety-critical mistakes — missing a cardiac arrest is catastrophic
- Balance competing priorities across 4–12 patients per episode
- Reward signal varies meaningfully across every decision

---

## Team

**VisionVerse** — OpenEnv Hackathon 2026

### Team Setup Notes
- Each team member registers individually on the platform
- Only the Team Lead creates and submits the team form
- Team confirmation is final — cannot be changed after confirming
- Only the latest submission before the deadline is evaluated

---

## Environment Design

### The agent makes two decisions per patient

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
| Safety penalty — IMMEDIATE→LOW/STANDARD | **−0.50** | Life-threatening miss |
| Safety penalty — IMMEDIATE→URGENT | −0.10 | Close but risky |
| Safety penalty — URGENT→LOW | −0.25 | Dangerous delay |
| Resource penalty — ICU when 0 beds | −0.15 | Overuse of scarce ICU |
| Time bonus — critical seen fast | +0.10 | IMMEDIATE, waiting < 5 min, correct |

**Final score = clip(sum, 0.0, 1.0)**

### Off-by-one level calculation

Levels are ordered: IMMEDIATE(4) > URGENT(3) > STANDARD(2) > LOW(1)

`diff = abs(agent_rank - true_rank)`
- diff == 0 → +0.50
- diff == 1 → +0.25
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

## Action Space

The agent submits two fields per step:

```json
{
  "triage_level": "IMMEDIATE",
  "disposition": "ICU"
}
```

- `triage_level`: one of `IMMEDIATE`, `URGENT`, `STANDARD`, `LOW`
- `disposition`: one of `ICU`, `GENERAL`, `OBSERVATION`, `DISCHARGE`

Invalid values are rejected automatically by Pydantic (422 error).

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

### /grade/{task_id} details

- **No request body required**
- Runs 5 episodes using oracle (classify_severity) policy
- Returns average score + individual episode scores

Example response:
```json
{
  "task_id": "task_easy",
  "episodes_run": 5,
  "average_score": 0.825,
  "scores": [0.8, 0.825, 0.825, 0.85, 0.825],
  "grader": "oracle (classify_severity ground truth)",
  "score_range": "0.0–1.0"
}
```

---

## Edge Case Handling

| Situation | Behaviour |
|---|---|
| Invalid triage_level or disposition | 422 — rejected by Pydantic |
| step() before reset() | 400 — "Not initialised. Call reset() first." |
| step() after done=True | 400 — "Episode finished. Call reset() to start a new one." |
| ICU beds exhausted | Resource penalty applied, no crash |
| Missing fields in request | 422 — FastAPI schema validation |

---

## How Round 1 Evaluation Works

Round 1 evaluates submissions across these dimensions:

| Criterion | Weight | What judges check |
|---|---|---|
| Real-world utility | 30% | Is this a genuine task? Real RL value? |
| Task & grader quality | 25% | 3+ tasks? Scores vary? Hard task is hard? |
| Environment design | 20% | Clean state? Good reward shaping? |
| Code quality & spec compliance | 15% | OpenEnv spec, Docker, HF Space, baseline |
| Creativity & novelty | 10% | Original domain? Interesting mechanics? |

**Phase 1** (automated): HF Space deploys, endpoints respond, Dockerfile builds, graders work
**Phase 2** (agentic): LLM agent runs against all 3 tasks, scores are evaluated
**Phase 3** (human): Meta and HF engineers review top submissions

---

## Local Setup

```bash
git clone https://github.com/tanukalla09/clinical-triage-env
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

## Example API Usage

```bash
# Health check
curl https://tanukalla09-clinical-triage-env.hf.space/health

# Start episode
curl -X POST https://tanukalla09-clinical-triage-env.hf.space/reset \
  -H "Content-Type: application/json" \
  -d '{"difficulty": "easy"}'

# Submit decision
curl -X POST https://tanukalla09-clinical-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"triage_level": "IMMEDIATE", "disposition": "ICU"}'

# Run grader
curl -X POST https://tanukalla09-clinical-triage-env.hf.space/grade/task_easy
```

---

## Optional Local Evaluation Script

`inference.py` is an optional demo script that runs a language model against the environment locally. It is **not required** for the environment to function — the core environment runs fully offline with no external APIs.

To run it locally with a remote model:

```bash
# Windows
set API_BASE_URL=https://api-inference.huggingface.co/v1
set MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
set HF_TOKEN=your_hf_token_here
python inference.py

# Mac/Linux
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export HF_TOKEN=your_hf_token_here
python inference.py
```

The script emits structured JSON logs in `[START]`, `[STEP]`, `[END]` format and saves `baseline_scores.json`.

**Note:** `HF_TOKEN` is only needed if calling a remote LLM endpoint. The environment itself runs fully offline.

---

## Baseline Scores

Produced by a deterministic heuristic policy (`classify_severity` oracle) that maps patient vitals directly to the correct triage level and disposition using clinical rules:

| Task | Approximate Score | Policy |
|---|---|---|
| task_easy | ~0.81–0.83 | Oracle (classify_severity) |
| task_medium | ~0.63–0.67 | Oracle (classify_severity) |
| task_hard | ~0.53–0.57 | Oracle (classify_severity) |

> Scores may vary slightly between runs due to random patient generation. For exact reproducibility, set a fixed random seed before running.

---

## Synthetic Data Notice

All patient data in this environment is procedurally generated using randomised templates and rules. No real patient data is used. This environment is not intended for clinical use, medical decision support, or real triage assistance. It is a synthetic simulation for AI research and RL benchmarking purposes only.

---

*ClinicalTriage OpenEnv — VisionVerse — Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026*
