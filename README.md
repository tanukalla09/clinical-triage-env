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

# 🏥 ClinicalTriage OpenEnv

**An emergency-room triage simulation environment for reinforcement learning agents.**
Built solo by **Tanushree Kalla**, submitted under team **VisionVerse**, for the **Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026**.

> ⚠️ **Synthetic simulation only.** All patient data is procedurally generated. Not for real medical use, diagnosis, or triage assistance.

**Live demo:** https://tanukalla09-clinical-triage-env.hf.space
**Repo:** [`tanukalla09/clinical-triage-env`](https://github.com/tanukalla09/clinical-triage-env)

---

## What This Is

An RL agent plays the role of a triage nurse in an ER. For each patient it sees two things — symptoms and vitals — and must make two decisions: how severe is this patient (`triage_level`), and where should they go (`disposition`). Get it right and you score well; misjudge a critical patient as low-priority and you take a heavy safety penalty, the way a real triage mistake would carry real consequences.

It's built as a fully offline, self-contained **FastAPI** service that speaks the OpenEnv `reset` / `step` / `grade` contract, packaged for Hugging Face Spaces via Docker.

**Why it's a non-trivial RL problem:**
- Classify severity under incomplete information and time pressure
- Allocate genuinely scarce resources — as few as 1 ICU bed for 12 patients in hard mode
- Safety-critical failure modes — missing a cardiac arrest is the worst possible mistake, and the reward function reflects that asymmetry
- Balance 4–12 patients per episode with competing priorities
- Every decision produces a shaped, per-step reward — not one sparse signal at the end

---

## Architecture

```
clinical-triage-env/
├── environment.py      ← Core simulation: patient generator, oracle classifier, reward logic
├── app.py               ← FastAPI app — wraps environment.py in the OpenEnv HTTP contract
├── server/app.py         ← Thin alternate entrypoint (imports app.py, runs uvicorn directly)
├── inference.py          ← Optional: runs a remote LLM against the live environment
├── static/index.html     ← Landing page served at "/"
├── openenv.yaml           ← Machine-readable spec: tasks, action/observation space, reward structure
├── Dockerfile             ← Container build for HF Spaces (python:3.11-slim, port 7860)
├── requirements.txt       ← Runtime dependencies
└── pyproject.toml         ← Package metadata
```

The environment logic (`environment.py`) has zero external dependencies — no network calls, no database — so it runs identically locally, in Docker, and on HF Spaces. `inference.py` is the only piece that talks to an external API, and only if you choose to run it.

---

## Environment Design

### Two decisions per patient

**1. Triage level** — how severe is this patient?

| Level | Meaning | Example signal |
|---|---|---|
| `IMMEDIATE` | Life-threatening — act now | SpO2 < 88%, HR > 150, chest pain + SpO2 < 92% |
| `URGENT` | Serious — within 30 min | SpO2 < 95% + shortness of breath, cardiac history |
| `STANDARD` | Stable — within 2 hrs | Temp > 38.5°C, moderate pain |
| `LOW` | Minor — can wait | Normal vitals, mild symptoms |

**2. Disposition** — where does this patient go?

| Disposition | Paired with |
|---|---|
| `ICU` | IMMEDIATE |
| `GENERAL` | URGENT |
| `OBSERVATION` | STANDARD |
| `DISCHARGE` | LOW |

### Ground truth: `classify_severity()`

A deterministic, rule-based function in `environment.py` maps a patient's vitals/symptoms to the *correct* `(triage_level, disposition)` pair — e.g. `chest_pain + SpO2 < 92` → `IMMEDIATE/ICU`, `temp > 38.5` → `STANDARD/OBSERVATION`, and so on down a fixed priority list ending in `LOW/DISCHARGE`. This same function is used both to grade an agent's decisions **and** as the built-in "oracle" policy (see Baseline Scores below).

### Tasks

| Task | Difficulty | Patients | ICU Beds | Doctors | Key challenge |
|---|---|---|---|---|---|
| Single Patient Triage | Easy | 4 | 5 | 4 | Clear signals, ample resources |
| Mixed Severity ER | Medium | 8 | 3 | 2 | Balance accuracy with moderate scarcity |
| Mass Casualty Event | Hard | 12 | 1 | 1 | Subtle signals + extreme scarcity |

---

## Reward Function

Computed **per patient, per step** (shaped, not sparse), then clipped to `[0.0, 1.0]`:

| Component | Score | Condition |
|---|---|---|
| Triage level — exact match | +0.50 | |
| Triage level — off by one | +0.25 | e.g. `URGENT` when correct is `IMMEDIATE` |
| Triage level — off by 2+ | 0.00 | |
| Disposition — exact match | +0.30 | |
| Disposition — ICU/GENERAL mix-up | +0.10 | |
| Disposition — other partial match | +0.15 | |
| Disposition — DISCHARGE when care was needed | 0.00 | |
| Safety penalty — IMMEDIATE → STANDARD/LOW | **−0.50** | Life-threatening miss |
| Safety penalty — IMMEDIATE → URGENT | −0.10 | Close but risky |
| Safety penalty — URGENT → LOW | −0.25 | Dangerous delay |
| Resource penalty — ICU requested, 0 beds left | −0.15 | |
| Resource penalty — GENERAL/OBSERVATION requested, 0 beds left | −0.10 | |
| Time bonus — IMMEDIATE, seen in < 5 min, correct | +0.10 | |

Level ordering for the "off-by-N" calculation: `IMMEDIATE(4) > URGENT(3) > STANDARD(2) > LOW(1)`, `diff = abs(agent_rank − true_rank)`.

---

## Observation & Action Space

**Observation** (what the agent sees each step):
```json
{
  "patient": {
    "patient_id": "P003", "age": 72,
    "symptoms": ["chest_pain", "shortness_of_breath"],
    "oxygen_saturation": 88, "heart_rate": 142,
    "blood_pressure": "low", "temperature": 37.2,
    "pain_level": 8, "risk_factors": ["cardiac_history", "diabetes"],
    "waiting_time_min": 3
  },
  "hospital": { "icu_beds": 2, "general_beds": 8, "doctors": 2, "nurses": 5 },
  "queue_length": 5, "step_num": 3, "total_patients": 8,
  "patients_handled": 3, "episode_done": false
}
```

**Action** (what the agent submits):
```json
{ "triage_level": "IMMEDIATE", "disposition": "ICU" }
```
Invalid enum values are rejected automatically with a `422` (Pydantic validation).

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Health check |
| POST | `/reset` | Start a new episode — body: `{"difficulty": "easy\|medium\|hard"}` |
| POST | `/step` | Submit a decision, get the reward + next observation |
| GET | `/state` | Full current environment state |
| GET | `/tasks` | All 3 tasks with full descriptions |
| POST | `/grade/{task_id}` | Runs 5 episodes with the built-in oracle policy, returns scores |

### Verified edge-case behavior

Tested directly against a running instance — all behave exactly as intended:

| Situation | Behavior |
|---|---|
| `/step` called before `/reset` | `400` — "Not initialised. Call reset() first." |
| Invalid `triage_level`/`disposition` | `422` — Pydantic schema error, names the allowed values |
| `/step` after episode is done | `400` — "Episode finished. Call reset() to start a new one." |
| ICU beds exhausted | Resource penalty applied, no crash |

---

## Baseline (Oracle) Scores — Measured

`/grade/{task_id}` runs an oracle policy that calls the same `classify_severity()` function used for grading, so it always classifies patients correctly by construction. Because the base score for a correct classification (`0.50 + 0.30 = 0.80`) is fixed, the oracle's score moves only with **resource penalties** and the **time bonus** — not with how "hard" the task nominally is. Running the grader repeatedly gives consistent results:

| Task | Measured average score | Notes |
|---|---|---|
| task_easy | ~0.81–0.82 | Plenty of beds → frequent time bonus, rare penalties |
| task_medium | ~0.78–0.80 | Some resource contention |
| task_hard | ~0.74–0.75 | Extreme scarcity causes more resource penalties, pulling the score down — but only modestly |

**Known limitation:** because the oracle can't be "wrong" about classification, the difficulty gap between tasks is much narrower than it would be for a learning agent (which *can* misclassify). A real LLM or RL agent evaluated on these tasks would be expected to show a much larger easy→hard gap than the oracle does, since only the oracle enjoys a guaranteed-correct classification floor.

---

## How Hackathon Evaluation Works

| Criterion | Weight | What's checked |
|---|---|---|
| Real-world utility | 30% | Is this a genuine, useful RL task? |
| Task & grader quality | 25% | 3+ tasks, scores vary meaningfully, hard task is actually hard |
| Environment design | 20% | Clean state management, good reward shaping |
| Code quality & spec compliance | 15% | OpenEnv spec, Docker, HF Space, baseline script |
| Creativity & novelty | 10% | Original domain, interesting mechanics |

**Phase 1** (automated) — HF Space deploys, endpoints respond, Dockerfile builds, graders run
**Phase 2** (agentic) — an LLM agent plays all 3 tasks, scores are evaluated
**Phase 3** (human) — Meta/Hugging Face engineers review top submissions

---

## Local Setup

```bash
git clone https://github.com/tanukalla09/clinical-triage-env
cd clinical-triage-env
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 7860
```

- Landing page: `http://localhost:7860`
- Interactive API docs (Swagger): `http://localhost:7860/docs`

## Docker

```bash
docker build -t clinical-triage-env .
docker run -p 7860:7860 clinical-triage-env
```

## Example Requests

```bash
curl https://tanukalla09-clinical-triage-env.hf.space/health

curl -X POST https://tanukalla09-clinical-triage-env.hf.space/reset \
  -H "Content-Type: application/json" -d '{"difficulty": "easy"}'

curl -X POST https://tanukalla09-clinical-triage-env.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"triage_level": "IMMEDIATE", "disposition": "ICU"}'

curl -X POST https://tanukalla09-clinical-triage-env.hf.space/grade/task_easy
```

## Optional: Local LLM Evaluation

`inference.py` runs a remote language model against the live environment — entirely optional, the environment itself needs no external API.

```bash
export API_BASE_URL=https://api-inference.huggingface.co/v1
export MODEL_NAME=mistralai/Mistral-7B-Instruct-v0.3
export HF_TOKEN=your_hf_token_here
python inference.py
```

Emits structured `[START]` / `[STEP]` / `[END]` logs and saves `baseline_scores.json`.

---

## Synthetic Data Notice

All patient data is procedurally generated from randomized templates and rules. No real patient data is used anywhere in this project. This environment is for AI research and RL benchmarking only — not for clinical use, medical decision support, or real triage.

---

## Credits

Built by **[Tanushree Kalla](https://github.com/tanukalla09)**, solo, under team **VisionVerse**, for the **Meta × PyTorch × Hugging Face OpenEnv Hackathon 2026**.
