"""
ClinicalTriage OpenEnv — API Server
Built by VisionVerse for Meta x PyTorch x Hugging Face OpenEnv Hackathon
Synthetic data only. Not for real medical use.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Optional, Literal
import os
from environment import ClinicalTriageEnvironment, Action, classify_severity

app = FastAPI(
    title="ClinicalTriage OpenEnv",
    description="""
## ClinicalTriage OpenEnv — Built by VisionVerse

Emergency room triage simulation for RL agents.

> **Synthetic data only. Not for real medical use.**

---

### How the environment works

The agent acts as a triage nurse. For each synthetic patient it must decide:

**Decision 1 — Triage Level** (how severe?)
| Level | Meaning |
|---|---|
| `IMMEDIATE` | Life-threatening — act right now |
| `URGENT` | Serious — within 30 minutes |
| `STANDARD` | Stable — within 2 hours |
| `LOW` | Minor — can wait |

**Decision 2 — Disposition** (where do they go?)
| Disposition | Meaning |
|---|---|
| `ICU` | Intensive care unit (for IMMEDIATE) |
| `GENERAL` | General ward (for URGENT) |
| `OBSERVATION` | Monitor and reassess (for STANDARD) |
| `DISCHARGE` | Safe to go home (for LOW) |

---

### Episode flow
1. `POST /reset` → get first patient + hospital state
2. `POST /step` → submit triage decision → get reward (0.0–1.0) + next patient
3. Repeat until `done=true`
4. `POST /grade/{task_id}` → standardised score across 5 episodes
""",
    version="2.0.0",
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.exists(static_dir):
    app.mount("/static", StaticFiles(directory=static_dir), name="static")

# Global env for /reset and /step (single user session)
env = ClinicalTriageEnvironment()

# ── Request schemas ───────────────────────────────────────────────────────────

class ResetRequest(BaseModel):
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(
        None,
        description="Episode difficulty. Defaults to 'medium'.",
        example="medium"
    )
    model_config = {"json_schema_extra": {"example": {"difficulty": "medium"}}}


class StepRequest(BaseModel):
    triage_level: Literal["IMMEDIATE", "URGENT", "STANDARD", "LOW"] = Field(
        ...,
        description="Severity classification. One of: IMMEDIATE, URGENT, STANDARD, LOW",
        example="IMMEDIATE"
    )
    disposition: Literal["ICU", "GENERAL", "OBSERVATION", "DISCHARGE"] = Field(
        ...,
        description="Ward assignment. One of: ICU, GENERAL, OBSERVATION, DISCHARGE",
        example="ICU"
    )
    model_config = {"json_schema_extra": {"example": {"triage_level": "IMMEDIATE", "disposition": "ICU"}}}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/", tags=["General"], include_in_schema=False)
def root():
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {"status": "ok", "env": "clinical-triage", "team": "VisionVerse", "version": "2.0.0"}


@app.get("/health", tags=["General"],
    summary="Health check",
    response_description="Status ok — automated validators ping this endpoint")
def health():
    """Returns 200 with status ok. Used by automated validation."""
    return {"status": "ok", "env": "clinical-triage", "team": "VisionVerse", "version": "2.0.0"}


@app.post("/reset", tags=["OpenEnv API"],
    summary="Start a new episode",
    response_description="Initial observation: first synthetic patient + hospital resources")
def reset(req: ResetRequest = None):
    """
    Start a new episode. Returns initial observation.

    **Difficulty levels:**
    - `easy`: 4 patients, clear symptoms, 5 ICU beds, 4 doctors
    - `medium`: 8 patients, mixed severity, 3 ICU beds, 2 doctors
    - `hard`: 12 patients, subtle symptoms, 1 ICU bed, 1 doctor
    """
    if req is None:
        req = ResetRequest()
    obs = env.reset(difficulty=req.difficulty)
    # FIX: use model_dump() instead of deprecated .dict()
    return obs.model_dump()


@app.post("/step", tags=["OpenEnv API"],
    summary="Submit a triage decision",
    response_description="Reward (0.0–1.0), next observation, done flag, and info dict")
def step(req: StepRequest):
    """
    Submit triage decision for current patient. Returns reward and next patient.

    **Reward components:**
    - Triage level accuracy: 0.50 max
    - Disposition accuracy: 0.30 max
    - Safety penalty: up to -0.50 for missing critical patients
    - Resource penalty: -0.15 if ICU beds exhausted
    - Time bonus: +0.10 for fast response to critical patients

    **Edge cases handled:**
    - Invalid enum values → rejected by Pydantic (422 error)
    - Step before reset → 400 error with message
    - Step after done → 400 error with message
    - Resource exhaustion → penalised in reward, not crashed
    """
    try:
        action = Action(triage_level=req.triage_level, disposition=req.disposition)
        obs, reward, done, info = env.step(action)
        # FIX: use model_dump() instead of deprecated .dict()
        return {
            "observation": obs.model_dump(),
            "reward": reward.model_dump(),
            "done": done,
            "info": info
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/state", tags=["OpenEnv API"],
    summary="Get current environment state")
def state():
    """Returns complete environment state including history, scores, and resources."""
    return env.state()


@app.get("/tasks", tags=["Tasks & Grading"],
    summary="List all 3 tasks")
def list_tasks():
    """Returns all tasks with full descriptions, difficulty parameters, and expected scores."""
    return {"tasks": [
        {
            "id":          "task_easy",
            "name":        "Single Patient Triage",
            "difficulty":  "easy",
            "patients":    4,
            "icu_beds":    5,
            "doctors":     4,
            "description": "4 patients with clear, unambiguous symptoms. Goal: learn basic severity classification.",
            "challenge":   "Symptoms are obvious — correct triage requires recognising clear clinical signals.",
            "valid_triage_levels": ["IMMEDIATE", "URGENT", "STANDARD", "LOW"],
            "valid_dispositions":  ["ICU", "GENERAL", "OBSERVATION", "DISCHARGE"],
            "expected_baseline":   0.81,
        },
        {
            "id":          "task_medium",
            "name":        "Mixed Severity ER",
            "difficulty":  "medium",
            "patients":    8,
            "icu_beds":    3,
            "doctors":     2,
            "description": "8 patients with mixed severity. Goal: prioritise correctly under limited resources.",
            "challenge":   "Must balance classification accuracy with ICU bed scarcity.",
            "valid_triage_levels": ["IMMEDIATE", "URGENT", "STANDARD", "LOW"],
            "valid_dispositions":  ["ICU", "GENERAL", "OBSERVATION", "DISCHARGE"],
            "expected_baseline":   0.65,
        },
        {
            "id":          "task_hard",
            "name":        "Mass Casualty Event",
            "difficulty":  "hard",
            "patients":    12,
            "icu_beds":    1,
            "doctors":     1,
            "description": "12 patients, subtle symptoms, 1 ICU bed, 1 doctor. Goal: optimal decisions under extreme pressure.",
            "challenge":   "Subtle clinical signals + extreme resource scarcity + safety-critical consequences.",
            "valid_triage_levels": ["IMMEDIATE", "URGENT", "STANDARD", "LOW"],
            "valid_dispositions":  ["ICU", "GENERAL", "OBSERVATION", "DISCHARGE"],
            "expected_baseline":   0.55,
        },
    ]}


@app.post("/grade/{task_id}", tags=["Tasks & Grading"],
    summary="Run oracle grader for a task",
    response_description="Average score 0.0–1.0 across 5 episodes")
def grade_task(task_id: str):
    """
    Runs a deterministic oracle agent (uses classify_severity ground truth)
    against the specified task for 5 episodes. Returns average score 0.0–1.0.

    Grader logic:
    - Uses classify_severity() for ground truth on each patient
    - Submits the exact correct action every step
    - Scores represent oracle (upper bound) performance
    - Results are reproducible given the same random seed pattern
    """
    difficulty_map = {
        "task_easy":   "easy",
        "task_medium": "medium",
        "task_hard":   "hard",
    }
    if task_id not in difficulty_map:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown task_id '{task_id}'. Valid: task_easy, task_medium, task_hard"
        )

    # FIX: use a FRESH local env instance — never reuse the global env for grading
    # This prevents state contamination between /grade calls and /step calls
    grader_env = ClinicalTriageEnvironment()

    scores = []
    for _ in range(5):
        obs  = grader_env.reset(difficulty=difficulty_map[task_id])
        ep   = 0.0
        done = False
        info = {"step": 1}

        while not done:
            if not obs.patient:
                break
            tl, td  = classify_severity(obs.patient)
            obs, reward, done, info = grader_env.step(Action(triage_level=tl, disposition=td))
            # FIX: safely get score whether reward is object or float
            score = reward.score if hasattr(reward, "score") else float(reward)
            ep += score

        steps = info.get("step", 1) if isinstance(info, dict) else 1
        scores.append(round(ep / max(steps, 1), 3))

    avg = round(sum(scores) / len(scores), 3)
    return {
        "task_id":       task_id,
        "episodes_run":  5,
        "average_score": avg,
        "scores":        scores,
        "grader":        "oracle (classify_severity ground truth)",
        "score_range":   "0.0–1.0",
    }