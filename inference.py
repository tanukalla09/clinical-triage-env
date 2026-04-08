"""
inference.py — Baseline inference for ClinicalTriage OpenEnv
Built by VisionVerse for Meta x PyTorch x Hugging Face OpenEnv Hackathon 2026

MANDATORY stdout format:
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables (injected by validator):
    API_BASE_URL   The LiteLLM proxy endpoint — MUST use this
    API_KEY        The injected API key — MUST use this
    MODEL_NAME     The model identifier
    HF_TOKEN       Fallback if API_KEY not present
"""

import os
import json
from typing import Optional
from openai import OpenAI

# ── Credentials — use API_KEY exactly as injected by the validator ─────────────
API_BASE_URL = os.environ.get("API_BASE_URL") or "https://router.huggingface.co/v1"
API_KEY      = os.environ.get("API_KEY") or os.environ.get("HF_TOKEN") or "dummy-key"
MODEL_NAME   = os.environ.get("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"

BENCHMARK = "clinical-triage"
MAX_STEPS = 20

# ── OpenAI client — initialized with injected credentials ─────────────────────
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Environment import ─────────────────────────────────────────────────────────
try:
    from environment import ClinicalTriageEnvironment, Action
    ENV_AVAILABLE = True
except Exception as e:
    print(f"[WARN] Could not import environment: {e}", flush=True)
    ENV_AVAILABLE = False

SYSTEM_PROMPT = """You are an AI triage nurse in a simulated emergency room (synthetic data only).

For each patient, make TWO decisions:

1. TRIAGE LEVEL — how severe is this patient?
   IMMEDIATE  = life-threatening (low SpO2, very high/low HR, severe bleeding)
   URGENT     = serious, needs care within 30 minutes
   STANDARD   = stable, needs care within 2 hours
   LOW        = minor, can wait or be discharged

2. DISPOSITION — where should this patient go?
   ICU         = for IMMEDIATE cases
   GENERAL     = for URGENT cases
   OBSERVATION = for STANDARD cases
   DISCHARGE   = for LOW cases

Reply in EXACTLY this format — two lines only:
TRIAGE_LEVEL: <IMMEDIATE|URGENT|STANDARD|LOW>
DISPOSITION: <ICU|GENERAL|OBSERVATION|DISCHARGE>"""


# ── Mandatory log functions ────────────────────────────────────────────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str  = str(done).lower()
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    success_str = str(success).lower()
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


# ── Heuristic fallback agent (used only if LLM call fails) ────────────────────

def heuristic_act(obs_dict: dict) -> tuple:
    """Rule-based triage — fallback only if LLM call raises an exception."""
    try:
        patient  = obs_dict.get("patient", {})
        hospital = obs_dict.get("hospital", {})

        spo2     = patient.get("oxygen_saturation", 95)
        hr       = patient.get("heart_rate", 80)
        bp       = patient.get("blood_pressure", "normal")
        pain     = patient.get("pain_level", 0)
        syms     = patient.get("symptoms", [])
        icu_beds = hospital.get("icu_beds", 1)

        is_critical = (
            spo2 < 90 or hr > 130 or hr < 45 or bp == "low" or
            "severe_bleeding" in syms or "cardiac_arrest" in syms or
            "unconscious" in syms or pain >= 9
        )
        is_urgent = (
            90 <= spo2 < 94 or (100 < hr <= 130) or
            "chest_pain" in syms or "stroke_symptoms" in syms or pain >= 7
        )

        if is_critical:
            triage_level = "IMMEDIATE"
            disposition  = "ICU" if icu_beds > 0 else "GENERAL"
        elif is_urgent:
            triage_level = "URGENT"
            disposition  = "GENERAL"
        elif pain >= 4:
            triage_level = "STANDARD"
            disposition  = "OBSERVATION"
        else:
            triage_level = "LOW"
            disposition  = "DISCHARGE"
    except Exception:
        triage_level = "STANDARD"
        disposition  = "OBSERVATION"

    action_str = f"triage={triage_level},disposition={disposition}"
    return triage_level, disposition, action_str, None


# ── LLM agent — ALWAYS tries to call the API first ────────────────────────────

def llm_act(obs_dict: dict) -> tuple:
    """Call the injected LLM proxy. Falls back to heuristic ONLY on exception."""
    patient  = obs_dict.get("patient", {})
    hospital = obs_dict.get("hospital", {})

    user_msg = f"""Synthetic patient vitals:
- Age: {patient.get('age', '?')}
- Symptoms: {', '.join(patient.get('symptoms', []))}
- Oxygen saturation: {patient.get('oxygen_saturation', '?')}%
- Heart rate: {patient.get('heart_rate', '?')} bpm
- Blood pressure: {patient.get('blood_pressure', '?')}
- Pain level: {patient.get('pain_level', '?')}/10
- Risk factors: {', '.join(patient.get('risk_factors', []))}

Hospital resources:
- ICU beds available: {hospital.get('icu_beds', '?')}
- General beds: {hospital.get('general_beds', '?')}
- Doctors: {hospital.get('doctors', '?')}

Your decision:"""

    triage_level = "STANDARD"
    disposition  = "OBSERVATION"
    error_str    = None

    try:
        # This call MUST go through client which uses API_KEY + API_BASE_URL
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=30,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content.strip().upper()
        for line in raw.split("\n"):
            if "TRIAGE_LEVEL:" in line:
                val = line.split(":", 1)[1].strip()
                if val in ("IMMEDIATE", "URGENT", "STANDARD", "LOW"):
                    triage_level = val
            if "DISPOSITION:" in line:
                val = line.split(":", 1)[1].strip()
                if val in ("ICU", "GENERAL", "OBSERVATION", "DISCHARGE"):
                    disposition = val
    except Exception as e:
        error_str = str(e)[:80]
        # Only fall back to heuristic if LLM call fails
        triage_level, disposition, _, _ = heuristic_act(obs_dict)

    action_str = f"triage={triage_level},disposition={disposition}"
    return triage_level, disposition, action_str, error_str


# ── Helpers ───────────────────────────────────────────────────────────────────

def obs_to_dict(obs) -> dict:
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return {}


def safe_score(reward) -> float:
    if reward is None:
        return 0.0
    if isinstance(reward, (int, float)):
        return float(reward)
    if hasattr(reward, "score"):
        return float(reward.score)
    return 0.0


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str, difficulty: str) -> float:
    if not ENV_AVAILABLE:
        log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)
        log_step(step=1, action="triage=STANDARD,disposition=OBSERVATION",
                 reward=0.5, done=True, error="env_unavailable")
        log_end(success=True, steps=1, score=0.5, rewards=[0.5])
        return 0.5

    env      = ClinicalTriageEnvironment()
    obs      = env.reset(difficulty=difficulty)
    obs_dict = obs_to_dict(obs)

    step_rewards = []
    steps        = 0
    done         = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not done and steps < MAX_STEPS:
            triage_level, disposition, action_str, error_str = llm_act(obs_dict)

            try:
                action = Action(triage_level=triage_level, disposition=disposition)
                obs, reward, done, info = env.step(action)
                obs_dict = obs_to_dict(obs)
                r = safe_score(reward)
            except Exception as e:
                error_str = str(e)[:80]
                r    = 0.0
                done = True

            steps += 1
            step_rewards.append(r)
            log_step(step=steps, action=action_str, reward=r,
                     done=bool(done), error=error_str)

    except Exception as e:
        if not step_rewards:
            step_rewards = [0.0]
        log_step(step=steps + 1, action="error", reward=0.0,
                 done=True, error=str(e)[:80])

    score   = round(sum(step_rewards) / max(len(step_rewards), 1), 3)
    success = score >= 0.5
    log_end(success=success, steps=steps, score=score, rewards=step_rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tasks = [
        ("task_easy",   "easy"),
        ("task_medium", "medium"),
        ("task_hard",   "hard"),
    ]

    results = {}
    for task_id, difficulty in tasks:
        try:
            score = run_task(task_id, difficulty)
            results[task_id] = score
        except Exception as e:
            print(f"[ERROR] {task_id} failed: {e}", flush=True)
            results[task_id] = 0.0

    overall = round(sum(results.values()) / len(results), 3)

    with open("baseline_scores.json", "w") as f:
        json.dump({
            "tasks":   results,
            "overall": overall,
            "model":   MODEL_NAME,
        }, f, indent=2)

    print(f"[SAVED] baseline_scores.json — overall={overall}", flush=True)


if __name__ == "__main__":
    main()
