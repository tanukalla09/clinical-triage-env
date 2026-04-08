"""
inference.py — Baseline inference for ClinicalTriage OpenEnv
Built by VisionVerse for Meta x PyTorch x Hugging Face OpenEnv Hackathon 2026

MANDATORY stdout format (exactly as required):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>

Environment variables:
    API_BASE_URL   The API endpoint for the LLM.
    MODEL_NAME     The model identifier to use for inference.
    HF_TOKEN       Your Hugging Face / API key. (no default — must be set)

Synthetic simulation only. Not for real medical use.
"""

import os
from typing import Optional
from openai import OpenAI
from environment import ClinicalTriageEnvironment, Action

# ── Credentials — defaults only for API_BASE_URL and MODEL_NAME, NOT HF_TOKEN ─
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")   # ← NO default, must be set externally

API_KEY = HF_TOKEN or os.getenv("API_KEY", "")

client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

BENCHMARK  = "clinical-triage"
MAX_STEPS  = 20

SYSTEM_PROMPT = """You are an AI triage nurse in a simulated emergency room (synthetic data only).

For each patient, make TWO decisions:

1. TRIAGE LEVEL — how severe is this patient?
   IMMEDIATE  = life-threatening (low SpO2, very high/low HR, severe bleeding)
   URGENT     = serious, needs care within 30 minutes
   STANDARD   = stable, needs care within 2 hours
   LOW        = minor, can wait or be discharged

2. DISPOSITION — where should this patient go?
   ICU         = for IMMEDIATE cases (intensive care)
   GENERAL     = for URGENT cases (general ward)
   OBSERVATION = for STANDARD cases (monitor)
   DISCHARGE   = for LOW cases (safe to go home)

Reply in EXACTLY this format — two lines only:
TRIAGE_LEVEL: <level>
DISPOSITION: <disposition>"""


# ── Mandatory log functions — exact format required by hackathon ──────────────

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_str} error={error_str}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: list) -> None:
    success_str  = "true" if success else "false"
    rewards_str  = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={success_str} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)


# ── Agent ─────────────────────────────────────────────────────────────────────

def agent_act(obs: dict) -> tuple:
    """Call LLM and parse triage decision. Returns (Action, action_str, error_str)."""
    patient  = obs.get("patient", {}) if isinstance(obs, dict) else {}
    hospital = obs.get("hospital", {}) if isinstance(obs, dict) else {}

    user_msg = f"""Synthetic patient vitals:
- Age: {patient.get('age', '?')}
- Symptoms: {', '.join(patient.get('symptoms', []))}
- Oxygen saturation: {patient.get('oxygen_saturation', '?')}%
- Heart rate: {patient.get('heart_rate', '?')} bpm
- Blood pressure: {patient.get('blood_pressure', '?')}
- Temperature: {patient.get('temperature', '?')} C
- Pain level: {patient.get('pain_level', '?')}/10
- Risk factors: {', '.join(patient.get('risk_factors', []))}
- Waiting: {patient.get('waiting_time_min', '?')} min

Hospital resources:
- ICU beds available: {hospital.get('icu_beds', '?')}
- General beds available: {hospital.get('general_beds', '?')}
- Doctors on shift: {hospital.get('doctors', '?')}

Your decision:"""

    error_str    = None
    triage_level = "STANDARD"
    disposition  = "OBSERVATION"

    try:
        resp = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_tokens=20,
            temperature=0.0,
        )
        raw   = resp.choices[0].message.content.strip().upper()
        lines = raw.split("\n")

        for line in lines:
            if "TRIAGE_LEVEL:" in line:
                val = line.split(":", 1)[1].strip()
                if val in ("IMMEDIATE", "URGENT", "STANDARD", "LOW"):
                    triage_level = val
            if "DISPOSITION:" in line:
                val = line.split(":", 1)[1].strip()
                if val in ("ICU", "GENERAL", "OBSERVATION", "DISCHARGE"):
                    disposition = val

    except Exception as e:
        error_str = str(e)[:100]

    action     = Action(triage_level=triage_level, disposition=disposition)
    action_str = f"triage={triage_level},disposition={disposition}"
    return action, action_str, error_str


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id: str, difficulty: str) -> float:
    """
    Run one episode for the given task.
    Emits mandatory [START], [STEP], [END] logs.
    Returns normalized score for this episode.
    """
    env      = ClinicalTriageEnvironment()
    obs      = env.reset(difficulty=difficulty)
    obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else (obs.dict() if hasattr(obs, "dict") else obs)

    step_rewards = []
    steps        = 0
    done         = False
    info         = {}
    last_error   = None

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        while not done and steps < MAX_STEPS:
            action, action_str, error_str = agent_act(obs_dict)
            last_error = error_str

            obs, reward, done, info = env.step(action)
            obs_dict = obs.model_dump() if hasattr(obs, "model_dump") else (obs.dict() if hasattr(obs, "dict") else obs)

            # safely extract reward score
            if hasattr(reward, "score"):
                r = float(reward.score)
            elif isinstance(reward, (int, float)):
                r = float(reward)
            else:
                r = 0.0

            steps += 1
            step_rewards.append(r)

            log_step(
                step=steps,
                action=action_str,
                reward=r,
                done=bool(done),
                error=error_str,
            )

    except Exception as e:
        last_error = str(e)[:100]
        if not step_rewards:
            step_rewards = [0.0]
        log_step(step=steps + 1, action="error", reward=0.0, done=True, error=last_error)

    # normalize score to 0.0–1.0
    score   = round(sum(step_rewards) / max(len(step_rewards), 1), 2)
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
        score = run_task(task_id, difficulty)
        results[task_id] = score

    # Save scores
    import json
    with open("baseline_scores.json", "w") as f:
        json.dump({
            "tasks":   results,
            "overall": round(sum(results.values()) / len(results), 2),
            "model":   MODEL_NAME,
        }, f, indent=2)

    print(f"[SAVED] baseline_scores.json — overall={round(sum(results.values())/len(results),2)}", flush=True)


if __name__ == "__main__":
    main()
