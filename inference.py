"""
inference.py — Baseline inference for ClinicalTriage OpenEnv
Built by VisionVerse

MANDATORY structured stdout logging: [START], [STEP], [END] format.
Deviation from this format will cause incorrect evaluation scoring.

Usage (Windows):
    set API_BASE_URL=https://api.openai.com/v1
    set MODEL_NAME=gpt-4o-mini
    set HF_TOKEN=your_hf_token_here
    python inference.py

Usage (Mac/Linux):
    export API_BASE_URL=https://api.openai.com/v1
    export MODEL_NAME=gpt-4o-mini
    export HF_TOKEN=your_hf_token_here
    python inference.py

Synthetic simulation only. Not for real medical use.
"""

import os
import json
import time
from openai import OpenAI
from environment import ClinicalTriageEnvironment, Action

# ── Credentials from environment variables (NEVER hardcode these) ─────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")   # ← NEVER put your token here

# HF_TOKEN is used as the API key because on Hugging Face Spaces,
# the LLM endpoint accepts your HF token as authentication.
client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

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


def _obs_to_dict(obs) -> dict:
    """Safely convert observation to dict whether it's a Pydantic model or already a dict."""
    if isinstance(obs, dict):
        return obs
    if hasattr(obs, "model_dump"):
        return obs.model_dump()
    if hasattr(obs, "dict"):
        return obs.dict()
    return {}


def _safe_score(reward) -> float:
    """Safely extract score from reward whether it's a Pydantic model or float."""
    if isinstance(reward, (int, float)):
        return float(reward)
    if hasattr(reward, "score"):
        return float(reward.score)
    return 0.0


def _safe_reason(reward) -> str:
    """Safely extract reason string from reward."""
    if hasattr(reward, "reason"):
        return str(reward.reason)
    return ""


def _safe_breakdown(reward) -> dict:
    """Safely extract breakdown dict from reward."""
    if hasattr(reward, "breakdown"):
        return reward.breakdown or {}
    return {}


def agent_act(obs: dict) -> Action:
    """Call LLM and parse triage decision."""
    patient  = obs.get("patient", {})
    hospital = obs.get("hospital", {})

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

        triage_level = "STANDARD"
        disposition  = "OBSERVATION"

        for line in lines:
            if "TRIAGE_LEVEL:" in line:
                val = line.split(":", 1)[1].strip()
                if val in ("IMMEDIATE", "URGENT", "STANDARD", "LOW"):
                    triage_level = val
            if "DISPOSITION:" in line:
                val = line.split(":", 1)[1].strip()
                if val in ("ICU", "GENERAL", "OBSERVATION", "DISCHARGE"):
                    disposition = val

        return Action(triage_level=triage_level, disposition=disposition)

    except Exception as e:
        # ── [LLM_ERROR] log ────────────────────────────────────────────────────
        print(json.dumps({"type": "LLM_ERROR", "error": str(e)}), flush=True)
        return Action(triage_level="STANDARD", disposition="OBSERVATION")


def run_task(task_id: str, difficulty: str, task_name: str, episodes: int = 5) -> float:
    """
    Run task and emit MANDATORY structured logs.
    Format: [START], [STEP], [END] — required by evaluation system.
    ANY deviation will cause incorrect evaluation scoring.
    """
    env = ClinicalTriageEnvironment()   # fresh instance per task — no shared state
    ep_scores = []

    for ep in range(1, episodes + 1):
        obs      = env.reset(difficulty=difficulty)
        obs_dict = _obs_to_dict(obs)
        ep_score = 0.0
        steps    = 0
        done     = False
        info     = {}

        # ── [START] MANDATORY LOG FORMAT ──────────────────────────────────────
        print(f'[START] {json.dumps({"task_id": task_id, "episode": ep, "difficulty": difficulty, "model": MODEL_NAME, "total_patients": obs_dict.get("total_patients", 0)})}', flush=True)

        while not done:
            action          = agent_act(obs_dict)
            obs, reward, done, info = env.step(action)
            obs_dict        = _obs_to_dict(obs)
            score           = _safe_score(reward)
            ep_score       += score
            steps          += 1

            # ── [STEP] MANDATORY LOG FORMAT ────────────────────────────────────
            print(f'[STEP] {json.dumps({"task_id": task_id, "episode": ep, "step": steps, "triage_level": action.triage_level, "disposition": action.disposition, "true_level": info.get("true_triage_level", ""), "true_disp": info.get("true_disposition", ""), "score": score, "reason": _safe_reason(reward), "breakdown": _safe_breakdown(reward)})}', flush=True)

        normalised = round(ep_score / max(steps, 1), 3)
        ep_scores.append(normalised)

        # ── [END] MANDATORY LOG FORMAT ─────────────────────────────────────────
        print(f'[END] {json.dumps({"task_id": task_id, "episode": ep, "steps": steps, "episode_score": normalised, "cumulative_score": info.get("cumulative_score", 0)})}', flush=True)

    avg = round(sum(ep_scores) / len(ep_scores), 3)
    return avg


def main():
    start_time = time.time()

    # Run start summary
    print(f'[RUN_START] {json.dumps({"env": "clinical-triage", "team": "VisionVerse", "model": MODEL_NAME, "api_url": API_BASE_URL, "note": "Synthetic simulation only. Not for real medical use."})}', flush=True)

    tasks = [
        ("task_easy",   "easy",   "Single Patient Triage"),
        ("task_medium", "medium", "Mixed Severity ER"),
        ("task_hard",   "hard",   "Mass Casualty Event"),
    ]

    results = {}
    for task_id, difficulty, task_name in tasks:
        avg = run_task(task_id, difficulty, task_name, episodes=5)
        results[task_id] = avg

    overall = round(sum(results.values()) / len(results), 3)
    elapsed = round(time.time() - start_time, 1)

    # Run end summary
    print(f'[RUN_END] {json.dumps({"task_scores": results, "overall": overall, "elapsed_sec": elapsed, "model": MODEL_NAME})}', flush=True)

    # Save scores to file
    with open("baseline_scores.json", "w") as f:
        json.dump({"tasks": results, "overall": overall, "model": MODEL_NAME}, f, indent=2)

    print(f'[SAVED] {json.dumps({"file": "baseline_scores.json"})}', flush=True)


if __name__ == "__main__":
    main()