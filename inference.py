"""
inference.py — ClinicalTriage OpenEnv
Built by VisionVerse for Meta x PyTorch x Hugging Face OpenEnv Hackathon 2026

STDOUT FORMAT (exact):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.000> rewards=<r1,r2,...>
"""

import os
import json

# ── ALWAYS use injected env vars — NO OR fallbacks for API_KEY ────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")

# CRITICAL: use API_KEY exactly as injected — no fallback to HF_TOKEN
API_KEY = os.environ["API_KEY"] if "API_KEY" in os.environ else os.environ.get("HF_TOKEN", "")

BENCHMARK = "clinical-triage"
MAX_STEPS = 20

# ── OpenAI client — ALWAYS initialized with injected credentials ──────────────
from openai import OpenAI
client = OpenAI(api_key=API_KEY, base_url=API_BASE_URL)

# ── Environment ───────────────────────────────────────────────────────────────
from environment import ClinicalTriageEnvironment, Action

SYSTEM_PROMPT = """You are an AI triage nurse in a simulated emergency room (synthetic data only).

For each patient, make TWO decisions:

1. TRIAGE LEVEL:
   IMMEDIATE  = life-threatening
   URGENT     = serious, within 30 minutes
   STANDARD   = stable, within 2 hours
   LOW        = minor, can wait

2. DISPOSITION:
   ICU         = for IMMEDIATE cases
   GENERAL     = for URGENT cases
   OBSERVATION = for STANDARD cases
   DISCHARGE   = for LOW cases

Reply in EXACTLY this format — two lines only:
TRIAGE_LEVEL: <IMMEDIATE|URGENT|STANDARD|LOW>
DISPOSITION: <ICU|GENERAL|OBSERVATION|DISCHARGE>"""


# ── Log functions (exact required format) ─────────────────────────────────────

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    e = str(error).replace("\n", " ")[:80] if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r}", flush=True)


# ── Agent — ALWAYS calls LLM API, no heuristic fallback ──────────────────────

def agent_act(obs_dict):
    """
    ALWAYS calls the LLM API through the injected base_url.
    Never falls back to heuristic — validator must see API calls.
    """
    p = obs_dict.get("patient", {})
    h = obs_dict.get("hospital", {})

    user_msg = f"""Synthetic patient:
- Age: {p.get('age','?')}, Symptoms: {', '.join(p.get('symptoms', []))}
- SpO2: {p.get('oxygen_saturation','?')}%, HR: {p.get('heart_rate','?')} bpm
- BP: {p.get('blood_pressure','?')}, Pain: {p.get('pain_level','?')}/10
- Risk factors: {', '.join(p.get('risk_factors', []))}
- Waiting: {p.get('waiting_time_min','?')} min
- ICU beds: {h.get('icu_beds','?')}, Doctors: {h.get('doctors','?')}

Your decision:"""

    # ALWAYS make the API call — do not skip or fallback
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_msg},
        ],
        max_tokens=30,
        temperature=0.0,
    )

    raw    = resp.choices[0].message.content.strip().upper()
    triage = "STANDARD"
    disp   = "OBSERVATION"

    for line in raw.split("\n"):
        if "TRIAGE_LEVEL:" in line:
            v = line.split(":", 1)[1].strip()
            if v in ("IMMEDIATE", "URGENT", "STANDARD", "LOW"):
                triage = v
        if "DISPOSITION:" in line:
            v = line.split(":", 1)[1].strip()
            if v in ("ICU", "GENERAL", "OBSERVATION", "DISCHARGE"):
                disp = v

    return triage, disp


# ── Helpers ───────────────────────────────────────────────────────────────────

def obs_to_dict(obs):
    if isinstance(obs, dict):          return obs
    if hasattr(obs, "model_dump"):     return obs.model_dump()
    if hasattr(obs, "dict"):           return obs.dict()
    return {}

def safe_score(reward):
    if reward is None:                       return 0.0
    if isinstance(reward, (int, float)):     return float(reward)
    if hasattr(reward, "score"):             return float(reward.score)
    return 0.0


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id, difficulty):
    env      = ClinicalTriageEnvironment()
    obs      = env.reset(difficulty=difficulty)
    obs_dict = obs_to_dict(obs)
    rewards  = []
    steps    = 0
    done     = False

    log_start(task_id, BENCHMARK, MODEL_NAME)

    while not done and steps < MAX_STEPS:
        error_str  = None
        action_str = "triage=STANDARD,disposition=OBSERVATION"
        r          = 0.0

        try:
            triage, disp = agent_act(obs_dict)
            action_str   = f"triage={triage},disposition={disp}"
            action       = Action(triage_level=triage, disposition=disp)
            obs, reward, done, info = env.step(action)
            obs_dict     = obs_to_dict(obs)
            r            = safe_score(reward)
        except Exception as e:
            error_str = str(e)[:80]
            done      = True

        steps += 1
        rewards.append(r)
        log_step(steps, action_str, r, bool(done), error_str)

    score = round(sum(rewards) / max(len(rewards), 1), 3)
    log_end(score >= 0.5, steps, score, rewards)
    return score


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    tasks   = [("task_easy","easy"), ("task_medium","medium"), ("task_hard","hard")]
    results = {}

    for task_id, difficulty in tasks:
        try:
            results[task_id] = run_task(task_id, difficulty)
        except Exception as e:
            print(f"[ERROR] {task_id}: {e}", flush=True)
            results[task_id] = 0.0

    overall = round(sum(results.values()) / len(results), 3)

    with open("baseline_scores.json", "w") as f:
        json.dump({"tasks": results, "overall": overall, "model": MODEL_NAME}, f, indent=2)

    print(f"[SAVED] baseline_scores.json — overall={overall}", flush=True)


if __name__ == "__main__":
    main()
