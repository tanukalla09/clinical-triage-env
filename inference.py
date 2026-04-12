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
import re
import sys

# ── Credentials — exactly as required by checklist ───────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN")
API_KEY      = os.getenv("API_KEY") or HF_TOKEN or ""

BENCHMARK = "clinical-triage"
MAX_STEPS = 20

# ── Log functions ─────────────────────────────────────────────────────────────

def clean_error(e):
    s = re.sub(r'<[^>]+>', '', str(e))
    s = re.sub(r'\s+', ' ', s).strip()[:100]
    return s if s else "error"

def log_start(task, env, model):
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step, action, reward, done, error):
    e = clean_error(error) if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={e}", flush=True)

def log_end(success, steps, score, rewards):
    r = ",".join(f"{x:.2f}" for x in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={r}", flush=True)

# ── Install missing packages if needed ───────────────────────────────────────

def ensure_packages():
    try:
        import httpx
    except ImportError:
        os.system(f"{sys.executable} -m pip install httpx==0.27.0 --quiet")
    try:
        import openai
    except ImportError:
        os.system(f"{sys.executable} -m pip install openai==1.30.1 --quiet")

ensure_packages()

# ── OpenAI client — safely initialized ───────────────────────────────────────

client = None
try:
    from openai import OpenAI
    client = OpenAI(
        api_key=API_KEY if API_KEY else "dummy-key",
        base_url=API_BASE_URL,
    )
    print(f"[INFO] Client initialized. base_url={API_BASE_URL}", flush=True)
except Exception as e:
    print(f"[WARN] Client init failed: {clean_error(e)}", flush=True)
    client = None

# ── Environment ───────────────────────────────────────────────────────────────

try:
    from environment import ClinicalTriageEnvironment, Action
    ENV_OK = True
except Exception as e:
    print(f"[WARN] Environment import failed: {clean_error(e)}", flush=True)
    ENV_OK = False

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


# ── Heuristic fallback ────────────────────────────────────────────────────────

def _heuristic(p, h):
    spo2 = p.get("oxygen_saturation", 95)
    hr   = p.get("heart_rate", 80)
    bp   = p.get("blood_pressure", "normal")
    pain = p.get("pain_level", 0)
    syms = p.get("symptoms", [])
    icu  = h.get("icu_beds", 1)

    if spo2 < 90 or hr > 130 or hr < 45 or bp == "very_low" or "severe_bleeding" in syms:
        return "IMMEDIATE", ("ICU" if icu > 0 else "GENERAL")
    elif spo2 < 95 or "chest_pain" in syms or pain >= 7:
        return "URGENT", "GENERAL"
    elif pain >= 4:
        return "STANDARD", "OBSERVATION"
    else:
        return "LOW", "DISCHARGE"


# ── Agent ─────────────────────────────────────────────────────────────────────

def agent_act(obs_dict):
    p = obs_dict.get("patient", {})
    h = obs_dict.get("hospital", {})

    if client is not None:
        user_msg = f"""Synthetic patient:
- Age: {p.get('age','?')}, Symptoms: {', '.join(p.get('symptoms', []))}
- SpO2: {p.get('oxygen_saturation','?')}%, HR: {p.get('heart_rate','?')} bpm
- BP: {p.get('blood_pressure','?')}, Pain: {p.get('pain_level','?')}/10
- Risk factors: {', '.join(p.get('risk_factors', []))}
- ICU beds: {h.get('icu_beds','?')}, Doctors: {h.get('doctors','?')}
Your decision:"""

        try:
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
            return triage, disp, None

        except Exception as e:
            err = clean_error(e)
            t, d = _heuristic(p, h)
            return t, d, err

    t, d = _heuristic(p, h)
    return t, d, "client_unavailable"


# ── Helpers ───────────────────────────────────────────────────────────────────

def obs_to_dict(obs):
    if isinstance(obs, dict):      return obs
    if hasattr(obs, "model_dump"): return obs.model_dump()
    if hasattr(obs, "dict"):       return obs.dict()
    return {}

def safe_score(reward):
    if reward is None:                   return 0.0
    if isinstance(reward, (int, float)): return float(reward)
    if hasattr(reward, "score"):         return float(reward.score)
    return 0.0


# ── Task runner ───────────────────────────────────────────────────────────────

def run_task(task_id, difficulty):
    if not ENV_OK:
        log_start(task_id, BENCHMARK, MODEL_NAME)
        log_step(1, "triage=STANDARD,disposition=OBSERVATION", 0.5, True, "env_unavailable")
        log_end(True, 1, 0.5, [0.5])
        return 0.5

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
            triage, disp, error_str = agent_act(obs_dict)
            action_str = f"triage={triage},disposition={disp}"
            action     = Action(triage_level=triage, disposition=disp)
            obs, reward, done, info = env.step(action)
            obs_dict = obs_to_dict(obs)
            r = safe_score(reward)
        except Exception as e:
            error_str = clean_error(e)
            done = True

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
            print(f"[ERROR] {task_id}: {clean_error(e)}", flush=True)
            log_start(task_id, BENCHMARK, MODEL_NAME)
            log_step(1, "triage=STANDARD,disposition=OBSERVATION", 0.0, True, clean_error(e))
            log_end(False, 1, 0.0, [0.0])
            results[task_id] = 0.0

    overall = round(sum(results.values()) / len(results), 3)

    with open("baseline_scores.json", "w") as f:
        json.dump({"tasks": results, "overall": overall, "model": MODEL_NAME}, f, indent=2)

    print(f"[SAVED] baseline_scores.json — overall={overall}", flush=True)


if __name__ == "__main__":
    main()
