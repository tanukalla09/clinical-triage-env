"""
ClinicalTriage OpenEnv
Built by VisionVerse for Meta x PyTorch x Hugging Face OpenEnv Hackathon

Emergency room triage simulation for RL agents.
Synthetic data only. Not for real medical use.
"""

import random
from typing import Optional, Literal
from pydantic import BaseModel, Field

# ── Enums (enforced by Pydantic — invalid values auto-rejected) ───────────────

TriageLevel = Literal["IMMEDIATE", "URGENT", "STANDARD", "LOW"]
Disposition  = Literal["ICU", "GENERAL", "OBSERVATION", "DISCHARGE"]

# Ordering for off-by-one calculation
LEVEL_ORDER = {"IMMEDIATE": 4, "URGENT": 3, "STANDARD": 2, "LOW": 1}

# Canonical mapping: triage level → correct disposition
LEVEL_TO_DISP = {
    "IMMEDIATE": "ICU",
    "URGENT":    "GENERAL",
    "STANDARD":  "OBSERVATION",
    "LOW":       "DISCHARGE",
}

SYMPTOMS_POOL = [
    "chest_pain", "shortness_of_breath", "confusion",
    "severe_bleeding", "high_fever", "abdominal_pain",
    "headache", "dizziness", "nausea", "mild_cough",
    "back_pain", "rash", "minor_laceration"
]
RISK_POOL = ["diabetes", "hypertension", "elderly", "pregnant", "cardiac_history", "none"]

# ── Typed Models (OpenEnv spec) ───────────────────────────────────────────────

class Observation(BaseModel):
    patient:          dict  = Field(..., description="Synthetic patient vitals and symptoms")
    hospital:         dict  = Field(..., description="Current hospital resource availability")
    queue_length:     int   = Field(..., description="Patients still waiting to be triaged")
    step_num:         int   = Field(..., description="Current step number in this episode")
    total_patients:   int   = Field(..., description="Total patients in this episode")
    patients_handled: int   = Field(..., description="Patients already triaged")
    episode_done:     bool  = Field(False, description="True when all patients have been triaged")

class Action(BaseModel):
    triage_level: TriageLevel = Field(..., description="Severity: IMMEDIATE | URGENT | STANDARD | LOW")
    disposition:  Disposition = Field(..., description="Ward: ICU | GENERAL | OBSERVATION | DISCHARGE")

class Reward(BaseModel):
    score:     float = Field(..., ge=0.0, le=1.0, description="Normalised score 0.0–1.0")
    raw_score: float = Field(..., description="Raw reward before normalisation")
    reason:    str   = Field(..., description="Human-readable explanation")
    breakdown: dict  = Field(..., description="Per-component score breakdown")

# ── Deterministic severity classifier ────────────────────────────────────────

def classify_severity(patient: dict) -> tuple:
    """
    Deterministic rules mapping patient vitals → (triage_level, disposition).
    This is the ground truth used by graders.

    Rules (in priority order):
      IMMEDIATE / ICU:
        - chest_pain AND oxygen_saturation < 92
        - confusion AND (heart_rate > 130 OR temperature > 39.5)
        - severe_bleeding AND pain_level >= 7
        - oxygen_saturation < 88
        - heart_rate > 150 OR heart_rate < 40

      URGENT / GENERAL:
        - shortness_of_breath AND oxygen_saturation < 95
        - chest_pain AND cardiac_history risk factor
        - elderly risk AND pain_level >= 7
        - temperature > 39.0 AND confusion symptom
        - blood_pressure == very_low

      STANDARD / OBSERVATION:
        - temperature > 38.5
        - abdominal_pain AND pain_level >= 5
        - high_fever symptom
        - pain_level >= 6

      LOW / DISCHARGE: all other cases
    """
    symptoms = set(patient.get("symptoms", []))
    risks    = set(patient.get("risk_factors", []))
    spo2     = patient.get("oxygen_saturation", 98)
    hr       = patient.get("heart_rate", 75)
    temp     = patient.get("temperature", 37.0)
    pain     = patient.get("pain_level", 0)
    bp       = patient.get("blood_pressure", "normal")

    if "chest_pain" in symptoms and spo2 < 92:                     return "IMMEDIATE", "ICU"
    if "confusion"  in symptoms and (hr > 130 or temp > 39.5):     return "IMMEDIATE", "ICU"
    if "severe_bleeding" in symptoms and pain >= 7:                 return "IMMEDIATE", "ICU"
    if spo2 < 88:                                                   return "IMMEDIATE", "ICU"
    if hr > 150 or hr < 40:                                         return "IMMEDIATE", "ICU"

    if "shortness_of_breath" in symptoms and spo2 < 95:            return "URGENT", "GENERAL"
    if "chest_pain" in symptoms and "cardiac_history" in risks:     return "URGENT", "GENERAL"
    if "elderly" in risks and pain >= 7:                            return "URGENT", "GENERAL"
    if temp > 39.0 and "confusion" in symptoms:                     return "URGENT", "GENERAL"
    if bp == "very_low":                                            return "URGENT", "GENERAL"

    if temp > 38.5:                                                 return "STANDARD", "OBSERVATION"
    if "abdominal_pain" in symptoms and pain >= 5:                  return "STANDARD", "OBSERVATION"
    if "high_fever" in symptoms:                                    return "STANDARD", "OBSERVATION"
    if pain >= 6:                                                   return "STANDARD", "OBSERVATION"

    return "LOW", "DISCHARGE"

# ── Patient generator ─────────────────────────────────────────────────────────

def generate_patient(difficulty: str = "medium", patient_id: int = 0) -> dict:
    """Generate a synthetic patient. All data is procedurally generated."""
    if difficulty == "easy":
        templates = [
            {"symptoms": ["chest_pain"],     "oxygen_saturation": 88,  "heart_rate": 145, "blood_pressure": "low",    "temperature": 37.0, "pain_level": 9, "risk_factors": ["cardiac_history"], "age": 65},
            {"symptoms": ["mild_cough"],      "oxygen_saturation": 99,  "heart_rate": 72,  "blood_pressure": "normal", "temperature": 37.1, "pain_level": 1, "risk_factors": ["none"],            "age": 28},
            {"symptoms": ["headache"],        "oxygen_saturation": 98,  "heart_rate": 80,  "blood_pressure": "normal", "temperature": 37.0, "pain_level": 3, "risk_factors": ["none"],            "age": 34},
            {"symptoms": ["severe_bleeding"], "oxygen_saturation": 91,  "heart_rate": 132, "blood_pressure": "low",    "temperature": 37.5, "pain_level": 8, "risk_factors": ["none"],            "age": 40},
        ]
        t = random.choice(templates)
    elif difficulty == "hard":
        templates = [
            {"symptoms": ["dizziness","nausea"],    "oxygen_saturation": 94, "heart_rate": 105, "blood_pressure": "low",      "temperature": 38.8, "pain_level": 5, "risk_factors": ["elderly","diabetes"],        "age": 78},
            {"symptoms": ["back_pain","rash"],      "oxygen_saturation": 96, "heart_rate": 98,  "blood_pressure": "normal",   "temperature": 39.2, "pain_level": 6, "risk_factors": ["hypertension"],              "age": 55},
            {"symptoms": ["shortness_of_breath"],   "oxygen_saturation": 93, "heart_rate": 118, "blood_pressure": "normal",   "temperature": 37.4, "pain_level": 4, "risk_factors": ["pregnant"],                  "age": 29},
            {"symptoms": ["confusion","high_fever"],"oxygen_saturation": 95, "heart_rate": 128, "blood_pressure": "very_low", "temperature": 40.1, "pain_level": 3, "risk_factors": ["elderly"],                   "age": 82},
            {"symptoms": ["chest_pain"],            "oxygen_saturation": 97, "heart_rate": 88,  "blood_pressure": "high",     "temperature": 37.0, "pain_level": 5, "risk_factors": ["cardiac_history","diabetes"], "age": 58},
        ]
        t = random.choice(templates)
    else:
        t = {
            "symptoms":          random.sample(SYMPTOMS_POOL, k=random.randint(1, 3)),
            "oxygen_saturation": random.randint(82, 100),
            "heart_rate":        random.randint(50, 155),
            "blood_pressure":    random.choice(["very_low","low","normal","high"]),
            "temperature":       round(random.uniform(36.5, 40.5), 1),
            "pain_level":        random.randint(0, 10),
            "risk_factors":      random.sample(RISK_POOL, k=random.randint(1, 2)),
            "age":               random.randint(5, 90),
        }

    return {
        "patient_id":        f"P{patient_id:03d}",
        "age":               t["age"],
        "symptoms":          t["symptoms"],
        "oxygen_saturation": t["oxygen_saturation"],
        "heart_rate":        t["heart_rate"],
        "blood_pressure":    t["blood_pressure"],
        "temperature":       t["temperature"],
        "pain_level":        t["pain_level"],
        "risk_factors":      t["risk_factors"],
        "waiting_time_min":  random.randint(0, 20),
    }

# ── Environment ───────────────────────────────────────────────────────────────

class ClinicalTriageEnvironment:
    """
    ClinicalTriage OpenEnv — Built by VisionVerse
    Implements the full OpenEnv interface: reset() / step() / state()
    """

    def __init__(self):
        self.patients         = []
        self.current_index    = 0
        self.hospital         = {}
        self.cumulative_score = 0.0
        self.step_count       = 0
        self.history          = []
        self.difficulty       = "medium"
        self._episode_done    = False

    def reset(self, difficulty: Optional[str] = None) -> Observation:
        """
        Start a new episode. Generates synthetic patient queue.
        Returns initial Observation.
        """
        self.difficulty       = difficulty or "medium"
        self.step_count       = 0
        self.cumulative_score = 0.0
        self.history          = []
        self.current_index    = 0
        self._episode_done    = False

        num = {"easy": 4, "medium": 8, "hard": 12}.get(self.difficulty, 8)
        self.patients = [generate_patient(self.difficulty, i) for i in range(num)]

        if self.difficulty == "easy":
            self.hospital = {"icu_beds": 5, "general_beds": 20, "doctors": 4, "nurses": 8}
        elif self.difficulty == "hard":
            self.hospital = {"icu_beds": 1, "general_beds": 5,  "doctors": 1, "nurses": 3}
        else:
            self.hospital = {"icu_beds": 3, "general_beds": 12, "doctors": 2, "nurses": 5}

        return self._make_observation()

    def step(self, action: Action) -> tuple:
        """
        Submit triage decision for current patient.
        Returns (Observation, Reward, done: bool, info: dict)

        Edge cases handled:
          - step() before reset()     → ValueError
          - step() after done=True    → ValueError
          - no patients remaining     → ValueError
          - invalid action values     → rejected by Pydantic before reaching here
          - resource exhaustion       → penalised in reward, not crashed
        """
        if not self.patients:
            raise ValueError("Not initialised. Call reset() first.")
        if self._episode_done:
            raise ValueError("Episode finished. Call reset() to start a new one.")
        if self.current_index >= len(self.patients):
            raise ValueError("No more patients. Call reset().")

        patient               = self.patients[self.current_index]
        true_level, true_disp = classify_severity(patient)
        reward                = self._compute_reward(action, true_level, true_disp, patient)

        # Consume resources
        if action.disposition == "ICU":
            self.hospital["icu_beds"]     = max(0, self.hospital["icu_beds"] - 1)
        elif action.disposition in ("GENERAL", "OBSERVATION"):
            self.hospital["general_beds"] = max(0, self.hospital["general_beds"] - 1)

        self.cumulative_score += reward.score
        self.step_count       += 1
        self.history.append({
            "patient_id":  patient["patient_id"],
            "true_level":  true_level,
            "true_disp":   true_disp,
            "agent_level": action.triage_level,
            "agent_disp":  action.disposition,
            "score":       reward.score,
            "breakdown":   reward.breakdown,
        })

        self.current_index += 1
        done = self.current_index >= len(self.patients)
        if done:
            self._episode_done = True

        info = {
            "step":               self.step_count,
            "cumulative_score":   round(self.cumulative_score, 3),
            "true_triage_level":  true_level,
            "true_disposition":   true_disp,
            "level_diff":         abs(LEVEL_ORDER[action.triage_level] - LEVEL_ORDER[true_level]),
            "hospital_resources": dict(self.hospital),
            "patients_remaining": len(self.patients) - self.current_index,
        }
        return self._make_observation(), reward, done, info

    def state(self) -> dict:
        """Return complete current environment state."""
        return {
            "difficulty":         self.difficulty,
            "current_index":      self.current_index,
            "total_patients":     len(self.patients),
            "step_count":         self.step_count,
            "cumulative_score":   round(self.cumulative_score, 3),
            "average_score":      round(self.cumulative_score / max(self.step_count, 1), 3),
            "hospital_resources": dict(self.hospital),
            "episode_done":       self._episode_done,
            "history":            self.history,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _make_observation(self) -> Observation:
        patient = self.patients[self.current_index] if self.current_index < len(self.patients) else {}
        return Observation(
            patient=patient,
            hospital=dict(self.hospital),
            queue_length=max(0, len(self.patients) - self.current_index),
            step_num=self.step_count,
            total_patients=len(self.patients),
            patients_handled=self.current_index,
            episode_done=self._episode_done,
        )

    def _compute_reward(self, action: Action, true_level: str, true_disp: str, patient: dict) -> Reward:
        """
        Reward is computed per patient across 5 components:

        1. triage_level_score (max 0.50)
           - exact match → 0.50
           - off by 1 level → 0.25  (e.g. URGENT when correct is IMMEDIATE)
           - off by 2+ levels → 0.00

        2. disposition_score (max 0.30)
           - exact match → 0.30
           - sending patient home (DISCHARGE) when care needed → 0.00
           - ICU vs GENERAL mix-up → 0.10
           - other partial match → 0.15

        3. safety_penalty (negative only)
           - IMMEDIATE → STANDARD or LOW: -0.50 (life-threatening miss)
           - IMMEDIATE → URGENT: -0.10 (close but risky)
           - URGENT → LOW: -0.25 (dangerous delay)

        4. resource_penalty (negative only)
           - ICU assigned when icu_beds == 0: -0.15
           - GENERAL/OBSERVATION when general_beds == 0: -0.10

        5. time_bonus (positive only)
           - IMMEDIATE patient, waiting < 5 min, correct action: +0.10

        Final score = clip(sum of all components, 0.0, 1.0)
        """
        diff       = abs(LEVEL_ORDER[action.triage_level] - LEVEL_ORDER[true_level])

        # 1. Triage level
        if diff == 0:   level_score = 0.50
        elif diff == 1: level_score = 0.25
        else:           level_score = 0.00

        # 2. Disposition
        if action.disposition == true_disp:
            disp_score = 0.30
        elif action.disposition == "DISCHARGE" and true_disp != "DISCHARGE":
            disp_score = 0.00
        elif {action.disposition, true_disp} == {"ICU", "GENERAL"}:
            disp_score = 0.10
        else:
            disp_score = 0.15

        # 3. Safety penalty
        safety = 0.0
        if true_level == "IMMEDIATE" and action.triage_level in ("STANDARD", "LOW"): safety = -0.50
        elif true_level == "IMMEDIATE" and action.triage_level == "URGENT":          safety = -0.10
        elif true_level == "URGENT"    and action.triage_level == "LOW":             safety = -0.25

        # 4. Resource penalty
        resource = 0.0
        if action.disposition == "ICU" and self.hospital["icu_beds"] == 0:                          resource = -0.15
        elif action.disposition in ("GENERAL","OBSERVATION") and self.hospital["general_beds"] == 0: resource = -0.10

        # 5. Time bonus
        time_bonus = 0.0
        if true_level == "IMMEDIATE" and patient.get("waiting_time_min", 99) < 5 and diff == 0:
            time_bonus = 0.10

        raw        = level_score + disp_score + safety + resource + time_bonus
        normalised = round(min(1.0, max(0.0, raw)), 3)

        breakdown = {
            "triage_level_score": level_score,
            "disposition_score":  disp_score,
            "safety_penalty":     safety,
            "resource_penalty":   resource,
            "time_bonus":         time_bonus,
            "level_diff":         diff,
        }

        parts = [f"Triage: {action.triage_level} (correct={true_level}, diff={diff})"]
        parts.append(f"Disposition: {action.disposition} (correct={true_disp})")
        if safety   < 0: parts.append(f"SAFETY PENALTY ({safety}): critical patient under-triaged.")
        if resource < 0: parts.append(f"RESOURCE PENALTY ({resource}): bed unavailable.")
        if time_bonus > 0: parts.append("TIME BONUS: critical patient seen promptly.")

        return Reward(score=normalised, raw_score=round(raw, 3), reason=" | ".join(parts), breakdown=breakdown)