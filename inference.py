"""
inference.py — Run DataCleaningEnv episodes locally (no HTTP needed)
=====================================================================
Runs the environment by importing env.py directly — works inside the
hackathon checker's Docker container and locally.

STDOUT FORMAT (required by hackathon checker):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=0.00 done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import math
import os
import re
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Force stdout unbuffered — required so checker sees output immediately
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Environment variables (hackathon spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

BENCHMARK               = "data-cleaning-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Import local environment directly (all files are in /app in Docker)
# ---------------------------------------------------------------------------
from env import DataCleaningEnv

# ---------------------------------------------------------------------------
# Structured stdout loggers — exact format the checker parses
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}", flush=True)

# ---------------------------------------------------------------------------
# NaN-safe helper — pandas returns float('nan') for missing, not None
# ---------------------------------------------------------------------------
def _is_missing(val) -> bool:
    """Return True if val is None or float NaN."""
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False

# ---------------------------------------------------------------------------
# Baseline agent — rule-based, handles both None and float NaN missing values
# ---------------------------------------------------------------------------
EXPECTED_NUMERIC   = {"age", "salary", "experience", "rating", "bonus",
                      "years_at_company", "performance_score", "overtime_hours"}
OUTLIER_THRESHOLDS = {"salary": 300_000, "bonus": 80_000}
SCORE_COLS         = {"rating", "performance_score"}


def _baseline_agent(observation) -> int:
    """Rule-based agent. Returns 0=skip, 1=impute_missing, 3=fix_outlier."""
    if observation is None:
        return 0

    # Handle both dict (from local env) and object (from HTTP client)
    if isinstance(observation, dict):
        row_data = observation["row_data"]
    else:
        row_data = observation.row_data

    # 1 — type mismatch: non-parseable string in a numeric column
    for key, val in row_data.items():
        if key in EXPECTED_NUMERIC and isinstance(val, str):
            try:
                float(val)
            except (ValueError, TypeError):
                return 3

    # 2 — whitespace padding
    for val in row_data.values():
        if isinstance(val, str) and val != val.strip():
            return 3

    # 3 — column-specific outlier thresholds
    for key, val in row_data.items():
        if key in OUTLIER_THRESHOLDS and isinstance(val, (int, float)) and not _is_missing(val):
            if val > OUTLIER_THRESHOLDS[key]:
                return 3

    # 4 — invalid score range (> 5)
    for key, val in row_data.items():
        if key in SCORE_COLS and isinstance(val, (int, float)) and not _is_missing(val):
            if val > 5:
                return 3

    # 5 — invalid negative in numeric column
    for key, val in row_data.items():
        if key in EXPECTED_NUMERIC and isinstance(val, (int, float)) and not _is_missing(val):
            if val < 0:
                return 3

    # 6 — missing value (None OR float NaN from pandas)
    for val in row_data.values():
        if _is_missing(val):
            return 1

    # 7 — no issue detected
    return 0


# ---------------------------------------------------------------------------
# LLM agent — calls HF inference API, falls back to baseline on any error
# ---------------------------------------------------------------------------
def _llm_agent(observation) -> int:
    if not HF_TOKEN:
        return _baseline_agent(observation)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        if isinstance(observation, dict):
            row_data = observation["row_data"]
        else:
            row_data = observation.row_data

        row_lines = []
        for col, val in row_data.items():
            if _is_missing(val):
                row_lines.append(f"  {col}: NULL")
            elif isinstance(val, str):
                row_lines.append(f"  {col}: \"{val}\"")
            else:
                row_lines.append(f"  {col}: {val}")

        system_prompt = (
            "You are an expert data quality analyst reviewing rows from an employee dataset.\n\n"
            "DATASET SCHEMA AND VALID RANGES:\n"
            "  age: integer 22-62 | salary: integer 35000-180000 | city: string\n"
            "  experience: integer 0-30 | rating: float 0-5 | department: string\n"
            "  bonus: integer 0-25000 | years_at_company: integer 0-20\n"
            "  performance_score: float 0-5 | overtime_hours: integer 0-80\n\n"
            "ACTIONS: 0=skip, 1=impute_missing (NULL cell), 3=fix_outlier (any other issue)\n"
            "Respond ONLY with JSON: {\"action\": <0|1|3>, \"reason\": \"<one sentence>\"}"
        )
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": "Examine:\n" + "\n".join(row_lines) +
                 "\nRespond ONLY with JSON: {\"action\": <0|1|3>, \"reason\": \"...\"}"},
            ],
            temperature=0,
            max_tokens=80,
            stream=False,
        )
        raw = response.choices[0].message.content.strip()
        try:
            action = int(json.loads(raw)["action"])
        except Exception:
            m = re.search(r'"action"\s*:\s*(\d)', raw)
            action = int(m.group(1)) if m else 0
        return action if action in {0, 1, 3} else 0
    except Exception:
        return _baseline_agent(observation)


# ---------------------------------------------------------------------------
# Action label map
# ---------------------------------------------------------------------------
VALID_ACTIONS = {0: "skip", 1: "impute_missing", 3: "fix_outlier"}


# ---------------------------------------------------------------------------
# Episode runner — uses local DataCleaningEnv directly (no HTTP)
# ---------------------------------------------------------------------------
def run_episode(task_level: str, use_llm: bool, agent_name: str) -> dict:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_level, env=BENCHMARK, model=agent_name)

    try:
        env  = DataCleaningEnv()
        obs  = env.reset(task_level=task_level)
        done = (obs is None)

        while not done:
            steps_taken += 1

            action_id    = _llm_agent(obs) if use_llm else _baseline_agent(obs)
            action_label = VALID_ACTIONS.get(action_id, str(action_id))

            obs, reward, done, _info = env.step(action_id)
            reward_f = float(reward)
            rewards.append(reward_f)

            log_step(step=steps_taken, action=action_label, reward=reward_f, done=done, error=None)

        raw_state = env.state()
        score     = round(float(raw_state.get("score", 0.0)), 2)
        success   = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        # Send errors to stderr only — stdout must stay clean for the checker
        print(f"[DEBUG] Episode error in {task_level}: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_level, "score": score, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main — guarded so module-level import doesn't auto-run episodes
# ---------------------------------------------------------------------------
def main():
    agent_choice = os.environ.get("AGENT_TYPE", "baseline")
    levels       = ("easy", "medium", "hard")

    run_baseline = (agent_choice in ("baseline", "both", "auto")) or (not HF_TOKEN)
    run_llm      = (agent_choice in ("llm", "both")) and bool(HF_TOKEN)

    if run_baseline:
        for lvl in levels:
            run_episode(lvl, use_llm=False, agent_name="baseline")

    if run_llm:
        for lvl in levels:
            run_episode(lvl, use_llm=True, agent_name=MODEL_NAME)


if __name__ == "__main__":
    main()
