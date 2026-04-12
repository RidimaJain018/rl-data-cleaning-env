"""
inference.py — Run DataCleaningEnv episodes with trained Q-policy + LLM agent
===============================================================================
Agent priority (automatic, no flags needed):
    1. Trained Q-policy  (policy.pkl)  — loaded if file exists
    2. LLM agent         (API call)    — used if API_BASE_URL + API_KEY set
    3. Baseline rule-based agent       — deterministic fallback, never crashes

STDOUT FORMAT (required by hackathon checker):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=0.00 done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import math
import os
import pickle
import re
import sys
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Force stdout unbuffered — required so checker sees output immediately
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Environment variables (checker injects API_KEY and API_BASE_URL)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME", "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("API_KEY") or os.getenv("HF_TOKEN", "")
POLICY_PATH  = os.getenv("POLICY_PATH", "policy.pkl")

BENCHMARK               = "data-cleaning-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Import local environment directly (all files are in /app in Docker)
# ---------------------------------------------------------------------------
from env import DataCleaningEnv, EXPECTED_NUMERIC

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
# NaN-safe helper
# ---------------------------------------------------------------------------
def _is_missing(val) -> bool:
    if val is None:
        return True
    if isinstance(val, float) and math.isnan(val):
        return True
    return False

# ---------------------------------------------------------------------------
# Feature extraction for Q-policy (must match train.py exactly)
# ---------------------------------------------------------------------------
def _parseable_as_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def _obs_to_state_key(obs) -> Tuple:
    """Convert raw observation to the same state key used during training."""
    if obs is None:
        return ("__terminal__",)

    if isinstance(obs, dict):
        row = obs["row_data"]
    else:
        row = obs.row_data

    numeric_vals = {
        k: float(v) for k, v in row.items()
        if k in EXPECTED_NUMERIC
        and isinstance(v, (int, float))
        and not (isinstance(v, float) and math.isnan(float(v)))
    }

    if numeric_vals:
        vals = list(numeric_vals.values())
        mean = sum(vals) / len(vals)
        std  = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5 or 1.0
    else:
        mean, std = 0.0, 1.0

    features = []
    for col in sorted(row.keys()):
        val = row[col]

        is_null = val is None or (isinstance(val, float) and math.isnan(float(val)))

        is_str_in_numeric = (
            col in EXPECTED_NUMERIC
            and isinstance(val, str)
            and not _parseable_as_float(val)
        )

        has_whitespace = isinstance(val, str) and val != val.strip()

        if col in EXPECTED_NUMERIC and isinstance(val, (int, float)) and not is_null:
            z = (float(val) - mean) / std
            if z < -2:
                bucket = -2
            elif z < -0.5:
                bucket = -1
            elif z <= 0.5:
                bucket = 0
            elif z <= 2:
                bucket = 1
            else:
                bucket = 2
        else:
            bucket = 0

        features.append((col, int(is_null), int(is_str_in_numeric),
                         int(has_whitespace), bucket))

    return tuple(features)


# ---------------------------------------------------------------------------
# Agent 1 — Trained Q-policy
# ---------------------------------------------------------------------------
_loaded_policy: Optional[Dict[Tuple, int]] = None
_policy_load_attempted = False


def _load_policy() -> Optional[Dict[Tuple, int]]:
    global _loaded_policy, _policy_load_attempted
    if _policy_load_attempted:
        return _loaded_policy
    _policy_load_attempted = True
    if not os.path.exists(POLICY_PATH):
        print(f"[DEBUG] policy.pkl not found at {POLICY_PATH} — skipping Q-agent", file=sys.stderr, flush=True)
        return None
    try:
        with open(POLICY_PATH, "rb") as f:
            artifact = pickle.load(f)
        _loaded_policy = artifact["policy"]
        eps = artifact.get("episodes_trained", "?")
        print(f"[DEBUG] Loaded Q-policy ({len(_loaded_policy)} states, trained {eps} eps)", file=sys.stderr, flush=True)
        return _loaded_policy
    except Exception as exc:
        print(f"[DEBUG] Failed to load policy.pkl: {exc}", file=sys.stderr, flush=True)
        return None


def _q_policy_agent(obs) -> Optional[int]:
    """Return action from trained Q-policy, or None if policy unavailable."""
    policy = _load_policy()
    if policy is None:
        return None
    state  = _obs_to_state_key(obs)
    action = policy.get(state)
    if action is None:
        # Unseen state — fall through to LLM/baseline
        print(f"[DEBUG] Q-policy: unseen state, falling through", file=sys.stderr, flush=True)
        return None
    return action


# ---------------------------------------------------------------------------
# Agent 2 — LLM agent (OpenAI-compatible, proxied through checker)
# ---------------------------------------------------------------------------
EXPECTED_NUMERIC_RANGES = {
    "age": (22, 62),
    "salary": (35000, 180000),
    "experience": (0, 30),
    "rating": (0.0, 5.0),
    "bonus": (0, 25000),
    "years_at_company": (0, 20),
    "performance_score": (0.0, 5.0),
    "overtime_hours": (0, 80),
}


def _llm_agent(obs) -> int:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        if isinstance(obs, dict):
            row_data = obs["row_data"]
        else:
            row_data = obs.row_data

        # Build row display — raw values only, no hints about issue type
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

    except Exception as exc:
        print(f"[DEBUG] LLM call failed: {exc}", file=sys.stderr, flush=True)
        return _baseline_agent(obs)


# ---------------------------------------------------------------------------
# Agent 3 — Rule-based baseline (deterministic fallback, never crashes)
# ---------------------------------------------------------------------------
OUTLIER_THRESHOLDS = {"salary": 300_000, "bonus": 80_000}
SCORE_COLS         = {"rating", "performance_score"}


def _baseline_agent(obs) -> int:
    if obs is None:
        return 0

    if isinstance(obs, dict):
        row_data = obs["row_data"]
    else:
        row_data = obs.row_data

    # type mismatch: non-parseable string in numeric column
    for key, val in row_data.items():
        if key in EXPECTED_NUMERIC and isinstance(val, str):
            try:
                float(val)
            except (ValueError, TypeError):
                return 3

    # whitespace padding
    for val in row_data.values():
        if isinstance(val, str) and val != val.strip():
            return 3

    # column-specific outlier thresholds
    for key, val in row_data.items():
        if key in OUTLIER_THRESHOLDS and isinstance(val, (int, float)) and not _is_missing(val):
            if val > OUTLIER_THRESHOLDS[key]:
                return 3

    # invalid score range (> 5)
    for key, val in row_data.items():
        if key in SCORE_COLS and isinstance(val, (int, float)) and not _is_missing(val):
            if val > 5:
                return 3

    # invalid negative
    for key, val in row_data.items():
        if key in EXPECTED_NUMERIC and isinstance(val, (int, float)) and not _is_missing(val):
            if val < 0:
                return 3

    # missing value
    for val in row_data.values():
        if _is_missing(val):
            return 1

    return 0


# ---------------------------------------------------------------------------
# Master agent — Q-policy → LLM → baseline (in that order)
# ---------------------------------------------------------------------------
def _select_action(obs) -> int:
    # 1. Try trained Q-policy first
    q_action = _q_policy_agent(obs)
    if q_action is not None:
        return q_action

    # 2. Try LLM (checker requires API calls through proxy when available)
    if HF_TOKEN:
        return _llm_agent(obs)

    # 3. Deterministic baseline fallback
    return _baseline_agent(obs)


# ---------------------------------------------------------------------------
# Action label map
# ---------------------------------------------------------------------------
VALID_ACTIONS = {0: "skip", 1: "impute_missing", 2: "flag_for_review", 3: "fix_outlier"}


# ---------------------------------------------------------------------------
# Episode runner — uses local DataCleaningEnv directly (no HTTP)
# ---------------------------------------------------------------------------
def run_episode(task_level: str, agent_name: str) -> dict:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_level, env=BENCHMARK, model=agent_name)

    try:
        env  = DataCleaningEnv()
        # Reset twice so the rotating seed picks a different domain
        # than the default episode 1 — gives the checker varied observations
        obs  = env.reset(task_level=task_level)
        done = (obs is None)

        while not done:
            steps_taken += 1

            action_id    = _select_action(obs)
            action_label = VALID_ACTIONS.get(action_id, str(action_id))

            obs, reward, done, _info = env.step(action_id)
            reward_f = float(reward)
            rewards.append(reward_f)

            log_step(step=steps_taken, action=action_label, reward=reward_f, done=done, error=None)

        raw_state = env.state()
        score     = round(float(raw_state.get("score", 0.01)), 4)
        # NOTE: score clamping kept as-is — checker rejects exactly 0.0 or 1.0
        score     = min(max(score, 0.01), 0.99)
        success   = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error in {task_level}: {exc}", file=sys.stderr, flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_level, "score": score, "steps": steps_taken}


# ---------------------------------------------------------------------------
# Main — runs all 3 levels
# ---------------------------------------------------------------------------
def main():
    # Determine which agent is active and log it
    policy = _load_policy()
    if policy is not None:
        active = f"Q-policy({len(policy)}_states)"
    elif HF_TOKEN:
        active = f"LLM({MODEL_NAME})"
    else:
        active = "baseline"

    print(f"[DEBUG] Active agent: {active}", file=sys.stderr, flush=True)

    levels = ("easy", "medium", "hard")
    for lvl in levels:
        run_episode(lvl, agent_name=active)


if __name__ == "__main__":
    main()
