"""
inference.py — Run DataCleaningEnv episodes via the live HF Space API
======================================================================
Uses only Python standard library (urllib + json) — no pandas, numpy,
or local env.py import needed. Connects to the deployed HF Space.

STDOUT FORMAT (required by hackathon checker):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=0.00 done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
import urllib.request
import urllib.error
from typing import List, Optional

# ---------------------------------------------------------------------------
# Force stdout unbuffered — safe even if checker replaces sys.stdout
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

# Your live HF Space URL — overridable via env var
SPACE_URL = os.getenv(
    "SPACE_URL",
    "https://Ridi2007-rl-data-cleaning-env.hf.space",
).rstrip("/")

BENCHMARK               = "data-cleaning-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Minimal HTTP helpers (stdlib only — no requests/httpx needed)
# ---------------------------------------------------------------------------
def _post(path: str, body: dict, session_id: str = "checker") -> dict:
    url     = f"{SPACE_URL}{path}"
    data    = json.dumps(body).encode()
    headers = {
        "Content-Type":  "application/json",
        "X-Session-Id":  session_id,
    }
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())


def _get(path: str, session_id: str = "checker") -> dict:
    url     = f"{SPACE_URL}{path}"
    headers = {"X-Session-Id": session_id}
    req     = urllib.request.Request(url, headers=headers, method="GET")
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode())

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
# Baseline agent — pure Python, no deps
# ---------------------------------------------------------------------------
EXPECTED_NUMERIC    = {"age", "salary", "experience", "rating", "bonus",
                       "years_at_company", "performance_score", "overtime_hours"}
OUTLIER_THRESHOLDS  = {"salary": 300_000, "bonus": 80_000}
SCORE_COLS          = {"rating", "performance_score"}


def _baseline_agent(row_data: dict) -> int:
    """Rule-based agent — inspects row_data dict, returns action int."""
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

    # outlier
    for key, val in row_data.items():
        if key in OUTLIER_THRESHOLDS and isinstance(val, (int, float)):
            if val > OUTLIER_THRESHOLDS[key]:
                return 3

    # invalid score range
    for key, val in row_data.items():
        if key in SCORE_COLS and isinstance(val, (int, float)):
            if val > 5:
                return 3

    # invalid negative
    for key, val in row_data.items():
        if key in EXPECTED_NUMERIC and isinstance(val, (int, float)):
            if val < 0:
                return 3

    # missing value
    for val in row_data.values():
        if val is None:
            return 1

    return 0

# ---------------------------------------------------------------------------
# LLM agent — calls HF inference API via openai client
# ---------------------------------------------------------------------------
def _llm_agent(row_data: dict) -> int:
    if not HF_TOKEN:
        return _baseline_agent(row_data)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)

        row_lines = []
        for col, val in row_data.items():
            if val is None:
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
        return _baseline_agent(row_data)

# ---------------------------------------------------------------------------
# Episode runner — calls HF Space API endpoints
# ---------------------------------------------------------------------------
VALID_ACTIONS = {0: "skip", 1: "impute_missing", 3: "fix_outlier"}


def run_episode(task_level: str, use_llm: bool, agent_name: str) -> dict:
    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    session_id               = f"checker-{task_level}-{agent_name}"

    log_start(task=task_level, env=BENCHMARK, model=agent_name)

    try:
        # Reset episode
        reset_resp = _post("/reset", {"task_level": task_level}, session_id)
        obs        = reset_resp.get("observation")
        done       = (obs is None)
        step_num   = 0

        while not done:
            step_num += 1
            row_data  = obs["row_data"] if obs else {}

            # Choose action
            if use_llm:
                action_id = _llm_agent(row_data)
            else:
                action_id = _baseline_agent(row_data)

            action_label = VALID_ACTIONS.get(action_id, str(action_id))

            # Step
            step_resp = _post("/step", {"action": action_id}, session_id)
            reward_f  = float(step_resp["reward"]["value"])
            done      = bool(step_resp["reward"]["done"])
            obs       = step_resp.get("observation")
            rewards.append(reward_f)
            steps_taken = step_num

            log_step(step=step_num, action=action_label, reward=reward_f, done=done, error=None)

        # Get final score from state
        state_resp = _get("/state", session_id)
        score      = round(float(state_resp.get("score", 0.0)), 2)
        success    = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_level, "score": score, "steps": steps_taken}

# ---------------------------------------------------------------------------
# Main — unconditional call (works for direct run AND import by checker)
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


main()
