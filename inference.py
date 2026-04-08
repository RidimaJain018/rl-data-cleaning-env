"""
inference.py — Run DataCleaningEnv episodes with different agents
=================================================================
STDOUT FORMAT (required by hackathon checker — do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=0.00 done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import re
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure env.py is always importable regardless of the checker's cwd
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Force stdout unbuffered — works whether checker pipes or captures output
# ---------------------------------------------------------------------------
sys.stdout.reconfigure(line_buffering=True)

# ---------------------------------------------------------------------------
# Required environment variables (hackathon spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

BENCHMARK               = "data-cleaning-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# ---------------------------------------------------------------------------
# Official structured stdout loggers — exact format the checker parses
# ---------------------------------------------------------------------------
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val  = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.2f} rewards={rewards_str}",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Baseline agent (inline — no import needed, guaranteed to work)
# ---------------------------------------------------------------------------
def _baseline_agent(observation: dict | None) -> int:
    if observation is None:
        return 0
    row = observation["row_data"]

    EXPECTED_NUMERIC = {
        "age", "salary", "experience", "rating",
        "bonus", "years_at_company", "performance_score", "overtime_hours",
    }
    OUTLIER_THRESHOLDS = {"salary": 300_000, "bonus": 80_000}
    SCORE_COLS = {"rating", "performance_score"}

    for key, val in row.items():
        if key in EXPECTED_NUMERIC and isinstance(val, str):
            try:
                float(val)
            except (ValueError, TypeError):
                return 3

    for val in row.values():
        if isinstance(val, str) and val != val.strip():
            return 3

    for key, val in row.items():
        if key in OUTLIER_THRESHOLDS and isinstance(val, (int, float)):
            if val > OUTLIER_THRESHOLDS[key]:
                return 3

    for key, val in row.items():
        if key in SCORE_COLS and isinstance(val, (int, float)):
            if val > 5:
                return 3

    for key, val in row.items():
        if key in EXPECTED_NUMERIC and isinstance(val, (int, float)):
            if val < 0:
                return 3

    for val in row.values():
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return 1

    return 0


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------
def _llm_agent(observation: dict | None) -> int:
    if observation is None:
        return 0
    if not HF_TOKEN:
        return _baseline_agent(observation)

    try:
        from openai import OpenAI
        client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
        row = observation["row_data"]

        row_lines = []
        for col, val in row.items():
            if val is None or (isinstance(val, float) and str(val) == "nan"):
                row_lines.append(f"  {col}: NULL")
            elif isinstance(val, str):
                row_lines.append(f"  {col}: \"{val}\"")
            else:
                row_lines.append(f"  {col}: {val}")

        system_prompt = (
            "You are an expert data quality analyst reviewing rows from an employee dataset.\n\n"
            "DATASET SCHEMA AND VALID RANGES:\n"
            "  age              : integer, valid range 22-62\n"
            "  salary           : integer USD, valid range 35000-180000\n"
            "  city             : string, one of [NY, LA, SF, Chicago, Austin, Seattle, Boston, Denver]\n"
            "  experience       : integer years, valid range 0-30\n"
            "  rating           : float, valid range 0.0-5.0\n"
            "  department       : string, one of [Engineering, Sales, HR, Marketing, Finance, Operations]\n"
            "  bonus            : integer USD, valid range 0-25000\n"
            "  years_at_company : integer years, valid range 0-20\n"
            "  performance_score: float, valid range 0.0-5.0\n"
            "  overtime_hours   : integer, valid range 0-80\n\n"
            "ACTIONS:\n"
            "  0 = skip           - row has no data quality issue\n"
            "  1 = impute_missing - row has a NULL / missing cell\n"
            "  3 = fix_outlier    - row has any other issue\n\n"
            "Respond ONLY with JSON: {\"action\": <0|1|3>, \"reason\": \"<one sentence>\"}"
        )
        user_prompt = (
            "Examine this employee record:\n\n"
            + "\n".join(row_lines)
            + "\n\nRespond ONLY with JSON: {\"action\": <0|1|3>, \"reason\": \"...\"}"
        )

        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            max_tokens=80,
            stream=False,
        )
        raw = response.choices[0].message.content.strip()
        try:
            parsed = json.loads(raw)
            action = int(parsed["action"])
        except (json.JSONDecodeError, KeyError, ValueError):
            m = re.search(r'"action"\s*:\s*(\d)', raw)
            action = int(m.group(1)) if m else 0

        return action if action in {0, 1, 3} else 0

    except Exception:
        return _baseline_agent(observation)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(task_level: str, agent_fn, agent_name: str) -> dict:
    VALID_ACTIONS = {0: "skip", 1: "impute_missing", 3: "fix_outlier"}

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False
    env                      = None

    log_start(task=task_level, env=BENCHMARK, model=agent_name)

    try:
        from env import DataCleaningEnv
        env      = DataCleaningEnv()
        obs      = env.reset(task_level=task_level)
        done     = False
        step_num = 0

        while not done:
            step_num    += 1
            action_id    = agent_fn(obs)
            action_label = VALID_ACTIONS.get(action_id, str(action_id))
            obs, reward, done, info = env.step(action_id)
            reward_f = float(reward)
            rewards.append(reward_f)
            steps_taken = step_num
            error = info.get("error") if isinstance(info, dict) else None
            log_step(step=step_num, action=action_label, reward=reward_f, done=done, error=error)

        score   = round(env.grade(), 2)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        if env is not None and hasattr(env, "grade"):
            try:
                score = round(env.grade(), 2)
            except Exception:
                score = 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task":         task_level,
        "score":        score,
        "total_reward": sum(rewards),
        "steps":        steps_taken,
    }


# ---------------------------------------------------------------------------
# Main — runs unconditionally at module load time.
# This guarantees output whether the checker does:
#   python inference.py          (direct execution)
#   python -m inference          (module execution)
#   import inference             (import — __main__ guard would silently skip)
# ---------------------------------------------------------------------------
def main():
    agent_choice = os.environ.get("AGENT_TYPE", "baseline")
    levels       = ("easy", "medium", "hard")

    run_baseline = (agent_choice in ("baseline", "both", "auto")) or (not HF_TOKEN)
    run_llm      = (agent_choice in ("llm", "both")) and bool(HF_TOKEN)

    if run_baseline:
        for lvl in levels:
            run_episode(lvl, _baseline_agent, agent_name="baseline")

    if run_llm:
        for lvl in levels:
            run_episode(lvl, _llm_agent, agent_name=MODEL_NAME)


# Unconditional call — no __main__ guard intentionally.
# Works for direct execution AND import-based invocation by the checker.
main()
