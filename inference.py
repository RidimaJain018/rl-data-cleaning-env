"""
inference.py — Run DataCleaningEnv episodes with different agents
=================================================================
Default behaviour (no flags):
  - Always runs the baseline agent across all three task levels.
  - If API_BASE_URL + HF_TOKEN are set, also runs the LLM agent.

Required environment variables (hackathon spec):
    API_BASE_URL   Base URL for the LLM API endpoint
    MODEL_NAME     The model identifier to use for inference
    HF_TOKEN       Your Hugging Face / API key

Optional:
    AGENT_TYPE     "baseline" or "llm" (overrides --agent flag)

Usage
-----
    python inference.py                          # baseline on all tasks
    python inference.py --agent llm --task hard  # LLM on hard only
    python inference.py --agent both             # both agents, all tasks

STDOUT FORMAT (required by hackathon checker — do not change):
    [START] task=<task_name> env=<benchmark> model=<model_name>
    [STEP]  step=<n> action=<action_str> reward=0.00 done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<score> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import argparse
import json
import os
import re
from typing import List, Optional

from openai import OpenAI

from agent import baseline_agent
from env import DataCleaningEnv
from models import VALID_ACTIONS

# ---------------------------------------------------------------------------
# Required environment variables (hackathon spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

# Benchmark name emitted in [START] line
BENCHMARK = "data-cleaning-env"

# Score threshold to call an episode "successful"
SUCCESS_SCORE_THRESHOLD = 0.5

# ── Colours (disabled when not a TTY so checker output stays clean) ────────
_IS_TTY = os.isatty(1)
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
RED    = lambda t: _c("31", t)


# ---------------------------------------------------------------------------
# Official structured stdout loggers
# These three functions produce the EXACT format the checker parses.
# Do NOT change field names, ordering, or formatting.
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
# LLM Agent — uses OpenAI client as required by hackathon spec
# ---------------------------------------------------------------------------
def llm_agent(observation: dict | None) -> int:
    """
    LLM-powered agent using the OpenAI client (mandatory per hackathon spec).
    Reasons from raw row values + schema rules only.
    Falls back to baseline_agent on any failure.
    """
    if observation is None:
        return 0

    if not HF_TOKEN:
        return baseline_agent(observation)

    client = OpenAI(api_key=HF_TOKEN, base_url=API_BASE_URL)
    row    = observation["row_data"]

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
        "ISSUE TYPES TO DETECT:\n"
        "  - NULL value: any cell showing NULL or missing\n"
        "  - Outlier: numeric value far outside the valid range shown above\n"
        "  - Invalid rating: rating or performance_score above 5.0\n"
        "  - Invalid negative: negative value in a column that must be >= 0\n"
        "  - Type mismatch: non-numeric string (e.g. 'N/A', 'ten') in a numeric column\n"
        "  - Whitespace padding: string with leading or trailing spaces\n\n"
        "ACTIONS:\n"
        "  0 = skip           - row has no data quality issue\n"
        "  1 = impute_missing - row has a NULL / missing cell\n"
        "  3 = fix_outlier    - row has any other issue (outlier, invalid value,\n"
        "                       type mismatch, whitespace padding)\n\n"
        "Respond ONLY with JSON: {\"action\": <0|1|3>, \"reason\": \"<one sentence>\"}"
    )

    user_prompt = (
        "Examine this employee record and identify any data quality issue:\n\n"
        + "\n".join(row_lines)
        + "\n\nRespond ONLY with JSON: {\"action\": <0|1|3>, \"reason\": \"...\"}"
    )

    try:
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
        return action if action in VALID_ACTIONS else 0
    except Exception:
        return baseline_agent(observation)


# ---------------------------------------------------------------------------
# Episode runner — emits official structured stdout per the checker spec
# ---------------------------------------------------------------------------
def run_episode_structured(task_level: str, agent_fn, agent_name: str) -> dict:
    """
    Run one complete episode. Emits [START]/[STEP]/[END] to stdout.
    Always emits [END] even if an exception occurs.
    Returns a summary dict for human-readable printing.
    """
    env  = DataCleaningEnv()
    obs  = env.reset(task_level=task_level)

    rewards:     List[float] = []
    steps_taken: int         = 0
    score:       float       = 0.0
    success:     bool        = False

    log_start(task=task_level, env=BENCHMARK, model=agent_name)

    try:
        done     = False
        step_num = 0

        while not done:
            step_num += 1
            action_id    = agent_fn(obs)
            action_label = VALID_ACTIONS.get(action_id, str(action_id))

            obs, reward, done, info = env.step(action_id)

            reward_f = float(reward)
            rewards.append(reward_f)
            steps_taken = step_num

            error = info.get("error") if isinstance(info, dict) else None

            log_step(
                step   = step_num,
                action = action_label,
                reward = reward_f,
                done   = done,
                error  = error,
            )

        score   = round(env.grade(), 2)
        success = score >= SUCCESS_SCORE_THRESHOLD

    except Exception as exc:
        print(f"[DEBUG] Episode error: {exc}", flush=True)
        score = round(env.grade(), 2) if hasattr(env, "grade") else 0.0

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {
        "task":         task_level,
        "score":        score,
        "total_reward": sum(rewards),
        "steps":        steps_taken,
        "fixed":        env.total_issues_at_start - len(env.issues),
        "total_issues": env.total_issues_at_start,
        "episode_log":  list(env.episode_log),
    }


# ---------------------------------------------------------------------------
# Human-readable printers (printed AFTER all structured blocks)
# ---------------------------------------------------------------------------
SEP = "─" * 52

def print_results(results: list[dict], agent_name: str) -> None:
    print(f"\n{BOLD('=== RESULTS — ' + agent_name.upper() + ' AGENT ===')}\n")
    print(f"{'Task':<8} {'Score':>6} {'Reward':>8} {'Steps':>6} {'Fixed':>7}")
    print(SEP)
    for r in results:
        score_str = GREEN(f"{r['score']:>6.2f}") if r['score'] == 1.0 else YELLOW(f"{r['score']:>6.2f}")
        print(
            f"{r['task']:<8} {score_str} {r['total_reward']:>8.1f} "
            f"{r['steps']:>6} {r['fixed']}/{r['total_issues']}"
        )
    avg = sum(r["score"] for r in results) / len(results)
    avg_str = GREEN(f"{avg:>6.2f}") if avg == 1.0 else YELLOW(f"{avg:>6.2f}")
    print(SEP)
    print(f"{'Average':>8} {avg_str}")


def print_episode_summary(results: list[dict], agent_name: str) -> None:
    print(f"\n{BOLD('Issue-type accuracy — ' + agent_name.upper())}")
    from collections import defaultdict
    issue_stats: dict[str, dict] = defaultdict(lambda: {"correct": 0, "total": 0})
    for r in results:
        for e in r.get("episode_log", []):
            issue_stats[e["issue"]]["total"] += 1
            if e["correct"]:
                issue_stats[e["issue"]]["correct"] += 1
    print(f"  {'Issue type':<24} {'Correct':>7} {'Total':>6} {'Accuracy':>9}")
    print(f"  {'─'*24} {'─'*7} {'─'*6} {'─'*9}")
    for issue, s in sorted(issue_stats.items()):
        acc     = s["correct"] / s["total"] if s["total"] > 0 else 0
        acc_str = GREEN(f"{acc:>9.0%}") if acc == 1.0 else YELLOW(f"{acc:>9.0%}")
        print(f"  {issue:<24} {s['correct']:>7} {s['total']:>6} {acc_str}")


# ---------------------------------------------------------------------------
# LLM availability check
# ---------------------------------------------------------------------------
def _llm_vars_set() -> bool:
    return bool(HF_TOKEN and API_BASE_URL)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DataCleaningEnv inference.")
    parser.add_argument(
        "--agent", choices=["baseline", "llm", "both"], default="auto",
        help=(
            "Agent to use. 'auto': always runs baseline; "
            "also runs LLM if API_BASE_URL + HF_TOKEN are set."
        ),
    )
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard", "all"], default="all",
        help="Task difficulty to run (default: all)",
    )
    args = parser.parse_args()

    agent_choice = os.environ.get("AGENT_TYPE", args.agent)
    levels       = ("easy", "medium", "hard") if args.task == "all" else (args.task,)

    run_baseline = agent_choice in ("baseline", "both", "auto")
    run_llm      = (
        agent_choice == "llm"
        or agent_choice == "both"
        or (agent_choice == "auto" and _llm_vars_set())
    )

    # ── Baseline ─────────────────────────────────────────────────────────
    baseline_results: list[dict] = []
    if run_baseline:
        for lvl in levels:
            r = run_episode_structured(lvl, baseline_agent, agent_name="baseline")
            baseline_results.append(r)

    # ── LLM agent ────────────────────────────────────────────────────────
    llm_results: list[dict] = []
    if run_llm:
        for lvl in levels:
            r = run_episode_structured(lvl, llm_agent, agent_name=MODEL_NAME)
            llm_results.append(r)

    # ── Human-readable summary (after all structured blocks) ──────────────
    if baseline_results:
        print_results(baseline_results, "baseline")
        print_episode_summary(baseline_results, "baseline")

    if llm_results:
        print_results(llm_results, "llm")
        print_episode_summary(llm_results, "llm")

    if not run_llm and agent_choice == "auto":
        print(
            f"\n{DIM('Tip: set HF_TOKEN + API_BASE_URL + MODEL_NAME to also run the LLM agent.')}"
        )
