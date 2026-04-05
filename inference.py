"""
inference.py — Run DataCleaningEnv episodes with different agents
=================================================================
Default behaviour (no flags):
  - Always runs the baseline agent across all three task levels.
  - If API_BASE_URL + HF_TOKEN are set, also runs the LLM agent and prints
    a side-by-side comparison table automatically.

Required environment variables (hackathon spec):
    API_BASE_URL   Base URL for the LLM API endpoint
    MODEL_NAME     Model identifier to use
    HF_TOKEN       API key / Hugging Face access token

Optional:
    AGENT_TYPE     "baseline" or "llm" (overrides --agent flag)

Usage
-----
    # Baseline only (no key needed)
    python inference.py

    # LLM agent — auto-detects keys and shows comparison
    export API_BASE_URL="https://api-inference.huggingface.co/v1"
    export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
    export HF_TOKEN="hf_..."
    python inference.py

    # Force a specific agent or task
    python inference.py --agent llm --task hard
    python inference.py --agent baseline
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time

from agent import baseline_agent
from env import DataCleaningEnv
from models import VALID_ACTIONS

# ── Colours (disabled when not a TTY) ────────────────────────────────────────
_IS_TTY = os.isatty(1)
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text

GREEN  = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN   = lambda t: _c("36", t)
BOLD   = lambda t: _c("1",  t)
DIM    = lambda t: _c("2",  t)
RED    = lambda t: _c("31", t)


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------
def llm_agent(observation: dict | None) -> int:
    """
    LLM-powered agent. The LLM reasons from raw row values and domain
    knowledge about valid ranges — no pre-computed issue hints are passed.
    This ensures the model is doing genuine data quality reasoning.

    Reads the three required hackathon env variables:
        HF_TOKEN      — API key
        API_BASE_URL  — base URL of the OpenAI-compatible endpoint
        MODEL_NAME    — model identifier

    Falls back to baseline_agent on any failure — never crashes.
    """
    if observation is None:
        return 0

    try:
        from openai import OpenAI
    except ImportError:
        print("[llm_agent] openai package not installed. Falling back to baseline.")
        return baseline_agent(observation)

    api_key  = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")
    base_url = os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL", None)
    model    = (
        os.environ.get("MODEL_NAME")
        or os.environ.get("OPENAI_MODEL")
        or "meta-llama/Llama-3.2-3B-Instruct"
    )

    if not api_key:
        print("[llm_agent] No API key found. Falling back to baseline.")
        return baseline_agent(observation)

    client = OpenAI(api_key=api_key, base_url=base_url)
    row    = observation["row_data"]

    # Format row values clearly — show NULL explicitly, preserve raw values
    # so the LLM reasons from the actual data, not pre-labelled hints.
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
        "  age              : integer, valid range 22–62\n"
        "  salary           : integer USD, valid range 35000–180000\n"
        "  city             : string, one of [NY, LA, SF, Chicago, Austin, Seattle, Boston, Denver]\n"
        "  experience       : integer years, valid range 0–30\n"
        "  rating           : float, valid range 0.0–5.0\n"
        "  department       : string, one of [Engineering, Sales, HR, Marketing, Finance, Operations]\n"
        "  bonus            : integer USD, valid range 0–25000\n"
        "  years_at_company : integer years, valid range 0–20\n"
        "  performance_score: float, valid range 0.0–5.0\n"
        "  overtime_hours   : integer, valid range 0–80\n\n"
        "ISSUE TYPES TO DETECT:\n"
        "  - NULL value: any cell showing NULL or missing\n"
        "  - Outlier: numeric value far outside the valid range shown above\n"
        "  - Invalid rating: rating or performance_score above 5.0\n"
        "  - Invalid negative: negative value in a column that must be ≥ 0\n"
        "  - Type mismatch: non-numeric string (e.g. 'N/A', 'ten') in a numeric column\n"
        "  - Whitespace padding: string with leading or trailing spaces (e.g. '\" NY \"')\n\n"
        "ACTIONS:\n"
        "  0 = skip           — row has no data quality issue\n"
        "  1 = impute_missing — row has a NULL / missing cell\n"
        "  3 = fix_outlier    — row has any other issue (outlier, invalid value,\n"
        "                       type mismatch, whitespace padding)\n\n"
        "Carefully examine each field against the valid ranges. "
        "Respond ONLY with a JSON object containing your chosen action and a brief reason:\n"
        '{"action": <0|1|3>, "reason": "<one concise sentence explaining what you found>"}'
    )

    user_prompt = (
        "Examine this employee record and identify any data quality issue:\n\n"
        + "\n".join(row_lines)
        + "\n\nCheck each field against the valid ranges in your instructions. "
        'Respond ONLY with JSON: {"action": <0|1|3>, "reason": "..."}'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            max_tokens=80,
        )
        raw = response.choices[0].message.content.strip()

        try:
            parsed = json.loads(raw)
            action = int(parsed["action"])
            reason = parsed.get("reason", "")
        except (json.JSONDecodeError, KeyError, ValueError):
            m = re.search(r'"action"\s*:\s*(\d)', raw)
            action = int(m.group(1)) if m else 0
            reason = ""

        if reason:
            print(f"    {DIM('[LLM]')} {reason}")

        return action if action in VALID_ACTIONS else 0

    except Exception as exc:
        print(f"[llm_agent] API call failed ({exc}). Falling back to baseline.")
        return baseline_agent(observation)


# ---------------------------------------------------------------------------
# Episode runner — standard (returns summary dict)
# ---------------------------------------------------------------------------
def run_episode(task_level: str, agent_fn) -> dict:
    """Run one complete episode; return a results dict."""
    env  = DataCleaningEnv()
    obs  = env.reset(task_level=task_level)
    done = False
    total_reward = 0

    while not done:
        action = agent_fn(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

    return {
        "task":         task_level,
        "score":        round(env.grade(), 3),
        "total_reward": total_reward,
        "steps":        env.steps,
        "fixed":        env.total_issues_at_start - len(env.issues),
        "total_issues": env.total_issues_at_start,
        "episode_log":  list(env.episode_log),
    }


# ---------------------------------------------------------------------------
# Episode runner — verbose (prints step-by-step trace during run)
# ---------------------------------------------------------------------------
def run_episode_verbose(task_level: str, agent_fn, agent_label: str = "agent") -> dict:
    """
    Run one complete episode and print a step-by-step decision trace.
    Shows what issue was found, what action was taken, and whether it was correct.
    """
    env  = DataCleaningEnv()
    obs  = env.reset(task_level=task_level)
    done = False
    total_reward = 0

    print(f"\n  {BOLD('Episode trace')} — {agent_label} / {task_level}")
    print(f"  {'Step':>4}  {'Col':<22} {'Issue':<22} {'Action':<18} {'✓':>2} {'Reward':>6}")
    print(f"  {'─'*4}  {'─'*22} {'─'*22} {'─'*18} {'─'*2} {'─'*6}")

    while not done:
        action = agent_fn(obs)
        obs, reward, done, _ = env.step(action)
        total_reward += reward

        if env.episode_log:
            e = env.episode_log[-1]
            mark = GREEN("✓") if e["correct"] else RED("✗")
            reward_str = GREEN(f"{e['reward']:+.0f}") if e["reward"] > 0 else RED(f"{e['reward']:+.0f}")
            print(
                f"  {e['step']:>4}  {e['col']:<22} {e['issue']:<22} "
                f"{e['action']:<18} {mark:>2} {reward_str:>6}"
            )

    score = env.grade()
    score_str = GREEN(f"{score:.3f}") if score == 1.0 else YELLOW(f"{score:.3f}")
    print(f"\n  {'─'*78}")
    print(f"  Score: {score_str}   Steps: {env.steps}   "
          f"Fixed: {env.total_issues_at_start - len(env.issues)}/{env.total_issues_at_start}   "
          f"Total reward: {total_reward}")

    return {
        "task":         task_level,
        "score":        round(score, 3),
        "total_reward": total_reward,
        "steps":        env.steps,
        "fixed":        env.total_issues_at_start - len(env.issues),
        "total_issues": env.total_issues_at_start,
        "episode_log":  list(env.episode_log),
    }


def run_all_tasks(agent_fn, label: str = "") -> list[dict]:
    levels = ("easy", "medium", "hard")
    results = []
    for lvl in levels:
        if label:
            print(f"  Running {label} / {lvl} ...", end="", flush=True)
            t0 = time.monotonic()
        r = run_episode(lvl, agent_fn)
        if label:
            elapsed = time.monotonic() - t0
            print(f"  score={r['score']:.2f}  ({elapsed:.1f}s)")
        results.append(r)
    return results


# ---------------------------------------------------------------------------
# Printers
# ---------------------------------------------------------------------------
SEP = "─" * 52

def print_results(results: list[dict], agent_name: str) -> None:
    print(f"\n{BOLD('=== RESULTS — ' + agent_name.upper() + ' AGENT ===')}\n")
    print(f"{'Task':<8} {'Score':>6} {'Reward':>8} {'Steps':>6} {'Fixed':>7}")
    print(SEP)
    for r in results:
        score_str = GREEN(f"{r['score']:>6.2f}") if r['score'] == 1.0 else YELLOW(f"{r['score']:>6.2f}")
        print(
            f"{r['task']:<8} {score_str} {r['total_reward']:>8} "
            f"{r['steps']:>6} {r['fixed']}/{r['total_issues']:>1}"
        )
    avg = sum(r["score"] for r in results) / len(results)
    avg_str = GREEN(f"{avg:>6.2f}") if avg == 1.0 else YELLOW(f"{avg:>6.2f}")
    print(SEP)
    print(f"{'Average':>8} {avg_str}")


def print_comparison(baseline: list[dict], llm: list[dict]) -> None:
    """
    Side-by-side comparison table.

    Example output:

    ╔══════════════════════════════════════════════════════════════════╗
    ║              AGENT COMPARISON — ALL TASK LEVELS                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║ Task    │  Baseline                │  LLM                  Δ    ║
    ║         │  Score  Reward  Steps   │  Score  Reward  Steps       ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║ easy    │   1.00    21.0      8   │   1.00    21.0      8  +0.00║
    ║ medium  │   1.00    29.0     12   │   1.00    29.0     12  +0.00║
    ║ hard    │   1.00    45.0     20   │   0.90    37.0     18  -0.10║
    ╠══════════════════════════════════════════════════════════════════╣
    ║ Average │   1.00                  │   0.97              Δ -0.03 ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    W = 68
    print(f"\n{'╔' + '═'*W + '╗'}")
    title = "AGENT COMPARISON — ALL TASK LEVELS"
    print(f"{'║'}{title:^{W}}{'║'}")
    print(f"{'╠' + '═'*W + '╣'}")
    print(f"║ {'Task':<7} │  {'Baseline':<22}  │  {'LLM':<22} {'Δ Score':>6} ║")
    print(f"║ {'':<7} │  {'Score':>5} {'Reward':>7} {'Steps':>6}   │  {'Score':>5} {'Reward':>7} {'Steps':>6}        ║")
    print(f"{'╠' + '═'*W + '╣'}")

    b_avg = sum(r["score"] for r in baseline) / len(baseline)
    l_avg = sum(r["score"] for r in llm)      / len(llm)

    for b, l in zip(baseline, llm):
        delta     = l["score"] - b["score"]
        delta_str = f"{delta:+.2f}"
        d_color   = GREEN if delta >= 0 else YELLOW

        b_score = GREEN(f"{b['score']:.2f}") if b["score"] == 1.0 else YELLOW(f"{b['score']:.2f}")
        l_score = GREEN(f"{l['score']:.2f}") if l["score"] == 1.0 else YELLOW(f"{l['score']:.2f}")

        print(
            f"║ {b['task']:<7} │  "
            f"{b_score} {float(b['total_reward']):>7.1f} {b['steps']:>6}   │  "
            f"{l_score} {float(l['total_reward']):>7.1f} {l['steps']:>6}  {d_color(delta_str):>7} ║"
        )

    print(f"{'╠' + '═'*W + '╣'}")
    total_delta = l_avg - b_avg
    td_str  = f"{total_delta:+.2f}"
    td_col  = GREEN if total_delta >= 0 else YELLOW
    b_avg_s = GREEN(f"{b_avg:.2f}") if b_avg == 1.0 else YELLOW(f"{b_avg:.2f}")
    l_avg_s = GREEN(f"{l_avg:.2f}") if l_avg == 1.0 else YELLOW(f"{l_avg:.2f}")
    print(
        f"║ {'Average':<7} │  {b_avg_s} {'':>21}  │  {l_avg_s} {'':>21} {td_col(td_str):>7} ║"
    )
    print(f"{'╚' + '═'*W + '╝'}")


def print_episode_summary(results: list[dict], agent_name: str) -> None:
    """Print a per-issue breakdown across all tasks for one agent."""
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
        acc = s["correct"] / s["total"] if s["total"] > 0 else 0
        acc_str = GREEN(f"{acc:>9.0%}") if acc == 1.0 else YELLOW(f"{acc:>9.0%}")
        print(f"  {issue:<24} {s['correct']:>7} {s['total']:>6} {acc_str}")


# ---------------------------------------------------------------------------
# LLM availability check
# ---------------------------------------------------------------------------
def _llm_vars_set() -> bool:
    return bool(
        (os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY"))
        and (os.environ.get("API_BASE_URL") or os.environ.get("OPENAI_BASE_URL"))
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run DataCleaningEnv episodes.")
    parser.add_argument(
        "--agent", choices=["baseline", "llm", "both"], default="auto",
        help=(
            "Agent to use. 'auto' (default): runs baseline always; "
            "also runs LLM if API_BASE_URL + HF_TOKEN are set. "
            "'both': force both even if LLM keys look invalid. "
            "Overridden by AGENT_TYPE env var."
        ),
    )
    parser.add_argument(
        "--task", choices=["easy", "medium", "hard", "all"], default="all",
        help="Task difficulty to run (default: all)",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print step-by-step episode trace for each run.",
    )
    args = parser.parse_args()

    # AGENT_TYPE env var takes precedence over --agent flag
    agent_choice = os.environ.get("AGENT_TYPE", args.agent)
    levels = ("easy", "medium", "hard") if args.task == "all" else (args.task,)

    # ── Decide which agents to run ────────────────────────────────────────
    run_baseline = agent_choice in ("baseline", "both", "auto")
    run_llm      = (
        agent_choice == "llm"
        or agent_choice == "both"
        or (agent_choice == "auto" and _llm_vars_set())
    )

    runner = run_episode_verbose if args.verbose else run_episode

    # ── Run baseline ──────────────────────────────────────────────────────
    baseline_results = None
    if run_baseline or run_llm:   # always need baseline for comparison
        print(f"\n{BOLD('Running baseline agent...')}")
        baseline_results = []
        for lvl in levels:
            if args.verbose:
                r = run_episode_verbose(lvl, baseline_agent, "baseline")
            else:
                print(f"  {lvl} ...", end="", flush=True)
                t0 = time.monotonic()
                r = run_episode(lvl, baseline_agent)
                print(f"  score={r['score']:.2f}  ({time.monotonic()-t0:.1f}s)")
            baseline_results.append(r)

    # ── Run LLM agent ─────────────────────────────────────────────────────
    llm_results = None
    if run_llm:
        model = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL", "unknown")
        print(f"\n{BOLD('Running LLM agent')} {DIM('(model: ' + model + ')')}")
        llm_results = []
        for lvl in levels:
            if args.verbose:
                r = run_episode_verbose(lvl, llm_agent, "llm")
            else:
                print(f"  {lvl} ...", end="", flush=True)
                t0 = time.monotonic()
                r = run_episode(lvl, llm_agent)
                print(f"  score={r['score']:.2f}  ({time.monotonic()-t0:.1f}s)")
            llm_results.append(r)

    # ── Print results ─────────────────────────────────────────────────────
    if llm_results and baseline_results and args.task == "all":
        print_comparison(baseline_results, llm_results)
        # Per-issue accuracy breakdown for both agents
        print_episode_summary(baseline_results, "baseline")
        print_episode_summary(llm_results, "llm")
    elif llm_results:
        print_results(baseline_results, "baseline")
        print_results(llm_results, "llm")
        print_episode_summary(llm_results, "llm")
    elif baseline_results:
        print_results(baseline_results, "baseline")
        print_episode_summary(baseline_results, "baseline")
        if not _llm_vars_set():
            print(
                f"\n{DIM('Tip: set HF_TOKEN + API_BASE_URL + MODEL_NAME to also run the LLM agent.')}"
            )
