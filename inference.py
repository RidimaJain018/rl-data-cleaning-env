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


# ---------------------------------------------------------------------------
# LLM Agent
# ---------------------------------------------------------------------------
def llm_agent(observation: dict | None) -> int:
    """
    LLM-powered agent. Reads the three required hackathon env variables:
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

    row_lines, hints = [], []
    for col, val in row.items():
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            row_lines.append(f"  {col}: MISSING (NaN)")
            hints.append(f"{col} is missing")
        else:
            row_lines.append(f"  {col}: {val}")
            if col in {"age", "salary", "experience", "rating"} and isinstance(val, str):
                hints.append(f"{col}='{val}' type mismatch")
            elif isinstance(val, str) and val != val.strip():
                hints.append(f"{col}='{val}' whitespace padding")
            elif col == "salary" and isinstance(val, (int, float)) and val > 200_000:
                hints.append(f"salary={val} is an outlier (> 200,000)")
            elif col == "rating" and isinstance(val, (int, float)) and val > 5:
                hints.append(f"rating={val} is invalid (> 5)")
            elif isinstance(val, (int, float)) and val < 0:
                hints.append(f"{col}={val} is negative (invalid)")

    system_prompt = (
        "You are a data-cleaning agent. Choose exactly one action:\n"
        "  0 = skip           (row has no issue)\n"
        "  1 = impute_missing  (row has a missing / NaN value)\n"
        "  3 = fix_outlier    (row has an outlier, invalid value, type mismatch, "
        "whitespace padding, or is a duplicate)\n\n"
        'Reply with ONLY a JSON object: {"action": <0|1|3>}'
    )
    user_prompt = (
        "Row data:\n" + "\n".join(row_lines) + "\n\n"
        f"Issue hint: {'; '.join(hints) or 'no obvious issue'}\n\n"
        'Reply ONLY with JSON, e.g. {"action": 1}'
    )

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0,
            max_tokens=32,
        )
        raw = response.choices[0].message.content.strip()
        try:
            action = int(json.loads(raw)["action"])
        except (json.JSONDecodeError, KeyError, ValueError):
            m = re.search(r'"action"\s*:\s*(\d)', raw)
            action = int(m.group(1)) if m else 0

        return action if action in VALID_ACTIONS else 0

    except Exception as exc:
        print(f"[llm_agent] API call failed ({exc}). Falling back to baseline.")
        return baseline_agent(observation)


# ---------------------------------------------------------------------------
# Episode runner
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
SEP = "─" * 48

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
    ║ easy    │   1.00    13.0      4   │   1.00    13.0      4  +0.00║
    ║ medium  │   1.00    15.0      5   │   1.00    15.0      5  +0.00║
    ║ hard    │   1.00    27.0     11   │   0.82    19.0      8  -0.18║
    ╠══════════════════════════════════════════════════════════════════╣
    ║ Average │   1.00                  │   0.94              Δ -0.06 ║
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

        # raw widths for alignment (strip ANSI for counting)
        raw_line = (
            f"║ {b['task']:<7} │  "
            f"{b['score']:>5.2f} {float(b['total_reward']):>7.1f} {b['steps']:>6}   │  "
            f"{l['score']:>5.2f} {float(l['total_reward']):>7.1f} {l['steps']:>6}  {delta_str:>7} ║"
        )
        # Coloured version
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

    # ── Run baseline ──────────────────────────────────────────────────────
    baseline_results = None
    if run_baseline or run_llm:   # always need baseline for comparison
        print(f"\n{BOLD('Running baseline agent...')}")
        baseline_results = [run_episode(lvl, baseline_agent) for lvl in levels]

    # ── Run LLM agent ─────────────────────────────────────────────────────
    llm_results = None
    if run_llm:
        model = os.environ.get("MODEL_NAME") or os.environ.get("OPENAI_MODEL", "unknown")
        print(f"\n{BOLD('Running LLM agent')} {DIM('(model: ' + model + ')')}")
        llm_results = []
        for lvl in levels:
            print(f"  {lvl} ...", end="", flush=True)
            t0 = time.monotonic()
            r  = run_episode(lvl, llm_agent)
            print(f"  score={r['score']:.2f}  ({time.monotonic()-t0:.1f}s)")
            llm_results.append(r)

    # ── Print results ─────────────────────────────────────────────────────
    if llm_results and baseline_results and args.task == "all":
        print_comparison(baseline_results, llm_results)
    elif llm_results:
        print_results(baseline_results, "baseline")
        print_results(llm_results, "llm")
    elif baseline_results:
        print_results(baseline_results, "baseline")
        if not _llm_vars_set():
            print(
                f"\n{DIM('Tip: set HF_TOKEN + API_BASE_URL + MODEL_NAME to also run the LLM agent.')}"
            )
