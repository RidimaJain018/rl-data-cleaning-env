"""
inference.py — Run DataCleaningEnv episodes with different agents
=================================================================
SELF-CONTAINED: DataCleaningEnv is embedded directly — no external imports needed.

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
# Ensure env.py is importable if present (fallback — not required)
# ---------------------------------------------------------------------------
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
except Exception:
    pass

# ---------------------------------------------------------------------------
# Force stdout unbuffered — safe even if checker replaces sys.stdout
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Required environment variables (hackathon spec)
# ---------------------------------------------------------------------------
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "meta-llama/Llama-3.2-3B-Instruct")
HF_TOKEN     = os.getenv("HF_TOKEN", "")

BENCHMARK               = "data-cleaning-env"
SUCCESS_SCORE_THRESHOLD = 0.5

# ===========================================================================
# EMBEDDED DataCleaningEnv — copied from env.py so this file is self-contained
# ===========================================================================
import pandas as pd
import numpy as np

_EXPECTED_NUMERIC = {
    "age", "salary", "experience", "rating",
    "bonus", "years_at_company", "performance_score", "overtime_hours",
}
_OUTLIER_THRESHOLDS = {"salary": 300_000, "bonus": 80_000}
_SCORE_COLS         = {"rating", "performance_score"}
_ACTION_LABELS      = {0: "skip", 1: "impute_missing", 3: "fix_outlier"}
_SEEDS              = {"easy": 42, "medium": 123, "hard": 999}
_CITIES             = ["NY", "LA", "SF", "Chicago", "Austin", "Seattle", "Boston", "Denver"]
_DEPARTMENTS        = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]


def _clean_row(rng: np.random.Generator) -> dict:
    return {
        "age":               int(rng.integers(22, 62)),
        "salary":            int(rng.integers(35_000, 180_000)),
        "city":              _CITIES[int(rng.integers(0, len(_CITIES)))],
        "experience":        int(rng.integers(0, 30)),
        "rating":            round(float(rng.uniform(2.5, 4.9)), 1),
        "department":        _DEPARTMENTS[int(rng.integers(0, len(_DEPARTMENTS)))],
        "bonus":             int(rng.integers(0, 25_000)),
        "years_at_company":  int(rng.integers(0, 20)),
        "performance_score": round(float(rng.uniform(2.5, 4.9)), 1),
        "overtime_hours":    int(rng.integers(0, 80)),
    }


class DataCleaningEnv:

    def __init__(self):
        self.df               = None
        self.original_df      = None
        self.issues           = []
        self.current_idx      = 0
        self.steps            = 0
        self.max_steps        = 100
        self.episode_log      = []
        self._task_level      = None
        self._is_custom       = False

    def load_task_data(self, task_level: str) -> list:
        rng = np.random.default_rng(_SEEDS[task_level])

        if task_level == "easy":
            rows = [_clean_row(rng) for _ in range(20)]
            rows[0]["age"]               = None
            rows[1]["city"]              = None
            rows[2]["experience"]        = None
            rows[3]["rating"]            = None
            rows[4]["department"]        = None
            rows[5]["bonus"]             = None
            rows[6]["years_at_company"]  = None
            rows[7]["performance_score"] = None
            return rows

        elif task_level == "medium":
            rows = [_clean_row(rng) for _ in range(30)]
            rows[0]["age"]               = None
            rows[1]["city"]              = None
            rows[2]["experience"]        = None
            rows[3]["department"]        = None
            rows[4]["overtime_hours"]    = None
            rows[5]["salary"]  = int(rng.integers(320_000, 900_000))
            rows[6]["salary"]  = int(rng.integers(600_000, 2_000_000))
            rows[7]["bonus"]   = int(rng.integers(90_000, 250_000))
            rows[8]["performance_score"]  = round(float(rng.uniform(5.5, 9.0)), 1)
            rows[9]["rating"]             = round(float(rng.uniform(5.5, 8.0)), 1)
            rows[10]["overtime_hours"]    = -int(rng.integers(5, 40))
            rows[11]["city"]              = f" {rows[11]['city']} "
            return rows

        elif task_level == "hard":
            rows = [_clean_row(rng) for _ in range(50)]
            rows[0]["age"]               = None
            rows[2]["city"]              = None
            rows[3]["experience"]        = None
            rows[4]["rating"]            = None
            rows[5]["department"]        = None
            rows[6]["performance_score"] = None
            rows[1]["salary"]  = int(rng.integers(400_000, 1_200_000))
            rows[7]["salary"]  = int(rng.integers(700_000, 2_500_000))
            rows[8]["salary"]  = int(rng.integers(1_000_000, 4_000_000))
            rows[9]["bonus"]   = int(rng.integers(100_000, 350_000))
            rows[10]["overtime_hours"]    = -int(rng.integers(5, 40))
            rows[11]["years_at_company"]  = -int(rng.integers(1, 10))
            rows[12] = dict(rows[1])
            rows[13] = dict(rows[7])
            rows[14]["age"]              = "N/A"
            rows[15]["years_at_company"] = "ten"
            rows[16]["city"]             = f" {rows[16]['city']} "
            rows[17]["department"]       = f"  {rows[17]['department']}  "
            rows[18]["performance_score"] = round(float(rng.uniform(5.5, 9.0)), 1)
            rows[19]["rating"]            = round(float(rng.uniform(5.5, 8.0)), 1)
            return rows

        else:
            raise ValueError(f"Unknown task_level {task_level!r}.")

    def reset(self, task_level: str = "medium"):
        self._task_level = task_level
        self._is_custom  = False
        self.df          = pd.DataFrame(self.load_task_data(task_level))
        self.original_df = self.df.copy()
        self.episode_log = []
        self._detect_builtin_schema()
        self.compute_stats()
        self._detect_builtin_issues()
        self.current_idx           = 0
        self.steps                 = 0
        self.total_issues_at_start = len(self.issues)
        return self.get_observation()

    def _detect_builtin_schema(self):
        self.numeric_cols = [c for c in self.df.columns if c in _EXPECTED_NUMERIC]
        self.cat_cols     = [c for c in self.df.columns if c not in _EXPECTED_NUMERIC]

    def compute_stats(self):
        self.means = {}
        for col in self.numeric_cols:
            s = pd.to_numeric(self.df[col], errors="coerce")
            m = s.mean(skipna=True)
            self.means[col] = float(m) if not pd.isna(m) else 0.0
        self.modes = {}
        for col in self.cat_cols:
            clean  = self.df[col].dropna().apply(lambda v: v.strip() if isinstance(v, str) else v)
            mode_s = clean.mode(dropna=True)
            self.modes[col] = str(mode_s.iloc[0]) if not mode_s.empty else ""

    def _row_key(self, row) -> tuple:
        return tuple(None if (isinstance(v, float) and pd.isna(v)) else v for v in row)

    def _detect_builtin_issues(self):
        self.issues  = []
        seen_keys: set = set()
        for idx, row in self.df.iterrows():
            key = self._row_key(row)
            if key in seen_keys:
                self.issues.append((idx, "__row__", "duplicate"))
                continue
            seen_keys.add(key)
            for col in self.df.columns:
                val = row[col]
                if col in _EXPECTED_NUMERIC and isinstance(val, str):
                    self.issues.append((idx, col, "type_mismatch")); break
                if isinstance(val, str) and val != val.strip():
                    self.issues.append((idx, col, "whitespace_padding")); break
                if not isinstance(val, str) and pd.isna(val):
                    self.issues.append((idx, col, "missing")); break
                if col in _OUTLIER_THRESHOLDS and isinstance(val, (int, float)) and val > _OUTLIER_THRESHOLDS[col]:
                    self.issues.append((idx, col, "outlier")); break
                if col in _SCORE_COLS and isinstance(val, (int, float)) and val > 5:
                    self.issues.append((idx, col, "invalid_rating")); break
                if col in _EXPECTED_NUMERIC and isinstance(val, (int, float)) and val < 0:
                    self.issues.append((idx, col, "invalid_negative")); break

    def get_observation(self):
        if not self.issues:
            return None
        if self.current_idx >= len(self.issues):
            self.current_idx = 0
        row_idx, _, _ = self.issues[self.current_idx]
        if row_idx not in self.df.index:
            self.current_idx = 0
            row_idx, _, _ = self.issues[0]
        _, _, issue_type = self.issues[self.current_idx]
        return {"row_data": self.df.loc[row_idx].to_dict(), "_issue": issue_type}

    def step(self, action: int):
        if not self.issues:
            return None, 0, True, {}
        self.steps += 1
        row_idx, col, issue_type = self.issues[self.current_idx]
        correct_action = 1 if issue_type == "missing" else 3
        correct  = (action == correct_action)
        old_val  = str(self.df.loc[row_idx, col]) if col != "__row__" else "duplicate_row"
        new_val  = "—"
        reward   = 0
        skip_pop = False

        if correct:
            if issue_type == "missing":
                if col in self.numeric_cols:
                    nv = self.means.get(col, 0.0)
                    if self.df[col].dtype == "int64":
                        self.df[col] = self.df[col].astype("float64")
                    self.df.loc[row_idx, col] = nv
                    new_val = str(round(nv, 4))
                else:
                    nv = self.modes.get(col, "")
                    self.df.loc[row_idx, col] = nv
                    new_val = str(nv)
            elif issue_type == "duplicate":
                new_val  = "dropped"
                self.df  = self.df.drop(index=row_idx).reset_index(drop=True)
                remaining = [(r, c, t) for i, (r, c, t) in enumerate(self.issues) if i != self.current_idx]
                self.issues = [(r - 1 if r > row_idx else r, c, t) for (r, c, t) in remaining]
                skip_pop = True
            elif issue_type == "type_mismatch":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                col_mean = pd.to_numeric(self.df[col], errors="coerce").mean(skipna=True)
                nv = float(col_mean) if not pd.isna(col_mean) else 0.0
                self.df.loc[row_idx, col] = nv
                new_val = str(round(nv, 4))
            elif issue_type == "whitespace_padding":
                nv = str(self.df.loc[row_idx, col]).strip()
                self.df.loc[row_idx, col] = nv
                new_val = nv
            else:
                nv = self.means.get(col, 0.0)
                if self.df[col].dtype == "int64":
                    self.df[col] = self.df[col].astype("float64")
                self.df.loc[row_idx, col] = nv
                new_val = str(round(nv, 4))

            reward = 2
            self._detect_builtin_schema()
            self.compute_stats()
            if not skip_pop:
                self.issues.pop(self.current_idx)
            if self.current_idx >= len(self.issues):
                self.current_idx = 0
        else:
            reward = -1
            self.current_idx += 1
            if self.current_idx >= len(self.issues):
                self.current_idx = 0

        self.episode_log.append({
            "step":      self.steps,
            "row":       int(row_idx),
            "col":       col,
            "issue":     issue_type,
            "action":    _ACTION_LABELS.get(action, str(action)),
            "correct":   correct,
            "old_value": old_val,
            "new_value": new_val,
            "reward":    float(reward),
        })

        if not self.issues:
            return None, reward + 5, True, {}
        if self.steps >= self.max_steps:
            return None, reward - 5, True, {}
        return self.get_observation(), reward, False, {}

    def grade(self) -> float:
        total = getattr(self, "total_issues_at_start", 0)
        if total == 0:
            return 1.0
        return round((total - len(self.issues)) / total, 3)

# ===========================================================================
# END OF EMBEDDED DataCleaningEnv
# ===========================================================================


# ---------------------------------------------------------------------------
# Official structured stdout loggers — exact format the checker parses
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
# Baseline agent
# ---------------------------------------------------------------------------
def _baseline_agent(observation: dict | None) -> int:
    if observation is None:
        return 0
    row = observation["row_data"]

    for key, val in row.items():
        if key in _EXPECTED_NUMERIC and isinstance(val, str):
            try:
                float(val)
            except (ValueError, TypeError):
                return 3

    for val in row.values():
        if isinstance(val, str) and val != val.strip():
            return 3

    for key, val in row.items():
        if key in _OUTLIER_THRESHOLDS and isinstance(val, (int, float)):
            if val > _OUTLIER_THRESHOLDS[key]:
                return 3

    for key, val in row.items():
        if key in _SCORE_COLS and isinstance(val, (int, float)):
            if val > 5:
                return 3

    for key, val in row.items():
        if key in _EXPECTED_NUMERIC and isinstance(val, (int, float)):
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
# Main — unconditional call so checker works via direct run OR import
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


main()
