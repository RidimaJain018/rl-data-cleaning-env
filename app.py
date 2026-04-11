"""
app.py — FastAPI OpenEnv server for DataCleaningEnv
====================================================
Endpoints (OpenEnv 3-method interface + extras):
    POST /reset      Start a new episode
    POST /step       Take one action
    GET  /state      Full episode snapshot (no side effects)
    POST /run        Run a complete episode with the baseline agent
    POST /upload     Upload any CSV and start a cleaning episode
    GET  /health     Liveness probe → {"status": "ok"}

Session isolation:
    Each client identifies itself via the X-Session-Id header.
    Multiple clients can run simultaneous episodes on the same server
    without interfering — each gets its own DataCleaningEnv instance.

Start locally:
    uvicorn app:app --host 0.0.0.0 --port 8000 --reload

Or via start.sh (production — runs FastAPI + Gradio together):
    ./start.sh
"""

from __future__ import annotations

import io
import os
from typing import Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from agent import baseline_agent, upload_agent
from env import DataCleaningEnv
from models import (
    ActionModel,
    EnvStateModel,
    ObservationModel,
    RewardModel,
    StepResultModel,
)

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = FastAPI(
    title="DataCleaningEnv — OpenEnv API",
    description=(
        "Reinforcement-learning environment for tabular data cleaning. "
        "Implements the OpenEnv 3-method interface: reset / step / state."
    ),
    version="1.2.0",
)

# ---------------------------------------------------------------------------
# Session store — one DataCleaningEnv per X-Session-Id
# ---------------------------------------------------------------------------
_sessions: dict[str, DataCleaningEnv] = {}

_DEFAULT_SESSION = "default"


def _get_env(session_id: str) -> DataCleaningEnv:
    """Return the env for this session, creating it if needed."""
    if session_id not in _sessions:
        _sessions[session_id] = DataCleaningEnv()
    return _sessions[session_id]


def _session_id(x_session_id: Optional[str]) -> str:
    return x_session_id or _DEFAULT_SESSION


# ---------------------------------------------------------------------------
# Request bodies
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_level: str = "medium"


class RunRequest(BaseModel):
    task_level: str = "medium"
    agent: str = "baseline"


# ---------------------------------------------------------------------------
# Helper — sanitise observation dict for JSON (NaN → None)
# ---------------------------------------------------------------------------
def _sanitise_obs(obs: dict | None) -> dict | None:
    """Replace float NaN with None so Pydantic / JSON serialisation works."""
    if obs is None:
        return None
    clean = {
        k: (None if (isinstance(v, float) and np.isnan(v)) else v)
        for k, v in obs["row_data"].items()
    }
    return {"row_data": clean}


def _obs_model(obs: dict | None) -> Optional[ObservationModel]:
    if obs is None:
        return None
    clean_row = {
        k: (None if (isinstance(v, float) and np.isnan(v)) else v)
        for k, v in obs["row_data"].items()
    }
    return ObservationModel(
        row_data=clean_row,
        column_stats=obs.get("column_stats"),
        step_progress=obs.get("step_progress"),
        issues_remaining=obs.get("issues_remaining"),
    )


# ---------------------------------------------------------------------------
# POST /reset
# ---------------------------------------------------------------------------
@app.post("/reset")
async def reset(request: Request, x_session_id: Optional[str] = Header(default=None)):
    """
    Start a new episode.

    Accepts an empty body OR {"task_level": "easy"|"medium"|"hard"}.
    The automated hackathon checker sends an empty POST — this is handled
    by reading the raw request body and falling back to defaults.
    """
    # Parse body flexibly — empty body and missing task_level are both valid
    try:
        body = await request.json()
    except Exception:
        body = {}

    task_level = body.get("task_level", "medium") if isinstance(body, dict) else "medium"

    if task_level not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task_level {task_level!r}. Must be 'easy', 'medium', or 'hard'.",
        )

    sid = _session_id(x_session_id)
    env = _get_env(sid)
    obs = env.reset(task_level=task_level)

    return {"observation": _sanitise_obs(obs)}


# ---------------------------------------------------------------------------
# POST /step
# ---------------------------------------------------------------------------
@app.post("/step", response_model=StepResultModel)
async def step(
    action_req: ActionModel,
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Take one action in the current episode.

    action: 0 = skip | 1 = impute_missing | 3 = fix_outlier
    """
    sid = _session_id(x_session_id)
    env = _get_env(sid)

    if env.df is None:
        raise HTTPException(
            status_code=400,
            detail="No active episode. Call POST /reset first.",
        )

    obs, reward, done, info = env.step(action_req.action)

    return StepResultModel(
        observation=_obs_model(obs),
        reward=RewardModel(value=float(reward), done=done, step=env.steps),
        info=info,
    )


# ---------------------------------------------------------------------------
# GET /state
# ---------------------------------------------------------------------------
@app.get("/state", response_model=EnvStateModel)
async def state(x_session_id: Optional[str] = Header(default=None)):
    """
    Return a full snapshot of the current episode without advancing it.
    Safe to call at any time — has no side effects.
    """
    sid = _session_id(x_session_id)
    env = _get_env(sid)

    raw = env.state()

    # Sanitise current_observation for Pydantic
    current_obs = None
    if raw.get("current_observation") is not None:
        current_obs = _obs_model(raw["current_observation"])

    return EnvStateModel(
        task_level=raw["task_level"],
        current_step=raw["current_step"],
        max_steps=raw["max_steps"],
        total_issues_at_start=raw["total_issues_at_start"],
        remaining_issues=raw["remaining_issues"],
        score=raw["score"],
        done=raw["done"],
        current_observation=current_obs,
        episode_log=raw["episode_log"],
    )


# ---------------------------------------------------------------------------
# POST /run  — run a full episode with the baseline agent, return final state
# ---------------------------------------------------------------------------
@app.post("/run", response_model=EnvStateModel)
async def run(
    run_req: RunRequest,
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Run a complete episode from start to finish using the baseline agent.
    Returns the final EnvStateModel. Useful for smoke tests and CI pipelines.
    """
    if run_req.task_level not in ("easy", "medium", "hard"):
        raise HTTPException(
            status_code=422,
            detail=f"Invalid task_level {run_req.task_level!r}.",
        )

    sid = _session_id(x_session_id)
    env = _get_env(sid)
    obs = env.reset(task_level=run_req.task_level)

    agent_fn = baseline_agent   # only baseline supported for /run
    done = False

    while not done:
        action = agent_fn(obs)
        obs, _, done, _ = env.step(action)

    raw = env.state()
    return EnvStateModel(
        task_level=raw["task_level"],
        current_step=raw["current_step"],
        max_steps=raw["max_steps"],
        total_issues_at_start=raw["total_issues_at_start"],
        remaining_issues=raw["remaining_issues"],
        score=raw["score"],
        done=raw["done"],
        current_observation=None,
        episode_log=raw["episode_log"],
    )


# ---------------------------------------------------------------------------
# POST /upload  — accept any CSV, detect issues, start cleaning episode
# ---------------------------------------------------------------------------
@app.post("/upload")
async def upload(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Upload any CSV file to start a cleaning episode.

    Auto-detects numeric vs categorical columns and applies IQR-based
    outlier detection — works with any column names and row counts.

    Returns the first observation plus dataset metadata.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=422,
            detail="Only CSV files are supported.",
        )

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not parse CSV: {exc}",
        )

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV is empty.")

    sid = _session_id(x_session_id)
    env = _get_env(sid)

    try:
        obs = env.reset_from_dataframe(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    return {
        "observation":   _sanitise_obs(obs),
        "total_issues":  env.total_issues_at_start,
        "rows":          len(df),
        "columns":       list(df.columns),
    }



# ---------------------------------------------------------------------------
# POST /evaluate_upload  — upload CSV, run full episode with trained Q-agent,
#                          return score. Closes the "CSV not in eval loop" gap.
# ---------------------------------------------------------------------------
@app.post("/evaluate_upload")
async def evaluate_upload(
    file: UploadFile = File(...),
    x_session_id: Optional[str] = Header(default=None),
):
    """
    Upload any CSV and run a full cleaning episode using the trained Q-policy
    (falls back to baseline_agent if policy.pkl is not present).

    Returns: score, steps taken, issues found, and the full episode log.
    This endpoint closes the loop between CSV upload and RL evaluation —
    the trained agent is evaluated on unseen data, not just the 3 fixed tasks.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=422, detail="Only CSV files are supported.")

    try:
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(status_code=422, detail="Uploaded CSV is empty.")

    sid = _session_id(x_session_id)
    env = _get_env(sid)

    try:
        obs = env.reset_from_dataframe(df)
    except Exception as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    # Load trained Q-policy if available, else use baseline
    agent_fn   = baseline_agent
    agent_name = "baseline"
    policy_path = os.path.join(os.path.dirname(__file__), "policy.pkl")
    if os.path.exists(policy_path):
        try:
            import pickle, math
            from env import EXPECTED_NUMERIC as _ENC
            with open(policy_path, "rb") as _f:
                _artifact = pickle.load(_f)
            _policy = _artifact["policy"]

            def _parseable(s):
                try: float(s); return True
                except: return False

            def _q_agent(observation):
                if observation is None:
                    return 0
                row = observation["row_data"]
                nv = {k: float(v) for k, v in row.items()
                      if k in _ENC and isinstance(v, (int, float))
                      and not (isinstance(v, float) and math.isnan(v))}
                if nv:
                    vals = list(nv.values())
                    mean = sum(vals) / len(vals)
                    std  = (sum((x-mean)**2 for x in vals)/len(vals))**0.5 or 1.0
                else:
                    mean, std = 0.0, 1.0
                features = []
                for col in sorted(row.keys()):
                    val = row[col]
                    is_null = val is None or (isinstance(val, float) and math.isnan(val))
                    is_str  = col in _ENC and isinstance(val, str) and not _parseable(val)
                    has_ws  = isinstance(val, str) and val != val.strip()
                    if col in _ENC and isinstance(val, (int, float)) and not is_null:
                        z = (float(val) - mean) / std
                        bucket = -2 if z < -2 else (-1 if z < -0.5 else (0 if z <= 0.5 else (1 if z <= 2 else 2)))
                    else:
                        bucket = 0
                    features.append((col, int(is_null), int(is_str), int(has_ws), bucket))
                state = tuple(features)
                return _policy.get(state, upload_agent(observation))

            agent_fn   = _q_agent
            agent_name = f"Q-policy({len(_policy)}_states)"
        except Exception as _e:
            pass  # fall through to baseline

    done = False
    while not done:
        action = agent_fn(obs)
        obs, _, done, _ = env.step(action)

    raw = env.state()
    return {
        "agent":           agent_name,
        "score":           raw["score"],
        "steps":           raw["current_step"],
        "total_issues":    raw["total_issues_at_start"],
        "remaining":       raw["remaining_issues"],
        "rows":            len(df),
        "columns":         list(df.columns),
        "episode_log":     raw["episode_log"],
    }


# ---------------------------------------------------------------------------
# GET /health  — liveness probe
# ---------------------------------------------------------------------------
@app.get("/health")
async def health():
    """Liveness probe. Returns {"status": "ok"} when the server is running."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Entry point (used by start.sh via uvicorn app:app)
# ---------------------------------------------------------------------------
def main():
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    main()
