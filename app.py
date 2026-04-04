"""
app.py — OpenEnv-compliant FastAPI server for DataCleaningEnv v1.2
==================================================================
Endpoints
---------
POST /reset    → {"observation": ObservationModel | null}
POST /step     → StepResultModel
GET  /state    → EnvStateModel
POST /run      → EnvStateModel  (full baseline episode in one call)
POST /upload   → {"observation": ..., "total_issues": int, "rows": int, "columns": [...]}
GET  /health   → {"status": "ok"}

Sessions keyed by X-Session-Id header ("default" if absent).
LRU eviction at MAX_SESSIONS = 100 prevents memory growth in long-running Spaces.
"""
from __future__ import annotations

import io
from collections import OrderedDict
from typing import Literal, Optional

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from env import DataCleaningEnv
from models import (
    ActionModel, EnvStateModel, EpisodeLogEntryModel,
    ObservationModel, RewardModel, StepResultModel,
)

app = FastAPI(
    title="DataCleaningEnv — OpenEnv Server",
    description="RL environment for tabular data cleaning. "
                "Implements reset / step / state. Accepts user CSVs via /upload.",
    version="1.2.0",
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Session store — LRU-bounded OrderedDict
# ---------------------------------------------------------------------------
MAX_SESSIONS = 100
_sessions: OrderedDict[str, DataCleaningEnv] = OrderedDict()
DEFAULT_SESSION = "default"


def _get_env(sid: str) -> DataCleaningEnv:
    if sid in _sessions:
        _sessions.move_to_end(sid)
        return _sessions[sid]
    if len(_sessions) >= MAX_SESSIONS:
        _sessions.popitem(last=False)
    env = DataCleaningEnv()
    _sessions[sid] = env
    return env


def _require_env(sid: str) -> DataCleaningEnv:
    if sid not in _sessions:
        raise HTTPException(400, "No active session. Call POST /reset or POST /upload first.")
    _sessions.move_to_end(sid)
    return _sessions[sid]


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    task_level: str = "medium"

    model_config = {"extra": "allow"}

class RunRequest(BaseModel):
    task_level: Literal["easy", "medium", "hard"] = "medium"
    agent: Literal["baseline"] = "baseline"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _obs_to_model(raw: Optional[dict]) -> Optional[ObservationModel]:
    if raw is None:
        return None
    cleaned = {}
    for k, v in raw["row_data"].items():
        if isinstance(v, float) and np.isnan(v):       cleaned[k] = None
        elif isinstance(v, np.integer):                 cleaned[k] = int(v)
        elif isinstance(v, np.floating):                cleaned[k] = float(v)
        else:                                           cleaned[k] = v
    return ObservationModel(row_data=cleaned)


def _build_state(env: DataCleaningEnv, done: bool = False) -> EnvStateModel:
    total = getattr(env, "total_issues_at_start", 0)
    return EnvStateModel(
        task_level=env._task_level,
        current_step=env.steps,
        max_steps=env.max_steps,
        total_issues_at_start=total,
        remaining_issues=len(env.issues),
        score=env.grade() if total > 0 else 1.0,
        done=done,
        current_observation=_obs_to_model(env.get_observation() if not done else None),
        episode_log=[EpisodeLogEntryModel(**e) for e in env.episode_log],
    )


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset")
async def reset(request: Request, x_session_id: Optional[str] = Header(default=None)) -> dict:
    """
    Start a new episode. Accepts an optional JSON body with task_level.
    Works with empty body, null body, or {"task_level": "easy"|"medium"|"hard"}.
    """
    sid = x_session_id or DEFAULT_SESSION
    env = _get_env(sid)

    # Safely parse body — handle empty, null, or missing body gracefully
    task_level = "medium"
    try:
        body_bytes = await request.body()
        if body_bytes and body_bytes.strip() and body_bytes.strip() != b"null":
            import json
            body_json = json.loads(body_bytes)
            if isinstance(body_json, dict):
                task_level = body_json.get("task_level", "medium") or "medium"
    except Exception:
        pass  # Fall back to default task_level

    obs = _obs_to_model(env.reset(task_level=task_level))
    return {"observation": obs.model_dump() if obs else None}


@app.post("/upload")
async def upload(
    file: UploadFile = File(..., description="CSV file to clean"),
    x_session_id: Optional[str] = Header(default=None),
) -> dict:
    """
    Upload any CSV file to start a cleaning episode.

    The server auto-detects numeric vs categorical columns and applies
    IQR-based outlier detection — works with any column names and row counts.
    The same /step, /state, and /run endpoints work after uploading.
    """
    raw = await file.read()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")

    try:
        df = pd.read_csv(io.StringIO(text))
    except Exception as exc:
        raise HTTPException(400, f"Could not parse CSV: {exc}")

    if df.empty:
        raise HTTPException(400, "Uploaded CSV is empty.")
    if len(df.columns) < 2:
        raise HTTPException(400, "CSV must have at least 2 columns.")

    sid = x_session_id or DEFAULT_SESSION
    env = _get_env(sid)
    try:
        obs = env.reset_from_dataframe(df)
    except Exception as exc:
        raise HTTPException(400, f"Could not load DataFrame: {exc}")

    return {
        "observation":  _obs_to_model(obs).model_dump() if obs else None,
        "total_issues": env.total_issues_at_start,
        "rows":         len(df),
        "columns":      list(df.columns),
    }


@app.post("/step")
def step(body: ActionModel, x_session_id: Optional[str] = Header(default=None)) -> StepResultModel:
    sid = x_session_id or DEFAULT_SESSION
    env = _require_env(sid)
    try:
        raw_obs, reward_val, done, _ = env.step(body.action)
    except (IndexError, KeyError):
        raise HTTPException(400, "Episode is done. Call /reset or /upload first.")
    return StepResultModel(
        observation=_obs_to_model(raw_obs),
        reward=RewardModel(value=float(reward_val), done=done, step=env.steps),
        info={},
    )


@app.get("/state")
def state(x_session_id: Optional[str] = Header(default=None)) -> EnvStateModel:
    sid  = x_session_id or DEFAULT_SESSION
    env  = _require_env(sid)
    done = len(env.issues) == 0 or env.steps >= env.max_steps
    return _build_state(env, done=done)


@app.post("/run")
def run(body: RunRequest, x_session_id: Optional[str] = Header(default=None)) -> EnvStateModel:
    """Run a full episode with the baseline agent and return the final state."""
    from agent import baseline_agent
    sid = x_session_id or DEFAULT_SESSION
    env = _get_env(sid)
    obs = env.reset(task_level=body.task_level)
    done = False
    while not done:
        obs, _, done, _ = env.step(baseline_agent(obs))
    return _build_state(env, done=True)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
