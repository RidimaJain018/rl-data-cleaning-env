from __future__ import annotations

from typing import Dict, List, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Primitive value type used across row data
# ---------------------------------------------------------------------------
CellValue = Optional[Union[float, int, str]]


# ---------------------------------------------------------------------------
# Observation
# ---------------------------------------------------------------------------
class ColumnStats(BaseModel):
    mean: float = Field(..., description="Column mean (computed on clean values)")
    std:  float = Field(..., description="Column standard deviation")


class ObservationModel(BaseModel):
    row_data: Dict[str, CellValue] = Field(
        ...,
        description="Raw cell values for the row being inspected. "
                    "The issue type is NOT revealed — the agent must infer it "
                    "from raw values and column_stats.",
    )
    column_stats: Optional[Dict[str, ColumnStats]] = Field(
        None,
        description="Per-column mean and std for numeric columns. "
                    "Allows the agent to detect z-score outliers without "
                    "being told the issue type label.",
    )
    step_progress: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of max_steps consumed so far (0.0-1.0).",
    )
    issues_remaining: Optional[int] = Field(
        None,
        ge=0,
        description="Number of data quality issues still unfixed in this episode.",
    )

    @field_validator("row_data")
    @classmethod
    def row_data_not_empty(cls, v: Dict[str, CellValue]) -> Dict[str, CellValue]:
        if not v:
            raise ValueError("row_data must not be empty.")
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "row_data": {
                    "age": None,
                    "salary": 50000,
                    "city": "NY",
                    "experience": 2,
                    "rating": 4.5,
                },
                "column_stats": {
                    "salary": {"mean": 87000.0, "std": 22000.0},
                    "age":    {"mean": 38.5,    "std": 9.2},
                },
                "step_progress":    0.1,
                "issues_remaining": 5,
            }
        }


# ---------------------------------------------------------------------------
# Action
#   0 → skip
#   1 → impute_missing
#   3 → fix_outlier   (handles outliers, invalid values, duplicates,
#                       type mismatches, and whitespace padding)
# Note: action 2 is intentionally reserved for future use.
# ---------------------------------------------------------------------------
VALID_ACTIONS: Dict[int, str] = {
    0: "skip",
    1: "impute_missing",
    3: "fix_outlier",
}


class ActionModel(BaseModel):
    action: int = Field(
        ...,
        description="Integer action ID. "
                    "0 = skip, 1 = impute_missing, 3 = fix_outlier.",
    )

    @field_validator("action")
    @classmethod
    def action_must_be_valid(cls, v: int) -> int:
        if v not in VALID_ACTIONS:
            raise ValueError(
                f"Invalid action {v!r}. "
                f"Must be one of {list(VALID_ACTIONS.keys())} "
                f"({', '.join(f'{k}={lbl}' for k, lbl in VALID_ACTIONS.items())})."
            )
        return v

    @property
    def label(self) -> str:
        return VALID_ACTIONS[self.action]

    class Config:
        json_schema_extra = {"example": {"action": 1}}


# ---------------------------------------------------------------------------
# Reward
# ---------------------------------------------------------------------------
class RewardModel(BaseModel):
    value: float = Field(
        ...,
        description="Reward signal for this transition. "
                    "+2 correct fix, -1 wrong action, "
                    "+5 terminal bonus on completion, -5 on timeout.",
    )
    done: bool = Field(
        ...,
        description="True if the episode has ended.",
    )
    step: int = Field(..., ge=0, description="Current step count within the episode.")

    @field_validator("step")
    @classmethod
    def step_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError("step must be >= 0.")
        return v

    class Config:
        json_schema_extra = {
            "example": {"value": 2.0, "done": False, "step": 3}
        }


# ---------------------------------------------------------------------------
# Step result
# ---------------------------------------------------------------------------
class StepResultModel(BaseModel):
    observation: Optional[ObservationModel] = Field(
        None,
        description="Next observation. None when the episode is terminal.",
    )
    reward: RewardModel = Field(..., description="Reward, done flag, and step count.")
    info: Dict[str, CellValue] = Field(
        default_factory=dict,
        description="Optional extra info dict (reserved for future use).",
    )

    class Config:
        json_schema_extra = {
            "example": {
                "observation": {
                    "row_data": {
                        "age": 25,
                        "salary": 1000000,
                        "city": "LA",
                        "experience": 3,
                        "rating": 4.2,
                    }
                },
                "reward": {"value": 2.0, "done": False, "step": 1},
                "info": {},
            }
        }


# ---------------------------------------------------------------------------
# Episode log entry
# ---------------------------------------------------------------------------

# All issue types the environment can produce
_VALID_ISSUE_TYPES = {
    "missing",
    "outlier",
    "invalid_rating",
    "invalid_negative",
    "duplicate",          # FIX: added — exact duplicate row
    "type_mismatch",      # FIX: added — string value in expected-numeric column
    "whitespace_padding", # FIX: added — leading/trailing whitespace in string column
}


class EpisodeLogEntryModel(BaseModel):
    step: int = Field(..., ge=1, description="Step number (1-indexed).")
    row: int = Field(..., ge=0, description="DataFrame row index of the affected cell.")
    col: str = Field(..., description="Column name of the affected cell.")
    issue: str = Field(..., description="Issue type detected for this step.")
    action: str = Field(..., description="Human-readable action label.")
    correct: bool = Field(..., description="Whether the agent's action was correct.")
    old_value: str = Field(..., description="Cell value before the fix (string repr).")
    new_value: str = Field(..., description="Cell value after the fix, or '—' if skipped/wrong.")
    reward: float = Field(..., description="Reward received for this step.")

    @field_validator("issue")
    @classmethod
    def issue_must_be_known(cls, v: str) -> str:
        if v not in _VALID_ISSUE_TYPES:
            raise ValueError(
                f"Unknown issue type {v!r}. Must be one of {_VALID_ISSUE_TYPES}."
            )
        return v

    class Config:
        json_schema_extra = {
            "example": {
                "step": 1,
                "row": 0,
                "col": "age",
                "issue": "missing",
                "action": "impute_missing",
                "correct": True,
                "old_value": "nan",
                "new_value": "28.5",
                "reward": 2,
            }
        }


# ---------------------------------------------------------------------------
# Environment state
# ---------------------------------------------------------------------------
class EnvStateModel(BaseModel):
    task_level: Optional[str] = Field(
        None,
        description="Active task level: 'easy', 'medium', or 'hard'.",
    )
    current_step: int = Field(..., ge=0, description="Steps taken so far.")
    max_steps: int = Field(..., ge=1, description="Maximum steps allowed per episode.")
    total_issues_at_start: int = Field(..., ge=0, description="Total issues at episode start.")
    remaining_issues: int = Field(..., ge=0, description="Issues not yet fixed.")
    score: float = Field(..., ge=0.0, le=1.0, description="Partial-credit score (fixed / total).")
    done: bool = Field(..., description="True if the episode has ended.")
    current_observation: Optional[ObservationModel] = Field(None)
    episode_log: List[EpisodeLogEntryModel] = Field(default_factory=list)

    @model_validator(mode="after")
    def remaining_cannot_exceed_total(self) -> EnvStateModel:
        if self.remaining_issues > self.total_issues_at_start:
            raise ValueError(
                "remaining_issues cannot exceed total_issues_at_start."
            )
        return self

    class Config:
        json_schema_extra = {
            "example": {
                "task_level": "medium",
                "current_step": 2,
                "max_steps": 25,
                "total_issues_at_start": 5,
                "remaining_issues": 3,
                "score": 0.4,
                "done": False,
                "current_observation": {
                    "row_data": {
                        "age": 30,
                        "salary": 60000,
                        "city": None,
                        "experience": 4,
                        "rating": 4.2,
                    }
                },
                "episode_log": [],
            }
        }
