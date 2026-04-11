"""
train.py — Q-Learning agent trainer for DataCleaningEnv
=========================================================
Trains a tabular Q-learning policy on all three difficulty levels
(easy / medium / hard) and saves the learned policy to policy.pkl.

The trained policy is loaded by inference.py at evaluation time.

Observation features (what the agent actually sees — NO issue labels):
    For each column in the row:
        • is_null      — cell is NaN / None
        • is_string_in_numeric_col — non-parseable string in a numeric column
        • has_whitespace — string with leading/trailing spaces
        • numeric_value — z-score of the numeric value (0 if non-numeric)

The key design choice: the observation NEVER reveals the issue type.
The agent must learn to map raw cell signals → correct action.

Training loop:
    - 3,000 episodes per difficulty level (9,000 total)
    - Epsilon-greedy exploration: ε decays from 1.0 → 0.05
    - Learning rate α = 0.1, discount γ = 0.95
    - Policy is saved as {state_key: best_action} dict

Usage:
    python train.py                  # train and save policy.pkl
    python train.py --episodes 5000  # custom episode count per level
"""

from __future__ import annotations

import argparse
import math
import os
import pickle
import sys
from collections import defaultdict
from typing import Dict, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Import local env
# ---------------------------------------------------------------------------
from env import DataCleaningEnv, EXPECTED_NUMERIC

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
ACTIONS        = [0, 1, 3]          # skip, impute_missing, fix_outlier
ALPHA          = 0.1                # learning rate
GAMMA          = 0.95               # discount factor
EPSILON_START  = 1.0
EPSILON_END    = 0.05
POLICY_PATH    = "policy.pkl"
TASK_LEVELS    = ["easy", "medium", "hard"]


# ---------------------------------------------------------------------------
# Feature extraction — raw row values → discrete state key
# NO issue type is revealed. Agent must infer from raw signals.
# ---------------------------------------------------------------------------
def _obs_to_state_key(obs: dict) -> Tuple:
    """
    Convert a raw observation into a hashable state tuple.

    Features per cell (ordered by column name for reproducibility):
        (col_name, is_null, is_str_in_numeric, has_whitespace, value_bucket)

    value_bucket: bucketed z-score of numeric value so the Q-table stays finite.
        -2 = very negative (likely outlier/negative)
        -1 = below mean
         0 = near mean / non-numeric
        +1 = above mean
        +2 = very high (likely outlier)
    """
    if obs is None:
        return ("__terminal__",)

    row: dict = obs["row_data"]

    # compute per-column stats inline (cheap for 10 columns)
    numeric_vals = {
        k: float(v) for k, v in row.items()
        if k in EXPECTED_NUMERIC
        and isinstance(v, (int, float))
        and not (isinstance(v, float) and math.isnan(v))
    }

    # simple mean/std for z-scoring
    if numeric_vals:
        vals  = list(numeric_vals.values())
        mean  = sum(vals) / len(vals)
        std   = (sum((x - mean) ** 2 for x in vals) / len(vals)) ** 0.5 or 1.0
    else:
        mean, std = 0.0, 1.0

    features = []
    for col in sorted(row.keys()):          # sorted → deterministic order
        val = row[col]

        is_null = val is None or (isinstance(val, float) and math.isnan(val))

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


def _parseable_as_float(s: str) -> bool:
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ---------------------------------------------------------------------------
# Epsilon-greedy action selection
# ---------------------------------------------------------------------------
def _choose_action(
    q_table: Dict[Tuple, Dict[int, float]],
    state:   Tuple,
    epsilon: float,
    rng:     np.random.Generator,
) -> int:
    if rng.random() < epsilon:
        return ACTIONS[int(rng.integers(0, len(ACTIONS)))]
    q_vals = q_table[state]
    if not q_vals:
        return ACTIONS[int(rng.integers(0, len(ACTIONS)))]
    return max(ACTIONS, key=lambda a: q_vals.get(a, 0.0))


# ---------------------------------------------------------------------------
# Q-table update (standard Bellman)
# ---------------------------------------------------------------------------
def _update_q(
    q_table:      Dict[Tuple, Dict[int, float]],
    state:        Tuple,
    action:       int,
    reward:       float,
    next_state:   Tuple,
    done:         bool,
) -> None:
    current_q = q_table[state].get(action, 0.0)
    if done or next_state == ("__terminal__",):
        target = reward
    else:
        best_next = max(q_table[next_state].values()) if q_table[next_state] else 0.0
        target    = reward + GAMMA * best_next
    q_table[state][action] = current_q + ALPHA * (target - current_q)


# ---------------------------------------------------------------------------
# Run one training episode
# ---------------------------------------------------------------------------
def _train_episode(
    env:     DataCleaningEnv,
    task:    str,
    q_table: Dict[Tuple, Dict[int, float]],
    epsilon: float,
    rng:     np.random.Generator,
) -> float:
    obs      = env.reset(task_level=task)
    state    = _obs_to_state_key(obs)
    total_r  = 0.0
    done     = False

    while not done:
        action              = _choose_action(q_table, state, epsilon, rng)
        obs, reward, done, _= env.step(action)
        next_state          = _obs_to_state_key(obs)
        total_r            += float(reward)
        _update_q(q_table, state, action, float(reward), next_state, done)
        state = next_state

    return total_r


# ---------------------------------------------------------------------------
# Extract greedy policy from Q-table
# ---------------------------------------------------------------------------
def _extract_policy(q_table: Dict[Tuple, Dict[int, float]]) -> Dict[Tuple, int]:
    policy = {}
    for state, q_vals in q_table.items():
        if q_vals:
            policy[state] = max(ACTIONS, key=lambda a: q_vals.get(a, 0.0))
    return policy


# ---------------------------------------------------------------------------
# Evaluate the greedy policy (no exploration)
# ---------------------------------------------------------------------------
def _evaluate(
    env:    DataCleaningEnv,
    policy: Dict[Tuple, int],
    task:   str,
    n_eval: int = 10,
) -> float:
    scores = []
    for _ in range(n_eval):
        obs  = env.reset(task_level=task)
        done = False
        while not done:
            state  = _obs_to_state_key(obs)
            action = policy.get(state, 1)   # default to impute_missing if unseen state
            obs, _, done, _ = env.step(action)
        raw_state = env.state()
        scores.append(float(raw_state.get("score", 0.0)))
    return sum(scores) / len(scores)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------
def train(episodes_per_level: int = 3000) -> None:
    # Each level gets its own full epsilon decay so easy isn't stuck at high noise.
    # Easy gets 1.5x episodes — it's the foundation the other levels build on.
    eps_per_task = {
        "easy":   max(episodes_per_level, int(episodes_per_level * 1.5)),
        "medium": episodes_per_level,
        "hard":   episodes_per_level,
    }
    total_eps = sum(eps_per_task.values())

    print("=" * 60)
    print("DataCleaningEnv — Q-Learning Training")
    print(f"Episodes easy/medium/hard  : "
          f"{eps_per_task['easy']}/{eps_per_task['medium']}/{eps_per_task['hard']}")
    print(f"Total episodes     : {total_eps}")
    print(f"Actions            : {ACTIONS}")
    print(f"Alpha / Gamma      : {ALPHA} / {GAMMA}")
    print(f"Epsilon per level  : {EPSILON_START} -> {EPSILON_END} (independent decay)")
    print("=" * 60)

    rng     = np.random.default_rng(0)
    env     = DataCleaningEnv()
    q_table: Dict[Tuple, Dict[int, float]] = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    for task in TASK_LEVELS:
        n_eps = eps_per_task[task]
        print(f"\n-- Training on '{task}' ({n_eps} episodes) --------------------")
        rewards_window = []

        for ep in range(n_eps):
            # Each level gets its OWN full 1.0 -> 0.05 epsilon decay.
            # Q-table is shared so knowledge from easy carries into medium/hard.
            epsilon = EPSILON_START - (EPSILON_START - EPSILON_END) * (ep / max(1, n_eps - 1))

            r = _train_episode(env, task, q_table, epsilon, rng)
            rewards_window.append(r)

            if (ep + 1) % 500 == 0 or ep == 0:
                avg = sum(rewards_window[-100:]) / min(100, len(rewards_window))
                print(f"  ep {ep+1:>5}/{n_eps}  "
                      f"eps={epsilon:.3f}  avg_reward(100)={avg:+.1f}  "
                      f"Q-states={len(q_table)}")

    print("\n── Extracting greedy policy ─────────────────────────────")
    policy = _extract_policy(q_table)
    print(f"  Policy size: {len(policy)} states")

    print("\n── Evaluating policy (10 episodes per level) ────────────")
    for task in TASK_LEVELS:
        avg_score = _evaluate(env, policy, task, n_eval=10)
        print(f"  {task:>6}  avg score = {avg_score:.4f}")

    print(f"\n── Saving policy → {POLICY_PATH} ──────────────────────")
    # Save both the full Q-table (for further training) and the greedy policy
    artifact = {
        "policy":           policy,
        "q_table":          dict(q_table),
        "actions":          ACTIONS,
        "episodes_trained": total_eps,
        "alpha":            ALPHA,
        "gamma":            GAMMA,
    }
    with open(POLICY_PATH, "wb") as f:
        pickle.dump(artifact, f)
    size_kb = os.path.getsize(POLICY_PATH) / 1024
    print(f"  Saved ({size_kb:.1f} KB)")
    print("\nTraining complete. Run inference.py to evaluate.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Q-learning agent on DataCleaningEnv")
    parser.add_argument(
        "--episodes",
        type=int,
        default=3000,
        help="Number of training episodes per difficulty level (default: 3000)",
    )
    args = parser.parse_args()
    train(episodes_per_level=args.episodes)
