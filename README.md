---
title: RL Data Cleaning Agent
emoji: 🧹
colorFrom: purple
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# RL Data Cleaning Agent

> **Meta × Scaler OpenEnv Hackathon** — OpenEnv-compliant reinforcement-learning environment for tabular data cleaning.

---

## Overview

Real-world datasets are almost never clean. Data scientists spend an estimated **60–80% of their time** handling missing values, outliers, type mismatches, and other quality issues — before any modelling can begin.

This project frames **data cleaning as a sequential decision-making problem**: an RL agent inspects one dirty row at a time and chooses the best remediation action. Three difficulty levels expose progressively harder data quality challenges, from simple missing-value imputation all the way to handling duplicates, type mismatches, and whitespace padding.

The environment is fully compliant with the **OpenEnv 3-method interface** (`reset` / `step` / `state`) and is deployable as a Hugging Face Space with a Docker runtime.

The choices involved in data cleaning — *should I impute this missing value or is this an outlier?* — are **sequential and context-dependent**, making RL a natural fit:

- **State**: the current dirty row being inspected
- **Action**: skip / impute_missing / fix_outlier
- **Reward**: +2 per correct fix, −1 for wrong actions, +5 completion bonus
- **Goal**: clean the entire dataset with maximum accuracy and minimum wasted steps

---

## Action Space

The agent chooses one discrete action per step:

| Action ID | Label | Description |
|:---------:|-------|-------------|
| `0` | `skip` | Do nothing; move to the next problematic row |
| `1` | `impute_missing` | Fill a `NaN` / `None` cell with the column **mean** (numeric) or **mode** (categorical) |
| `3` | `fix_outlier` | Replace an outlier, invalid, or corrupt value with the column **mean** |

> **Correct action rule**: action `1` is correct for `missing` issues; action `3` is correct for all other issue types (outlier, invalid rating, invalid negative, duplicate, type mismatch, whitespace padding).

> **Note:** Action `2` is intentionally absent — reserved for a future `flag_for_review` action.

---

## Observation Space

Each observation is a single row from the dirty 10-column employee DataFrame:

```json
{
  "row_data": {
    "age":               25,
    "salary":            85000,
    "city":              "Chicago",
    "experience":        5,
    "rating":            4.2,
    "department":        "Engineering",
    "bonus":             8000,
    "years_at_company":  3,
    "performance_score": 4.5,
    "overtime_hours":    12
  }
}
```

| Field | Type | Valid range | Issues |
|-------|------|-------------|--------|
| `age` | `float \| string \| null` | 22–62 | `null` = missing; negative = invalid_negative; `"N/A"` = type_mismatch |
| `salary` | `float` | 35k–180k | > 300,000 = outlier |
| `city` | `string \| null` | 8 US cities | `null` = missing; `" NY "` = whitespace_padding |
| `experience` | `float \| null` | 0–30 | `null` = missing; negative = invalid_negative |
| `rating` | `float \| null` | 2.5–4.9 | `null` = missing; > 5 = invalid_rating |
| `department` | `string \| null` | 6 departments | `null` = missing; `"  HR  "` = whitespace_padding |
| `bonus` | `float \| null` | 0–25k | `null` = missing; > 80,000 = outlier |
| `years_at_company` | `float \| string \| null` | 0–20 | `null` = missing; negative = invalid_negative; `"ten"` = type_mismatch |
| `performance_score` | `float \| null` | 2.5–4.9 | `null` = missing; > 5 = invalid_rating |
| `overtime_hours` | `float \| null` | 0–80 | `null` = missing; negative = invalid_negative |

---

## Issue Types

| Issue | Description | Correct action |
|-------|-------------|---------------|
| `missing` | Cell is `NaN` / `null` | `1` (impute_missing) |
| `outlier` | `salary > 300,000` or `bonus > 80,000` | `3` (fix_outlier) |
| `invalid_rating` | `rating` or `performance_score > 5` | `3` (fix_outlier) |
| `invalid_negative` | Negative value in a non-negative numeric column | `3` (fix_outlier) |
| `duplicate` | Exact copy of a previously seen row | `3` (fix_outlier — drops row) |
| `type_mismatch` | Non-parseable string in a numeric column (e.g. `"N/A"`, `"ten"`) | `3` (fix_outlier — coerces and imputes) |
| `whitespace_padding` | Leading/trailing spaces in a string column | `3` (fix_outlier — strips) |

### Dirty → Clean Examples

| Issue Type | Dirty value | Cleaned value | How |
|---|---|---|---|
| `missing` | `age: null` | `age: 34.5` | Column mean |
| `outlier` | `salary: 1,500,000` | `salary: 87,423` | Column mean |
| `invalid_rating` | `performance_score: 7.4` | `performance_score: 3.6` | Column mean |
| `invalid_negative` | `overtime_hours: -18` | `overtime_hours: 24.1` | Column mean |
| `duplicate` | identical copy of row 1 | row dropped | Drop |
| `type_mismatch` | `age: "N/A"` | `age: 34.5` | Coerce → column mean |
| `whitespace_padding` | `city: " NY "` | `city: "NY"` | Strip |

---

## Task Descriptions

### 🟢 Easy (20 rows, 8 issues)
8 missing values spread across 8 different columns: age, city, experience, rating, department, bonus, years_at_company, performance_score. Every correct action is `1` (impute_missing). Seed: 42.

### 🟡 Medium (30 rows, 12 issues)
5 missing values + 3 salary/bonus outliers + 2 invalid rating scores (> 5) + 1 invalid negative (overtime_hours) + 1 whitespace-padded city. The agent must distinguish imputation (action `1`) from outlier correction (action `3`). Seed: 123.

### 🔴 Hard (50 rows, 20 issues)
All 7 issue types across 10 columns: 6 missing, 4 outliers (salary/bonus), 2 invalid negatives, 2 duplicate rows, 2 type mismatches (`age="N/A"`, `years_at_company="ten"`), 2 whitespace-padded strings, 2 invalid rating scores. Seed: 999.

---

## Reward Structure

| Event | Reward |
|-------|-------:|
| Correct fix | `+2` |
| Wrong action | `−1` |
| All issues fixed (completion bonus) | `+5` |
| Episode timeout (`max_steps = 100`) | `−5` |

**Score** (0.0–1.0): `fixed_issues / total_issues_at_start`

### Reward Design Rationale

The reward structure is designed to produce **dense, meaningful learning signals at every step** — not just terminal rewards.

- **+2 per correct fix**: Immediate per-step feedback so the agent learns which actions are right without waiting until episode end. This avoids the sparse-reward problem.
- **−1 for wrong action**: Small enough not to be catastrophic, large enough to discourage random guessing. The +2/−1 asymmetry rewards correct decisions more than it punishes mistakes, which encourages exploration.
- **+5 completion bonus**: Rewards thoroughness — an agent that fixes 19/20 issues scores meaningfully lower than one that fixes all 20. Creates a clear gradient toward complete cleaning.
- **−5 timeout penalty**: Prevents a degenerate policy of cycling through rows indefinitely without committing to fixes.
- **Partial-credit score**: Reported separately from reward as `fixed/total` — a normalised metric comparable across episodes with different issue counts, suitable for automated grading.

---

## Baseline Scores

Rule-based `baseline_agent` (deterministic, reproducible):

| Task | Score | Total Reward | Steps | Fixed |
|------|------:|-------------:|------:|-------|
| easy | 1.000 | 21 | 8 | 8/8 |
| medium | 1.000 | 29 | 12 | 12/12 |
| hard | 1.000 | 45 | 20 | 20/20 |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- Docker (for containerised deployment)
- `openenv` CLI — `pip install openenv` (optional, for validation)

### Install Python dependencies

```bash
pip install -r requirements.txt
```

---

## Running Locally

### Start the FastAPI server

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

### Verify endpoints

```bash
# Health check
curl http://localhost:8000/health

# Start a new episode (easy level)
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_level": "easy"}'

# Take an action (impute_missing)
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": 1}'

# Get full episode state
curl http://localhost:8000/state
```

### Launch the Gradio demo UI (optional)

```bash
python gradio_ui.py
```

The demo opens at `http://localhost:7860` and lets you run episodes visually, compare before/after cleaning, and upload your own CSV.

---

## Using the Python Client

### Synchronous (recommended for scripts)

```python
from client import DataCleaningClient

with DataCleaningClient("http://localhost:8000").sync() as client:
    obs = client.reset(task_level="medium")
    print("First observation:", obs.row_data)

    done = False
    while not done:
        result = client.step(action=1)
        done = result.reward.done

    state = client.state()
    print(f"Score: {state.score}, Steps: {state.current_step}")
```

### Asynchronous

```python
import asyncio
from client import DataCleaningClient

async def main():
    async with DataCleaningClient("http://localhost:8000") as client:
        obs = await client.reset(task_level="hard")
        result = await client.step(action=3)
        state  = await client.state()
        print(f"Score: {state.score}")

asyncio.run(main())
```

---

## Running Inference

### Baseline agent (no API key needed)

```bash
python inference.py
# or explicitly:
python inference.py --agent baseline
```

### LLM agent

```bash
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="hf_your_token_here"

python inference.py --agent llm
```

### Both agents — side-by-side comparison

```bash
python inference.py --agent both
```

### Verbose mode — step-by-step decision trace

```bash
python inference.py --verbose
# Shows each step: column · issue type · action taken · correct? · reward
```

### Single task level

```bash
python inference.py --task hard
```

If the LLM call fails (bad key, network error), the agent automatically falls back to `baseline_agent` — the script never crashes.

### Live Space — quick test

```bash
curl -X POST https://Ridi2007-rl-data-cleaning-env.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_level": "medium"}'

curl https://Ridi2007-rl-data-cleaning-env.hf.space/health
```

---

## Docker

### Build the image

```bash
docker build -t rl-data-cleaning-env .
```

### Run the container

```bash
docker run -p 8000:8000 rl-data-cleaning-env
```

### Test the running container

```bash
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_level": "medium"}'
```

---

## Deploying to Hugging Face Spaces

1. **Create a new Space** at [huggingface.co/spaces](https://huggingface.co/spaces):
   - Runtime: **Docker**
   - Visibility: **Public**

2. **Ensure the README front matter** is present (already included at the top of this file):
   ```yaml
   ---
   sdk: docker
   app_port: 8000
   ---
   ```
   This tells Hugging Face to route traffic to port 8000. Without `app_port: 8000` the automated ping will get a connection refused.

3. **Push this repository** to the Space:
   ```bash
   git remote add space https://huggingface.co/spaces/<your-username>/<space-name>
   git push space main
   ```

4. **Confirm the endpoint is live**:
   ```bash
   curl -X POST https://<your-username>-<space-name>.hf.space/reset \
        -H "Content-Type: application/json" \
        -d '{"task_level": "easy"}'
   ```
   Expect HTTP 200 and a JSON body with `"observation"`.

---

## OpenEnv Validation

```bash
openenv validate
```

---

## Pre-submission Validation Script

```bash
chmod +x validate-submission.sh

# Run all checks (Space must be deployed first):
./validate-submission.sh https://<your-space>.hf.space .
```

---

## Project Structure

```
hackathon/
├── app.py                  # FastAPI OpenEnv server
├── client.py               # Typed HTTP client
├── models.py               # Pydantic models
├── env.py                  # DataCleaningEnv — environment logic
├── agent.py                # Baseline & upload agents
├── inference.py            # Baseline & LLM inference runners
├── gradio_ui.py            # Optional Gradio demo UI
├── openenv.yaml            # OpenEnv specification file
├── Dockerfile              # Container definition (serves FastAPI on port 8000)
├── requirements.txt        # Python dependencies
├── README.md               # This file
└── validate-submission.sh  # Pre-submission validator
```

---

## Environment Variables Reference

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API base URL | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | LLM model identifier | `meta-llama/Llama-3.2-3B-Instruct` |
| `HF_TOKEN` | Hugging Face API token (used as API key) | — |
| `OPENAI_API_KEY` | Alias for `HF_TOKEN` | — |
| `AGENT_TYPE` | `"baseline"` or `"llm"` | `"baseline"` |

---

## License

MIT
