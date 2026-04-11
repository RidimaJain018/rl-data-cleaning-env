---
title: RL Data Cleaning Agent
colorFrom: purple
colorTo: green
sdk: docker
app_port: 8000
pinned: false
---

# RL Data Cleaning Agent

> **Meta × Scaler OpenEnv Hackathon** — Multi-domain reinforcement-learning environment for tabular data cleaning across HR, Finance, Medical, and Ecommerce datasets.

---

## Overview

Real-world datasets are almost never clean. Data scientists spend an estimated **60–80% of their time** handling missing values, outliers, type mismatches, and other quality issues — before any modelling can begin.

This project frames **data cleaning as a sequential decision-making problem**: an RL agent inspects one dirty row at a time and chooses the best remediation action. Unlike single-schema environments, this environment rotates across **four real-world domains** per episode — the agent must generalise its cleaning policy without ever being told which domain it is operating on.

The environment is fully compliant with the **OpenEnv 3-method interface** (`reset` / `step` / `state`) and deploys to Hugging Face Spaces via Docker.

---

## Four Real-World Domains

| Domain | Columns | Example Issues |
|--------|---------|----------------|
| **HR** | age, salary, department, experience, performance_score, bonus, overtime_hours... | salary spike to $800k, age=None, overtime=-5 |
| **Finance** | amount, balance, credit_limit, transaction_fee, merchant, category... | transaction amount of $50,000 on a $200 avg, fee=-$10 |
| **Medical** | systolic_bp, glucose_level, bmi, dosage_mg, heart_rate, lab_value... | glucose=2400 (coma level), blood pressure=-40, BMI=0.2 |
| **Ecommerce** | price, quantity, discount_pct, shipping_days, review_score, return_rate... | discount=150%, review=9.0, price=-$5 |

**Difficulty controls which domains are active:**
- `easy` → HR only (single domain, 40 rows, 12 issues)
- `medium` → HR + Finance (two domains, 80 rows, 20 issues)
- `hard` → All four domains (120 rows, 30 issues, all 7 issue types)

The agent sees raw cell values and per-column statistics — **never** the domain name or issue type label. It must learn to map z-score spikes, null signals, and string anomalies to the correct action purely from reward feedback.

---

## Why RL over Rule-Based Cleaning?

| Scenario | Rule-Based Agent | RL Agent |
|----------|-----------------|----------|
| Known schema, fixed thresholds | ✅ Works | ✅ Works |
| Unknown domain (Finance, Medical) | ❌ No rules to apply | ✅ Generalises via z-score signals |
| Ambiguous cells (numeric string vs type mismatch) | ❌ Brittle regex | ✅ Learns from reward signal |
| Changing outlier distributions over time | ❌ Hardcoded thresholds go stale | ✅ Can be retrained |
| Any uploaded CSV with unknown columns | ❌ Breaks immediately | ✅ IQR + z-score generalise |

The `/evaluate_upload` endpoint demonstrates this directly: drop in any CSV from any domain and the trained Q-policy evaluates it — no schema knowledge required.

---

## Action Space

| Action ID | Label | Description | Reward |
|:---------:|-------|-------------|--------|
| `0` | `skip` | Row is clean, move on | 0 |
| `1` | `impute_missing` | Fill NaN with column mean (numeric) or mode (categorical) | +2 if correct |
| `2` | `flag_for_review` | Mark ambiguous row without fixing | -0.5 always |
| `3` | `fix_outlier` | Replace bad value with column mean; strip whitespace; drop duplicates | +2 if correct |

---

## Observation Space

Each observation exposes raw row values and column statistics — **never** the issue type label:

```json
{
  "row_data": {
    "patient_id":    "PAT12345",
    "age_years":     45,
    "systolic_bp":   -80,
    "glucose_level": null,
    "bmi":           24.3,
    "diagnosis":     "Hypertension"
  },
  "column_stats": {
    "systolic_bp":   {"mean": 112.4, "std": 14.2},
    "glucose_level": {"mean": 98.7,  "std": 18.1}
  },
  "step_progress":    0.12,
  "issues_remaining": 25
}
```

The agent must infer from `systolic_bp: -80` (negative, z-score = -13.5) that this is an `invalid_negative` requiring `fix_outlier`. The domain name "medical" is never revealed.

---

## Issue Types

| Issue | Description | Correct action |
|-------|-------------|---------------|
| `missing` | Cell is NaN / null | `1` (impute_missing) |
| `outlier` | Value far outside column distribution (IQR-based) | `3` (fix_outlier) |
| `invalid_negative` | Negative value in a non-negative numeric column | `3` (fix_outlier) |
| `invalid_range` | Value violates domain-specific bounds (e.g. discount > 100%) | `3` (fix_outlier) |
| `duplicate` | Exact copy of a previously seen row | `3` (fix_outlier — drops row) |
| `type_mismatch` | Non-parseable string in a numeric column (e.g. `"N/A"`, `"unknown"`) | `3` (fix_outlier) |
| `whitespace_padding` | Leading/trailing spaces in a string column | `3` (fix_outlier — strips) |

---

## Task Descriptions

### Easy (40 rows, 12 issues) — HR domain
Single domain. Missing values, salary/bonus outliers, invalid negatives across HR employee records. The agent learns the core skip/impute/fix distinction on a familiar schema. Seed: 123.

### Medium (80 rows, 20 issues) — HR + Finance domains
Two domains. The agent encounters financial transaction records (amount, balance, credit_limit, transaction_fee) for the first time — different column names, different value distributions. Adds type_mismatch and whitespace_padding. Seed: 999.

### Hard (120 rows, 30 issues) — All 4 domains
All four domains. Medical patient records and ecommerce orders join HR and Finance. All 7 issue types present. The agent cannot rely on column names — it must generalise purely from z-score signals and null/string flags. Seed: 1337.

---

## Reward Structure

| Event | Reward |
|-------|-------:|
| Correct fix | `+2` |
| Wrong action | `−1` |
| Flag for review | `−0.5` |
| All issues fixed (completion bonus) | `+5` |
| Episode timeout | `−5` |

**Score** (0.0–1.0): `fixed_issues / total_issues_at_start`

### Reward Design Rationale

- **+2 per correct fix**: Dense per-step signal — the agent gets immediate feedback without waiting for episode end. Avoids the sparse-reward problem.
- **−1 for wrong action**: Small enough not to be catastrophic, large enough to discourage guessing. The +2/−1 asymmetry rewards correct decisions more than it punishes mistakes.
- **−0.5 for flag_for_review**: Lighter than wrong_action — the agent is rewarded for flagging genuinely ambiguous rows rather than guessing wrong, but is incentivised to fix what it can identify.
- **+5 completion bonus**: Rewards thoroughness — fixing 29/30 scores meaningfully lower than 30/30.
- **−5 timeout penalty**: Prevents cycling through rows without committing to fixes.

---

## Trained Q-Policy

The environment ships with a trained Q-learning policy (`policy.pkl`) that is loaded automatically by `inference.py`.

**Training:** 10,500 episodes across all three difficulty levels (4,500 easy + 3,000 medium + 3,000 hard). Each level gets an independent ε decay from 1.0 → 0.05. The Q-table is shared across levels so knowledge from easy carries into hard.

**Policy design:** State keys are built from `(column_name, is_null, is_str_in_numeric, has_whitespace, z_score_bucket)` per column — **not** from column names alone. This makes the policy domain-agnostic: the same z-score spike feature triggers `fix_outlier` whether the column is `salary`, `glucose_level`, or `amount`.

**Evaluation scores:**

| Task | Avg Score | Steps |
|------|----------:|------:|
| easy | 0.9900 | 12 |
| medium | 0.9900 | 20 |
| hard | 0.9655 | 30 |

---

## Setup & Installation

```bash
pip install -r requirements.txt
```

---

## Running Locally

```bash
# Start FastAPI server
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# Health check
curl http://localhost:8000/health

# Start hard episode (all 4 domains)
curl -X POST http://localhost:8000/reset \
     -H "Content-Type: application/json" \
     -d '{"task_level": "hard"}'

# Take an action
curl -X POST http://localhost:8000/step \
     -H "Content-Type: application/json" \
     -d '{"action": 3}'

# Get state
curl http://localhost:8000/state
```

```bash
# Launch Gradio UI (optional)
python gradio_ui.py  # → http://localhost:7860
```

---

## Training the Agent

```bash
python train.py --episodes 3000
# Trains on easy/medium/hard across all 4 domains
# Saves policy.pkl (~11 KB)
```

---

## Running Inference

```bash
# Q-policy agent (loaded from policy.pkl automatically)
python inference.py

# LLM agent
export API_BASE_URL="https://api-inference.huggingface.co/v1"
export MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct"
export HF_TOKEN="hf_your_token_here"
python inference.py
```

---

## Live Space

```bash
curl https://Ridi2007-rl-data-cleaning-env.hf.space/health

curl -X POST https://Ridi2007-rl-data-cleaning-env.hf.space/reset \
     -H "Content-Type: application/json" \
     -d '{"task_level": "hard"}'
```

---

## Docker

```bash
docker build -t rl-data-cleaning-env .
docker run -p 8000:8000 rl-data-cleaning-env
```

---

## Project Structure

```
├── app.py          # FastAPI OpenEnv server
├── env.py          # Multi-domain DataCleaningEnv (HR/Finance/Medical/Ecommerce)
├── agent.py        # Domain-agnostic baseline agent (z-score + null signals)
├── train.py        # Q-learning trainer — saves policy.pkl
├── inference.py    # Q-policy → LLM → baseline agent runner
├── models.py       # Pydantic v2 typed models
├── client.py       # Typed HTTP client (sync + async)
├── gradio_ui.py    # Gradio demo UI
├── openenv.yaml    # OpenEnv specification
├── policy.pkl      # Trained Q-policy (10500 episodes)
├── Dockerfile      # FastAPI on 8000, Gradio on 7860
└── start.sh        # Production entrypoint
```

---

## Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `API_BASE_URL` | LLM API base URL | `https://api-inference.huggingface.co/v1` |
| `MODEL_NAME` | LLM model identifier | `meta-llama/Llama-3.2-3B-Instruct` |
| `HF_TOKEN` | Hugging Face API token | — |

---

## License

MIT
