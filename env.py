"""
env.py — Multi-Domain DataCleaningEnv
======================================
Four real-world domains rotate each episode:
    HR          — employee records (salary, performance, tenure)
    Finance     — transaction records (amounts, accounts, merchants)
    Medical     — patient records (vitals, lab values, diagnoses)
    Ecommerce   — order records (prices, quantities, ratings, shipping)

The RL agent sees raw row values + per-column statistics.
It never sees the domain name or the issue type label.
It must learn to map (null signals, z-score spikes, string anomalies) → action.

Action space (same across all domains):
    0 = skip            — row is clean, move on
    1 = impute_missing  — fill NaN with column mean/mode
    2 = flag_for_review — mark ambiguous row (-0.5 reward, doesn't fix)
    3 = fix_outlier     — replace bad value with column mean/mode

Difficulty controls how many domains are active and how many issue types appear:
    easy   — single domain per episode, 3 issue types (missing, outlier, invalid_negative)
    medium — two domains mixed, 5 issue types (adds type_mismatch, whitespace)
    hard   — all four domains mixed, all 7 issue types, higher issue density
"""

from __future__ import annotations

import math
import random
from typing import Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Preserved for backward compatibility with agent.py / inference.py imports
# ---------------------------------------------------------------------------
EXPECTED_NUMERIC = {
    "age", "salary", "experience", "rating",
    "bonus", "years_at_company", "performance_score", "overtime_hours",
    "amount", "balance", "credit_limit", "transaction_fee",
    "systolic_bp", "diastolic_bp", "heart_rate", "glucose_level",
    "bmi", "age_years", "lab_value", "dosage_mg",
    "price", "quantity", "discount_pct", "shipping_days",
    "review_score", "return_rate", "units_sold",
}

OUTLIER_THRESHOLDS: dict = {}   # domain-agnostic: use IQR instead
SCORE_COLS = {"rating", "review_score", "performance_score"}

_ACTION_LABELS = {0: "skip", 1: "impute_missing", 2: "flag_for_review", 3: "fix_outlier"}
_SEEDS         = {"easy": 123, "medium": 999, "hard": 1337}

# ---------------------------------------------------------------------------
# Domain definitions — each domain is a callable that returns a clean row dict
# ---------------------------------------------------------------------------

_HR_CITIES       = ["NY", "LA", "SF", "Chicago", "Austin", "Seattle", "Boston", "Denver"]
_HR_DEPTS        = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]
_HR_TITLES       = ["Analyst", "Senior Analyst", "Manager", "Director", "VP", "Individual Contributor"]

_FIN_MERCHANTS   = ["Amazon", "Walmart", "Target", "Starbucks", "Shell", "Delta Airlines",
                    "Netflix", "Apple Store", "Uber", "Costco"]
_FIN_CATEGORIES  = ["Retail", "Food & Drink", "Travel", "Entertainment", "Utilities",
                    "Healthcare", "Groceries", "Transport"]
_FIN_STATUS      = ["completed", "pending", "refunded", "disputed"]

_MED_DIAGNOSES   = ["Hypertension", "Type 2 Diabetes", "Asthma", "Hypothyroidism",
                    "Migraine", "Anxiety Disorder", "GERD", "Osteoarthritis"]
_MED_MEDS        = ["Metformin", "Lisinopril", "Atorvastatin", "Levothyroxine",
                    "Amlodipine", "Omeprazole", "Sertraline", "Albuterol"]
_MED_BLOOD_TYPES = ["A+", "A-", "B+", "B-", "O+", "O-", "AB+", "AB-"]

_ECO_CATEGORIES  = ["Electronics", "Clothing", "Home & Garden", "Sports", "Books",
                    "Toys", "Beauty", "Automotive"]
_ECO_CARRIERS    = ["FedEx", "UPS", "USPS", "DHL", "Amazon Logistics"]
_ECO_STATUS      = ["delivered", "shipped", "processing", "returned", "cancelled"]


def _hr_row(rng: np.random.Generator) -> dict:
    """Realistic HR / employee record."""
    exp = int(rng.integers(0, 30))
    return {
        "employee_id":       f"EMP{int(rng.integers(1000, 9999))}",
        "age":               int(rng.integers(22, 62)),
        "salary":            int(rng.integers(35_000, 180_000)),
        "department":        _HR_DEPTS[int(rng.integers(0, len(_HR_DEPTS)))],
        "job_title":         _HR_TITLES[int(rng.integers(0, len(_HR_TITLES)))],
        "city":              _HR_CITIES[int(rng.integers(0, len(_HR_CITIES)))],
        "experience":        exp,
        "years_at_company":  int(rng.integers(0, min(exp + 1, 20))),
        "performance_score": round(float(rng.uniform(2.5, 4.9)), 1),
        "bonus":             int(rng.integers(0, 25_000)),
        "overtime_hours":    int(rng.integers(0, 80)),
        "rating":            round(float(rng.uniform(2.5, 4.9)), 1),
    }


def _finance_row(rng: np.random.Generator) -> dict:
    """Realistic financial transaction record."""
    amount = round(float(rng.uniform(1.0, 5_000.0)), 2)
    return {
        "transaction_id":  f"TXN{int(rng.integers(100_000, 999_999))}",
        "account_id":      f"ACC{int(rng.integers(10_000, 99_999))}",
        "amount":          amount,
        "balance":         round(float(rng.uniform(100.0, 50_000.0)), 2),
        "credit_limit":    int(rng.integers(1_000, 25_000)),
        "transaction_fee": round(float(rng.uniform(0.0, 35.0)), 2),
        "merchant":        _FIN_MERCHANTS[int(rng.integers(0, len(_FIN_MERCHANTS)))],
        "category":        _FIN_CATEGORIES[int(rng.integers(0, len(_FIN_CATEGORIES)))],
        "status":          _FIN_STATUS[int(rng.integers(0, len(_FIN_STATUS)))],
        "country":         random.choice(["US", "UK", "CA", "AU", "DE", "FR", "JP"]),
    }


def _medical_row(rng: np.random.Generator) -> dict:
    """Realistic patient / medical record."""
    return {
        "patient_id":    f"PAT{int(rng.integers(10_000, 99_999))}",
        "age_years":     int(rng.integers(18, 90)),
        "systolic_bp":   int(rng.integers(90, 140)),
        "diastolic_bp":  int(rng.integers(60, 90)),
        "heart_rate":    int(rng.integers(55, 100)),
        "glucose_level": round(float(rng.uniform(70.0, 140.0)), 1),
        "bmi":           round(float(rng.uniform(18.5, 35.0)), 1),
        "dosage_mg":     round(float(rng.choice([5, 10, 25, 50, 100, 250, 500])), 1),
        "lab_value":     round(float(rng.uniform(0.5, 10.0)), 2),
        "diagnosis":     _MED_DIAGNOSES[int(rng.integers(0, len(_MED_DIAGNOSES)))],
        "medication":    _MED_MEDS[int(rng.integers(0, len(_MED_MEDS)))],
        "blood_type":    _MED_BLOOD_TYPES[int(rng.integers(0, len(_MED_BLOOD_TYPES)))],
    }


def _ecommerce_row(rng: np.random.Generator) -> dict:
    """Realistic ecommerce order record."""
    price = round(float(rng.uniform(5.0, 500.0)), 2)
    qty   = int(rng.integers(1, 20))
    return {
        "order_id":      f"ORD{int(rng.integers(100_000, 999_999))}",
        "sku":           f"SKU{int(rng.integers(1_000, 9_999))}",
        "price":         price,
        "quantity":      qty,
        "discount_pct":  round(float(rng.uniform(0.0, 40.0)), 1),
        "shipping_days": int(rng.integers(1, 14)),
        "review_score":  round(float(rng.uniform(1.0, 5.0)), 1),
        "return_rate":   round(float(rng.uniform(0.0, 0.3)), 3),
        "units_sold":    int(rng.integers(0, 10_000)),
        "category":      _ECO_CATEGORIES[int(rng.integers(0, len(_ECO_CATEGORIES)))],
        "carrier":       _ECO_CARRIERS[int(rng.integers(0, len(_ECO_CARRIERS)))],
        "status":        _ECO_STATUS[int(rng.integers(0, len(_ECO_STATUS)))],
    }


# Domain registry
_DOMAIN_GENERATORS = {
    "hr":         _hr_row,
    "finance":    _finance_row,
    "medical":    _medical_row,
    "ecommerce":  _ecommerce_row,
}

# Numeric columns per domain (for schema detection)
_DOMAIN_NUMERIC = {
    "hr":        {"age", "salary", "experience", "years_at_company",
                  "performance_score", "bonus", "overtime_hours", "rating"},
    "finance":   {"amount", "balance", "credit_limit", "transaction_fee"},
    "medical":   {"age_years", "systolic_bp", "diastolic_bp", "heart_rate",
                  "glucose_level", "bmi", "dosage_mg", "lab_value"},
    "ecommerce": {"price", "quantity", "discount_pct", "shipping_days",
                  "review_score", "return_rate", "units_sold"},
}

# Valid ranges per domain column (for invalid_range detection)
_DOMAIN_RANGES = {
    "hr": {
        "age":               (22, 62),
        "salary":            (35_000, 180_000),
        "experience":        (0, 30),
        "years_at_company":  (0, 20),
        "performance_score": (2.5, 4.9),
        "bonus":             (0, 25_000),
        "overtime_hours":    (0, 80),
        "rating":            (2.5, 4.9),
    },
    "finance": {
        "amount":          (0.01, 10_000),
        "balance":         (0, 100_000),
        "credit_limit":    (500, 50_000),
        "transaction_fee": (0, 100),
    },
    "medical": {
        "age_years":     (0, 120),
        "systolic_bp":   (70, 200),
        "diastolic_bp":  (40, 130),
        "heart_rate":    (30, 200),
        "glucose_level": (40, 500),
        "bmi":           (10, 70),
        "dosage_mg":     (0.1, 2000),
        "lab_value":     (0, 100),
    },
    "ecommerce": {
        "price":         (0.01, 10_000),
        "quantity":      (0, 10_000),
        "discount_pct":  (0, 100),
        "shipping_days": (0, 60),
        "review_score":  (1.0, 5.0),
        "return_rate":   (0.0, 1.0),
        "units_sold":    (0, 1_000_000),
    },
}

# Difficulty → which domains are active
_LEVEL_DOMAINS = {
    "easy":   ["hr"],                                          # single domain
    "medium": ["hr", "finance"],                              # two domains
    "hard":   ["hr", "finance", "medical", "ecommerce"],      # all four
}

# Difficulty → issue types allowed
_LEVEL_ISSUES = {
    "easy":   ["missing", "outlier", "invalid_negative"],
    "medium": ["missing", "outlier", "invalid_negative", "type_mismatch", "whitespace_padding"],
    "hard":   ["missing", "outlier", "invalid_negative", "type_mismatch",
               "whitespace_padding", "duplicate", "invalid_range"],
}

# Difficulty → (n_rows, n_issues_target)
_LEVEL_SIZE = {
    "easy":   (40,  12),
    "medium": (80,  20),
    "hard":   (120, 30),
}


# ---------------------------------------------------------------------------
# Issue injection helpers
# ---------------------------------------------------------------------------

def _inject_issues(
    rows:        list[dict],
    domain:      str,
    issue_types: list[str],
    n_issues:    int,
    rng:         np.random.Generator,
) -> list[tuple[int, str, str]]:
    """
    Inject exactly n_issues dirty values into rows.
    Returns list of (row_idx, col, issue_type) — the ground-truth issue list.
    """
    numeric_cols = list(_DOMAIN_NUMERIC[domain])
    ranges       = _DOMAIN_RANGES.get(domain, {})
    all_cols     = list(rows[0].keys())
    cat_cols     = [c for c in all_cols if c not in numeric_cols]

    injected: list[tuple[int, str, str]] = []
    used_rows: set[int] = set()       # one issue per row max
    attempts  = 0

    # Shuffle issue types to spread variety
    issue_pool = (issue_types * (n_issues // len(issue_types) + 2))[:n_issues]
    rng.shuffle(issue_pool)

    for issue in issue_pool:
        attempts += 1
        if attempts > n_issues * 10:
            break   # safety valve

        # Pick a row not already dirtied
        available = [i for i in range(len(rows)) if i not in used_rows]
        if not available:
            break
        row_idx = int(rng.choice(available))

        if issue == "missing":
            col = str(rng.choice(all_cols))
            rows[row_idx][col] = None
            injected.append((row_idx, col, "missing"))
            used_rows.add(row_idx)

        elif issue == "outlier" and numeric_cols:
            col = str(rng.choice(numeric_cols))
            if col in ranges:
                lo, hi = ranges[col]
                # inject a value well outside [lo, hi]
                spike = float(hi) * float(rng.uniform(3.0, 8.0))
                rows[row_idx][col] = round(spike, 2)
                injected.append((row_idx, col, "outlier"))
                used_rows.add(row_idx)

        elif issue == "invalid_negative" and numeric_cols:
            col = str(rng.choice(numeric_cols))
            if col in ranges and ranges[col][0] >= 0:
                rows[row_idx][col] = -abs(float(rng.uniform(1, 50)))
                injected.append((row_idx, col, "invalid_negative"))
                used_rows.add(row_idx)

        elif issue == "type_mismatch" and numeric_cols:
            col = str(rng.choice(numeric_cols))
            garbage = rng.choice(["N/A", "unknown", "??", "none", "TBD", "error"])
            rows[row_idx][col] = str(garbage)
            injected.append((row_idx, col, "type_mismatch"))
            used_rows.add(row_idx)

        elif issue == "whitespace_padding" and cat_cols:
            col = str(rng.choice(cat_cols))
            val = rows[row_idx].get(col, "")
            if isinstance(val, str):
                pad = rng.choice(["leading", "trailing", "both"])
                if pad == "leading":
                    rows[row_idx][col] = "  " + val
                elif pad == "trailing":
                    rows[row_idx][col] = val + "  "
                else:
                    rows[row_idx][col] = "  " + val + "  "
                injected.append((row_idx, col, "whitespace_padding"))
                used_rows.add(row_idx)

        elif issue == "duplicate" and len(injected) > 0:
            # Copy a previously-injected row's index (already fixed in clean copy)
            src_idx = injected[int(rng.integers(0, len(injected)))][0]
            rows[row_idx] = dict(rows[src_idx])
            injected.append((row_idx, "__row__", "duplicate"))
            used_rows.add(row_idx)

        elif issue == "invalid_range" and numeric_cols and ranges:
            # Value within schema bounds but violates domain logic
            # e.g. diastolic_bp > systolic_bp, discount > 100, review > 5
            col = str(rng.choice([c for c in numeric_cols if c in ranges]))
            lo, hi = ranges[col]
            # boundary violation: slightly above hi
            rows[row_idx][col] = round(float(hi) * float(rng.uniform(1.05, 1.5)), 2)
            injected.append((row_idx, col, "invalid_range"))
            used_rows.add(row_idx)

    return injected


# ===========================================================================
class DataCleaningEnv:
# ===========================================================================
    """
    Multi-domain RL environment for tabular data cleaning.

    OpenEnv interface: reset() / step() / state()

    Each episode samples one or more domains (depending on difficulty level),
    generates a realistic dirty DataFrame, and presents one dirty row at a time.
    The agent must select the correct remediation action from raw observations.
    """

    def __init__(self):
        self.df                    = None
        self.original_df           = None
        self.issues: list          = []
        self.current_idx           = 0
        self.steps                 = 0
        self.max_steps             = 100
        self.episode_log: list     = []
        self._task_level           = None
        self._is_custom            = False
        self._active_domain        = None
        self.numeric_cols: list    = []
        self.cat_cols: list        = []
        self.means: dict           = {}
        self.modes: dict           = {}
        self.total_issues_at_start = 0
        self._episode_count: int   = 0   # tracks calls to reset() for seed rotation

    # -----------------------------------------------------------------------
    # BUILT-IN TASK DATA — multi-domain procedural generation
    # -----------------------------------------------------------------------
    def load_task_data(self, task_level: str, episode_offset: int = 0) -> tuple[list[dict], str]:
        """
        Generate a dirty multi-domain dataset for the given difficulty level.

        The seed rotates with each episode_offset so consecutive reset() calls
        produce different domains and issue placements — giving natural score
        variance across runs while remaining fully reproducible per offset value.

        Returns (rows, domain_name, issues).
        """
        if task_level not in _SEEDS:
            raise ValueError(
                f"Unknown task_level {task_level!r}. Use 'easy', 'medium', or 'hard'."
            )

        # Rotating seed: base + episode offset (mod large prime keeps it varied)
        seed = _SEEDS[task_level] + (episode_offset * 31) % 9973
        rng  = np.random.default_rng(seed)

        active_domains = _LEVEL_DOMAINS[task_level]
        issue_types    = _LEVEL_ISSUES[task_level]
        n_rows, n_issues = _LEVEL_SIZE[task_level]

        # Domain rotates each episode — on hard all 4 domains appear across runs
        domain = active_domains[int(rng.integers(0, len(active_domains)))]
        gen    = _DOMAIN_GENERATORS[domain]

        # Generate clean rows then inject issues
        rows   = [gen(rng) for _ in range(n_rows)]
        issues = _inject_issues(rows, domain, issue_types, n_issues, rng)

        return rows, domain, issues

    # -----------------------------------------------------------------------
    # RESET — built-in dataset
    # -----------------------------------------------------------------------
    def reset(self, task_level: str = "medium"):
        if task_level not in _SEEDS:
            raise ValueError(
                f"Unknown task_level {task_level!r}. Use 'easy', 'medium', or 'hard'."
            )

        self._task_level = task_level
        self._is_custom  = False
        self.episode_log = []

        self._episode_count += 1
        rows, domain, issues = self.load_task_data(task_level, self._episode_count)
        self._active_domain  = domain

        self.df          = pd.DataFrame(rows)
        self.original_df = self.df.copy()

        # Schema from domain registry — exact, not inferred
        known_num = _DOMAIN_NUMERIC.get(domain, set())
        self.numeric_cols = [c for c in self.df.columns if c in known_num]
        self.cat_cols     = [c for c in self.df.columns if c not in known_num]

        self.compute_stats()

        # Use pre-computed ground-truth issues from injection
        self.issues = list(issues)

        self.current_idx           = 0
        self.steps                 = 0
        self.total_issues_at_start = len(self.issues)
        self.max_steps             = max(20, 3 * self.total_issues_at_start)

        return self.get_observation()

    # -----------------------------------------------------------------------
    # RESET — user-uploaded DataFrame (unchanged interface)
    # -----------------------------------------------------------------------
    def reset_from_dataframe(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("DataFrame is empty.")

        self._task_level    = "custom"
        self._is_custom     = True
        self._active_domain = "custom"
        self.df             = df.copy().reset_index(drop=True)
        self.original_df    = self.df.copy()
        self.episode_log    = []

        self._detect_generic_schema()
        self.compute_stats()
        self._detect_generic_issues()

        self.current_idx           = 0
        self.steps                 = 0
        self.total_issues_at_start = len(self.issues)
        self.max_steps             = max(20, 3 * self.total_issues_at_start)

        return self.get_observation()

    # -----------------------------------------------------------------------
    # SCHEMA — generic (infer from data, used for CSV upload)
    # -----------------------------------------------------------------------
    def _detect_generic_schema(self):
        self.numeric_cols, self.cat_cols = [], []
        for col in self.df.columns:
            non_null     = self.df[col].dropna()
            if len(non_null) == 0:
                self.cat_cols.append(col); continue
            coerced      = pd.to_numeric(non_null, errors="coerce")
            numeric_frac = coerced.notna().sum() / len(non_null)
            if numeric_frac >= 0.6:
                self.numeric_cols.append(col)
            else:
                self.cat_cols.append(col)

    # -----------------------------------------------------------------------
    # STATS
    # -----------------------------------------------------------------------
    def compute_stats(self):
        self.means = {}
        for col in self.numeric_cols:
            s = pd.to_numeric(self.df[col], errors="coerce")
            m = s.mean(skipna=True)
            self.means[col] = float(m) if not pd.isna(m) else 0.0

        self.modes = {}
        for col in self.cat_cols:
            clean  = self.df[col].dropna().apply(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            mode_s = clean.mode(dropna=True)
            self.modes[col] = str(mode_s.iloc[0]) if not mode_s.empty else ""

    # -----------------------------------------------------------------------
    # DUPLICATE KEY
    # -----------------------------------------------------------------------
    def _row_key(self, row) -> tuple:
        return tuple(
            None if (isinstance(v, float) and pd.isna(v)) else v
            for v in row
        )

    # -----------------------------------------------------------------------
    # ISSUE DETECTION — generic / user upload (IQR-based)
    # -----------------------------------------------------------------------
    def _detect_generic_issues(self):
        self.issues  = []
        seen_keys: set = set()

        def _to_float(v):
            if isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v)):
                return float(v)
            if isinstance(v, str):
                try: return float(v.strip())
                except: return None
            return None

        def _is_true_mismatch(v) -> bool:
            return isinstance(v, str) and _to_float(v) is None

        iqr_bounds: dict = {}
        for col in self.numeric_cols:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(series) < 4: continue
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr    = q3 - q1
            iqr_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        non_neg_cols = {
            col for col in self.numeric_cols
            if pd.to_numeric(self.df[col], errors="coerce").dropna().min() >= 0
        }

        for idx, row in self.df.iterrows():
            key = self._row_key(row)
            if key in seen_keys:
                self.issues.append((idx, "__row__", "duplicate")); continue
            seen_keys.add(key)

            for col in self.df.columns:
                val = row[col]

                if col in self.numeric_cols and isinstance(val, str):
                    try: float(val)
                    except:
                        self.issues.append((idx, col, "type_mismatch")); break

                if isinstance(val, str) and val != val.strip():
                    self.issues.append((idx, col, "whitespace_padding")); break

                if not isinstance(val, str) and pd.isna(val):
                    self.issues.append((idx, col, "missing")); break

                num = _to_float(val)
                if col in iqr_bounds and num is not None:
                    lo, hi = iqr_bounds[col]
                    if num < lo:
                        issue = "invalid_negative" if (num < 0 and col in non_neg_cols) else "outlier"
                        self.issues.append((idx, col, issue)); break
                    if num > hi:
                        self.issues.append((idx, col, "outlier")); break

    # -----------------------------------------------------------------------
    # OBSERVATION — Partially Observable
    # -----------------------------------------------------------------------
    def get_observation(self) -> dict | None:
        """
        Returns raw row values + column statistics.
        The domain name and issue type are NEVER included.
        The agent must infer what is wrong from z-score context alone.
        """
        if not self.issues:
            return None
        if self.current_idx >= len(self.issues):
            self.current_idx = 0

        row_idx, _, _ = self.issues[self.current_idx]
        if row_idx not in self.df.index:
            self.current_idx = 0
            row_idx, _, _ = self.issues[0]

        row_dict = self.df.loc[row_idx].to_dict()

        col_stats: dict = {}
        for col in self.numeric_cols:
            s = pd.to_numeric(self.df[col], errors="coerce")
            m = float(s.mean(skipna=True)) if not pd.isna(s.mean(skipna=True)) else 0.0
            d = float(s.std(skipna=True))  if not pd.isna(s.std(skipna=True))  else 1.0
            col_stats[col] = {"mean": round(m, 2), "std": round(max(d, 0.01), 2)}

        return {
            "row_data":         row_dict,
            "column_stats":     col_stats,
            "step_progress":    round(self.steps / max(self.max_steps, 1), 4),
            "issues_remaining": len(self.issues),
        }

    # -----------------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------------
    def step(self, action: int):
        if not self.issues:
            return None, 0, True, {}

        self.steps += 1
        row_idx, col, issue_type = self.issues[self.current_idx]

        # Correct action: 1 for missing, 3 for everything else
        correct_action = 1 if issue_type == "missing" else 3
        correct        = (action == correct_action)
        old_val        = str(self.df.loc[row_idx, col]) if col != "__row__" else "duplicate_row"
        new_val        = "—"
        reward         = 0
        skip_pop       = False

        if action == 2:
            # flag_for_review: marks but does not fix. Small negative reward.
            self.episode_log.append({
                "step": self.steps, "row": int(row_idx), "col": col,
                "issue": issue_type, "action": "flag_for_review",
                "correct": False, "old_value": old_val,
                "new_value": "flagged", "reward": -0.5,
            })
            self.current_idx = (self.current_idx + 1) % max(len(self.issues), 1)
            if not self.issues:
                return None, -0.5 + 5, True, {}
            if self.steps >= self.max_steps:
                return None, -0.5 - 5, True, {}
            return self.get_observation(), -0.5, False, {}

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
                remaining = [(r, c, t) for i, (r, c, t) in enumerate(self.issues)
                             if i != self.current_idx]
                self.issues = [(r - 1 if r > row_idx else r, c, t) for (r, c, t) in remaining]
                skip_pop = True

            elif issue_type == "type_mismatch":
                self.df[col] = pd.to_numeric(self.df[col], errors="coerce")
                col_mean     = pd.to_numeric(self.df[col], errors="coerce").mean(skipna=True)
                nv           = float(col_mean) if not pd.isna(col_mean) else 0.0
                self.df.loc[row_idx, col] = nv
                new_val = str(round(nv, 4))

            elif issue_type == "whitespace_padding":
                nv = str(self.df.loc[row_idx, col]).strip()
                self.df.loc[row_idx, col] = nv
                new_val = nv

            else:   # outlier, invalid_negative, invalid_range
                nv = self.means.get(col, 0.0)
                if col in self.df.columns and self.df[col].dtype == "int64":
                    self.df[col] = self.df[col].astype("float64")
                if col in self.df.columns:
                    self.df.loc[row_idx, col] = nv
                new_val = str(round(nv, 4))

            reward = 2
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

    # -----------------------------------------------------------------------
    # STATE (OpenEnv 3rd method)
    # -----------------------------------------------------------------------
    def state(self) -> dict:
        done  = (len(self.issues) == 0 or self.steps >= self.max_steps)
        total = getattr(self, "total_issues_at_start", 0)
        return {
            "task_level":            self._task_level,
            "domain":                self._active_domain,
            "episode":               self._episode_count,
            "current_step":          self.steps,
            "max_steps":             self.max_steps,
            "total_issues_at_start": total,
            "remaining_issues":      len(self.issues),
            "score":                 self.grade() if total > 0 else 0.99,
            "done":                  done,
            "current_observation":   self.get_observation(),
            "episode_log":           list(self.episode_log),
        }

    # -----------------------------------------------------------------------
    # GRADER
    # -----------------------------------------------------------------------
    def grade(self) -> float:
        total = getattr(self, "total_issues_at_start", 0)
        if total == 0:
            return 0.99
        raw = (total - len(self.issues)) / total
        return round(min(max(raw, 0.01), 0.99), 4)
