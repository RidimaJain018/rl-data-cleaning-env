import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Schema constants — built-in dataset
# ---------------------------------------------------------------------------
EXPECTED_NUMERIC = {
    "age", "salary", "experience", "rating",
    "bonus", "years_at_company", "performance_score", "overtime_hours",
}

# Per-column outlier upper bounds (built-in dataset only)
OUTLIER_THRESHOLDS = {
    "salary": 300_000,
    "bonus":  80_000,
}

# Columns where valid range is 0–5; anything above is "invalid_rating"
SCORE_COLS = {"rating", "performance_score"}

_ACTION_LABELS = {0: "skip", 1: "impute_missing", 3: "fix_outlier"}
_SEEDS         = {"easy": 42, "medium": 123, "hard": 999}
_CITIES        = ["NY", "LA", "SF", "Chicago", "Austin", "Seattle", "Boston", "Denver"]
_DEPARTMENTS   = ["Engineering", "Sales", "HR", "Marketing", "Finance", "Operations"]


# ---------------------------------------------------------------------------
def _clean_row(rng: np.random.Generator) -> dict:
    """One fully-valid 10-column employee row — no data quality issues."""
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


# ===========================================================================
class DataCleaningEnv:
# ===========================================================================

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

    # -----------------------------------------------------------------------
    # BUILT-IN TASK DATA
    # -----------------------------------------------------------------------
    def load_task_data(self, task_level: str) -> list[dict]:
        """
        Procedurally generate the dirty dataset for a built-in task level.

        Issue counts (fixed per level for reproducible baseline scores):
            easy   → 20 rows,  8 issues  (all missing, 8 different columns)
            medium → 30 rows, 12 issues  (5 missing + 3 outliers + 2 score + 1 negative + 1 whitespace)
            hard   → 50 rows, 20 issues  (6 missing + 4 outliers + 2 negatives + 2 duplicates
                                           + 2 type_mismatches + 2 whitespace + 2 invalid_rating)
        """
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
            # 5 missing
            rows[0]["age"]               = None
            rows[1]["city"]              = None
            rows[2]["experience"]        = None
            rows[3]["department"]        = None
            rows[4]["overtime_hours"]    = None
            # 3 outliers
            rows[5]["salary"] = int(rng.integers(320_000, 900_000))
            rows[6]["salary"] = int(rng.integers(600_000, 2_000_000))
            rows[7]["bonus"]  = int(rng.integers(90_000, 250_000))
            # 2 invalid score-range
            rows[8]["performance_score"]  = round(float(rng.uniform(5.5, 9.0)), 1)
            rows[9]["rating"]             = round(float(rng.uniform(5.5, 8.0)), 1)
            # 1 invalid negative
            rows[10]["overtime_hours"]    = -int(rng.integers(5, 40))
            # 1 whitespace padding
            rows[11]["city"]              = f" {rows[11]['city']} "
            return rows

        elif task_level == "hard":
            rows = [_clean_row(rng) for _ in range(50)]
            # 6 missing
            rows[0]["age"]               = None
            rows[2]["city"]              = None
            rows[3]["experience"]        = None
            rows[4]["rating"]            = None
            rows[5]["department"]        = None
            rows[6]["performance_score"] = None
            # 4 outliers
            rows[1]["salary"]  = int(rng.integers(400_000, 1_200_000))
            rows[7]["salary"]  = int(rng.integers(700_000, 2_500_000))
            rows[8]["salary"]  = int(rng.integers(1_000_000, 4_000_000))
            rows[9]["bonus"]   = int(rng.integers(100_000, 350_000))
            # 2 negatives
            rows[10]["overtime_hours"]    = -int(rng.integers(5, 40))
            rows[11]["years_at_company"]  = -int(rng.integers(1, 10))
            # 2 duplicates
            rows[12] = dict(rows[1])
            rows[13] = dict(rows[7])
            # 2 type mismatches (non-parseable string in numeric column)
            rows[14]["age"]              = "N/A"
            rows[15]["years_at_company"] = "ten"
            # 2 whitespace padding
            rows[16]["city"]             = f" {rows[16]['city']} "
            rows[17]["department"]       = f"  {rows[17]['department']}  "
            # 2 invalid rating
            rows[18]["performance_score"] = round(float(rng.uniform(5.5, 9.0)), 1)
            rows[19]["rating"]            = round(float(rng.uniform(5.5, 8.0)), 1)
            return rows

        else:
            raise ValueError(
                f"Unknown task_level {task_level!r}. Use 'easy', 'medium', or 'hard'."
            )

    # -----------------------------------------------------------------------
    # RESET — built-in dataset
    # -----------------------------------------------------------------------
    def reset(self, task_level: str = "medium"):
        self._task_level = task_level
        self._is_custom  = False
        self.df          = pd.DataFrame(self.load_task_data(task_level))
        self.original_df = self.df.copy()
        self.episode_log = []
        self._detect_builtin_schema()
        self.compute_stats()
        self._detect_builtin_issues()
        self.current_idx            = 0
        self.steps                  = 0
        self.total_issues_at_start  = len(self.issues)
        self.max_steps              = max(20, 3 * self.total_issues_at_start)
        return self.get_observation()

    # -----------------------------------------------------------------------
    # RESET — user-uploaded DataFrame
    # -----------------------------------------------------------------------
    def reset_from_dataframe(self, df: pd.DataFrame):
        """
        Load any user-provided DataFrame and detect issues using
        column-agnostic, IQR-based rules. Works with any column names.

        Parameters
        ----------
        df : pd.DataFrame — the dirty dataset. Index is reset internally.

        Returns
        -------
        dict | None — first observation, or None if data is already clean.
        """
        if df is None or df.empty:
            raise ValueError("DataFrame is empty.")

        self._task_level = "custom"
        self._is_custom  = True
        self.df          = df.copy().reset_index(drop=True)
        self.original_df = self.df.copy()
        self.episode_log = []
        self._detect_generic_schema()
        self.compute_stats()
        self._detect_generic_issues()
        self.current_idx           = 0
        self.steps                 = 0
        self.total_issues_at_start = len(self.issues)
        self.max_steps             = max(20, 3 * self.total_issues_at_start)
        return self.get_observation()

    # -----------------------------------------------------------------------
    # SCHEMA — built-in (EXPECTED_NUMERIC ground truth)
    # -----------------------------------------------------------------------
    def _detect_builtin_schema(self):
        self.numeric_cols = [c for c in self.df.columns if c in EXPECTED_NUMERIC]
        self.cat_cols     = [c for c in self.df.columns if c not in EXPECTED_NUMERIC]

    # -----------------------------------------------------------------------
    # SCHEMA — generic (infer from data)
    # -----------------------------------------------------------------------
    def _detect_generic_schema(self):
        """
        Classify each column as numeric or categorical by checking how many
        non-null values can be coerced to a number.  A column is numeric if
        ≥60% of its non-null values parse as float.
        """
        self.numeric_cols, self.cat_cols = [], []
        for col in self.df.columns:
            non_null = self.df[col].dropna()
            if len(non_null) == 0:
                self.cat_cols.append(col)
                continue
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
            clean   = self.df[col].dropna().apply(
                lambda v: v.strip() if isinstance(v, str) else v
            )
            mode_s  = clean.mode(dropna=True)
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
    # ISSUE DETECTION — built-in (domain-specific thresholds)
    # -----------------------------------------------------------------------
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

                if col in EXPECTED_NUMERIC and isinstance(val, str):
                    self.issues.append((idx, col, "type_mismatch")); break

                if isinstance(val, str) and val != val.strip():
                    self.issues.append((idx, col, "whitespace_padding")); break

                if not isinstance(val, str) and pd.isna(val):
                    self.issues.append((idx, col, "missing")); break

                if col in OUTLIER_THRESHOLDS and isinstance(val, (int, float)) \
                        and val > OUTLIER_THRESHOLDS[col]:
                    self.issues.append((idx, col, "outlier")); break

                if col in SCORE_COLS and isinstance(val, (int, float)) and val > 5:
                    self.issues.append((idx, col, "invalid_rating")); break

                if col in EXPECTED_NUMERIC and isinstance(val, (int, float)) and val < 0:
                    self.issues.append((idx, col, "invalid_negative")); break

    # -----------------------------------------------------------------------
    # ISSUE DETECTION — generic / user upload (IQR-based)
    # -----------------------------------------------------------------------
    def _detect_generic_issues(self):
        """
        Column-agnostic issue detection for user-uploaded data.

        Key insight: when pandas reads a column that has one bad value
        (e.g. 'N/A'), the whole column becomes object dtype and stores
        valid numbers as strings like '55000'.  We split strings into:
          • Numeric string  — '55000', '3.14'  → can be coerced, apply IQR
          • True mismatch   — 'N/A', 'unknown' → cannot be coerced → type_mismatch

        Priority per row: duplicate → type_mismatch → whitespace →
                          missing → outlier / invalid_negative
        """
        self.issues  = []
        seen_keys: set = set()

        # ── helpers ───────────────────────────────────────────────────────
        def _to_float(v):
            """Return float if v is numeric or a numeric string, else None."""
            if isinstance(v, (int, float)) and not (isinstance(v, float) and pd.isna(v)):
                return float(v)
            if isinstance(v, str):
                try:
                    return float(v.strip())
                except (ValueError, TypeError):
                    return None
            return None

        def _is_true_mismatch(v) -> bool:
            """String that cannot be parsed as a number (genuine type mismatch)."""
            return isinstance(v, str) and _to_float(v) is None

        # ── pre-compute IQR bounds ────────────────────────────────────────
        iqr_bounds: dict[str, tuple[float, float]] = {}
        for col in self.numeric_cols:
            series = pd.to_numeric(self.df[col], errors="coerce").dropna()
            if len(series) < 4:
                continue
            q1, q3 = float(series.quantile(0.25)), float(series.quantile(0.75))
            iqr    = q3 - q1
            iqr_bounds[col] = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)

        # Numeric columns where every valid value is ≥ 0 should never go negative
        non_neg_cols = {
            col for col in self.numeric_cols
            if pd.to_numeric(self.df[col], errors="coerce").dropna().min() >= 0
        }

        # ── scan rows ─────────────────────────────────────────────────────
        for idx, row in self.df.iterrows():
            key = self._row_key(row)
            if key in seen_keys:
                self.issues.append((idx, "__row__", "duplicate"))
                continue
            seen_keys.add(key)

            for col in self.df.columns:
                val = row[col]

                # True type mismatch (non-parseable string in numeric column)
                if col in self.numeric_cols and _is_true_mismatch(val):
                    self.issues.append((idx, col, "type_mismatch")); break

                # Whitespace padding in string column
                if isinstance(val, str) and val != val.strip():
                    self.issues.append((idx, col, "whitespace_padding")); break

                # Missing value
                if not isinstance(val, str) and pd.isna(val):
                    self.issues.append((idx, col, "missing")); break

                # IQR outlier / invalid negative (works for both float and numeric strings)
                num = _to_float(val)
                if col in iqr_bounds and num is not None:
                    lo, hi = iqr_bounds[col]
                    if num < lo:
                        issue = "invalid_negative" if (num < 0 and col in non_neg_cols) else "outlier"
                        self.issues.append((idx, col, issue)); break
                    if num > hi:
                        self.issues.append((idx, col, "outlier")); break

    # -----------------------------------------------------------------------
    # OBSERVATION
    # -----------------------------------------------------------------------
    def get_observation(self) -> dict | None:
        if not self.issues:
            return None
        if self.current_idx >= len(self.issues):
            self.current_idx = 0
        row_idx, _, _ = self.issues[self.current_idx]
        if row_idx not in self.df.index:          # guard after duplicate drop
            self.current_idx = 0
            row_idx, _, _ = self.issues[0]
        return {"row_data": self.df.loc[row_idx].to_dict()}

    # -----------------------------------------------------------------------
    # STEP
    # -----------------------------------------------------------------------
    def step(self, action: int):
        if not self.issues:
            return None, 0, True, {}

        self.steps += 1
        row_idx, col, issue_type = self.issues[self.current_idx]
        correct_action = 1 if issue_type == "missing" else 3
        correct        = (action == correct_action)
        old_val        = str(self.df.loc[row_idx, col]) if col != "__row__" else "duplicate_row"
        new_val        = "—"
        reward         = 0
        skip_pop       = False

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

            else:   # outlier / invalid_rating / invalid_negative
                nv = self.means.get(col, 0.0)
                if self.df[col].dtype == "int64":
                    self.df[col] = self.df[col].astype("float64")
                self.df.loc[row_idx, col] = nv
                new_val = str(round(nv, 4))

            reward = 2
            if self._is_custom:
                self._detect_generic_schema()
            else:
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

    # -----------------------------------------------------------------------
    # STATE (OpenEnv 3rd method)
    # -----------------------------------------------------------------------
    def state(self) -> dict:
        done  = (len(self.issues) == 0 or self.steps >= self.max_steps)
        total = getattr(self, "total_issues_at_start", 0)
        return {
            "task_level":            self._task_level,
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
            return 0.99  # clamp: checker requires strictly < 1.0
        raw = (total - len(self.issues)) / total
        # Clamp to open interval (0, 1) — checker rejects exactly 0.0 or 1.0
        return round(min(max(raw, 0.01), 0.99), 4)
