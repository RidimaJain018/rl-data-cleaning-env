from env import EXPECTED_NUMERIC, SCORE_COLS
import math


def baseline_agent(observation) -> int:
    """
    Rule-based baseline agent — domain-agnostic.
    Works across all four domains (HR, Finance, Medical, Ecommerce)
    using only observable row signals, no hardcoded thresholds.

    Priority:
      1. Non-parseable string in numeric column  → fix_outlier (3)
      2. Whitespace padding in any string value  → fix_outlier (3)
      3. Missing / NaN value                     → impute_missing (1)
      4. Negative in a non-negative column       → fix_outlier (3)
      5. Z-score > 3 vs column_stats             → fix_outlier (3)
      6. No detectable issue                     → skip (0)

    Note: action 2 (flag_for_review) intentionally never returned —
    the baseline always commits to a decision.
    """
    if observation is None:
        return 0

    row   = observation["row_data"]
    stats = observation.get("column_stats", {})

    # 1 — type mismatch: non-parseable string in a numeric-looking column
    for key, val in row.items():
        if isinstance(val, str) and val.strip() not in ("", "nan", "None", "null"):
            try:
                float(val)
            except (ValueError, TypeError):
                # Check if this column has numeric stats (implying it should be numeric)
                if key in stats:
                    return 3

    # 2 — whitespace padding
    for val in row.values():
        if isinstance(val, str) and val != val.strip():
            return 3

    # 3 — missing value
    for val in row.values():
        if val is None or (isinstance(val, float) and math.isnan(val)):
            return 1

    # 4 — negative in a column whose mean is positive
    for key, val in row.items():
        if isinstance(val, (int, float)) and not (isinstance(val, float) and math.isnan(val)):
            if val < 0 and key in stats and stats[key]["mean"] > 0:
                return 3

    # 5 — extreme z-score (>3 std from mean) using column_stats
    for key, val in row.items():
        if key in stats and isinstance(val, (int, float)):
            if isinstance(val, float) and math.isnan(val):
                continue
            mean = stats[key]["mean"]
            std  = stats[key]["std"] or 1.0
            z    = (float(val) - mean) / std
            if abs(z) > 3:
                return 3

    return 0


def upload_agent(observation) -> int:
    """Agent for user-uploaded CSVs. Identical logic to baseline_agent."""
    return baseline_agent(observation)
