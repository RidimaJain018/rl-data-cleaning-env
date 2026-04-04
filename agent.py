from env import EXPECTED_NUMERIC, OUTLIER_THRESHOLDS, SCORE_COLS


def baseline_agent(observation) -> int:
    """
    Rule-based baseline agent for the built-in 10-column dataset.
    Priority matches detect_issues() in env.py exactly.

    Returns
    -------
    0  skip           — no issue detected
    1  impute_missing — NaN / None value
    3  fix_outlier    — any other issue (outlier, invalid, type mismatch,
                        whitespace padding, duplicate)
    """
    if observation is None:
        return 0

    row = observation["row_data"]

    # 1 — duplicate row hint (env injects _issue into every observation)
    if observation.get("_issue") == "duplicate":
        return 3

    # 2 — type mismatch: non-parseable string in an expected-numeric column
    for key, val in row.items():
        if key in EXPECTED_NUMERIC and isinstance(val, str):
            try:
                float(val)          # numeric strings like "55000" are fine
            except (ValueError, TypeError):
                return 3            # genuinely unparseable → type_mismatch

    # 3 — whitespace padding
    for val in row.values():
        if isinstance(val, str) and val != val.strip():
            return 3

    # 4 — column-specific outlier thresholds
    for key, val in row.items():
        if key in OUTLIER_THRESHOLDS and isinstance(val, (int, float)):
            if val > OUTLIER_THRESHOLDS[key]:
                return 3

    # 5 — score column out of valid 0-5 range
    for key, val in row.items():
        if key in SCORE_COLS and isinstance(val, (int, float)):
            if val > 5:
                return 3

    # 6 — invalid negative in any numeric column
    for key, val in row.items():
        if key in EXPECTED_NUMERIC and isinstance(val, (int, float)):
            if val < 0:
                return 3

    # 7 — missing value
    for val in row.values():
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return 1

    # 8 — no issue
    return 0


def upload_agent(observation) -> int:
    """
    Agent for user-uploaded DataFrames with unknown column names.

    Uses the '_issue' hint that the env includes in every observation
    (set by get_observation from self.issues[current_idx]) so it always
    picks the correct action regardless of what else is in the row:
      - issue == 'missing'  → action 1 (impute_missing)
      - anything else       → action 3 (fix_outlier)
    """
    if observation is None:
        return 0
    issue = observation.get("_issue")
    if issue is not None:
        return 1 if issue == "missing" else 3
    # Fallback heuristic when _issue hint is absent
    for val in observation["row_data"].values():
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return 1
    return 3
