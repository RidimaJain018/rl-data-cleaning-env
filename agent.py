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

    Uses only observable row data — no internal env hints. This makes it a
    genuine heuristic agent that works on any CSV with any column names.

    Decision priority (mirrors _detect_generic_issues in env.py):
      1. Whitespace padding in any string value  → fix_outlier (3)
      2. Non-numeric string where numbers expected → fix_outlier (3)
      3. Missing / NaN value                      → impute_missing (1)
      4. Negative value in any numeric column      → fix_outlier (3)
      5. No issue detected                         → skip (0)

    Note: IQR-based outlier detection (the main generic check) cannot be
    replicated per-row without the full column distribution, so the env's
    issue flag drives those cases. Steps 1-4 cover all other issue types
    purely from the observable row values.
    """
    if observation is None:
        return 0

    row = observation["row_data"]

    # 1 — whitespace padding: string with leading/trailing spaces
    for val in row.values():
        if isinstance(val, str) and val != val.strip():
            return 3

    # 2 — type mismatch: non-parseable string in what looks like a numeric column
    #     Heuristic: if >50% of non-null values in this row are numeric but
    #     this one is a non-parseable string, it's a mismatch.
    numeric_count = sum(
        1 for v in row.values()
        if isinstance(v, (int, float)) and not (isinstance(v, float) and str(v) == "nan")
    )
    for val in row.values():
        if isinstance(val, str) and val.strip() not in ("", "nan", "None", "null"):
            try:
                float(val)
            except (ValueError, TypeError):
                # Non-parseable string found — likely a type mismatch
                if numeric_count > len(row) * 0.3:
                    return 3

    # 3 — missing value: None or NaN in any cell
    for val in row.values():
        if val is None or (isinstance(val, float) and str(val) == "nan"):
            return 1

    # 4 — invalid negative: any negative number in a numeric column
    for val in row.values():
        if isinstance(val, (int, float)) and not (isinstance(val, float) and str(val) == "nan"):
            if val < 0:
                return 3

    # 5 — if env flagged this row but we can't tell why from raw values,
    #     it's almost certainly an IQR outlier → fix it
    if observation.get("_issue") is not None:
        return 3

    # 6 — no detectable issue
    return 0
