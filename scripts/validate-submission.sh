#!/usr/bin/env bash
# =============================================================================
# validate-submission.sh — Pre-submission validator for RL Data Cleaning Agent
# =============================================================================
#
# Usage:
#   chmod +x scripts/validate-submission.sh
#   ./scripts/validate-submission.sh <SPACE_URL> [REPO_ROOT]
#
# Arguments:
#   SPACE_URL   Public URL of your Hugging Face Space (no trailing slash)
#               e.g. https://myuser-rl-data-cleaning.hf.space
#   REPO_ROOT   Path to repo root (default: current directory)
#
# What it checks:
#   Step 1/3 — POST <SPACE_URL>/reset returns HTTP 200
#   Step 2/3 — docker build succeeds in REPO_ROOT
#   Step 3/3 — openenv validate passes in REPO_ROOT
#
# Exit codes:
#   0 — all checks passed
#   1 — one or more checks failed
# =============================================================================

set -euo pipefail

SPACE_URL="${1:-}"
REPO_ROOT="${2:-.}"

# ---------------------------------------------------------------------------
# Colours
# ---------------------------------------------------------------------------
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'   # No colour

pass()  { echo -e "${GREEN}[PASS]${NC} $*"; }
fail()  { echo -e "${RED}[FAIL]${NC} $*"; FAILURES=$((FAILURES + 1)); }
info()  { echo -e "${YELLOW}[INFO]${NC} $*"; }

FAILURES=0

echo ""
echo "======================================================================"
echo "  RL Data Cleaning Agent — Pre-submission Validator"
echo "======================================================================"
echo ""

# ---------------------------------------------------------------------------
# Step 1/3 — HF Space /reset endpoint
# ---------------------------------------------------------------------------
echo "Step 1/3: Checking that POST ${SPACE_URL}/reset returns HTTP 200…"

if [[ -z "$SPACE_URL" ]]; then
    fail "SPACE_URL argument is required.  Usage: $0 <SPACE_URL> [REPO_ROOT]"
else
    HTTP_STATUS=$(curl -s -o /dev/null -w "%{http_code}" \
        -X POST "${SPACE_URL}/reset" \
        -H "Content-Type: application/json" \
        -d '{"task_level": "easy"}' \
        --max-time 30 || echo "000")

    if [[ "$HTTP_STATUS" == "200" ]]; then
        pass "POST ${SPACE_URL}/reset → HTTP ${HTTP_STATUS}"
    else
        fail "POST ${SPACE_URL}/reset → HTTP ${HTTP_STATUS} (expected 200)"
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Step 2/3 — Docker build
# ---------------------------------------------------------------------------
echo "Step 2/3: Building Docker image from ${REPO_ROOT}…"

if ! command -v docker &>/dev/null; then
    fail "'docker' command not found. Please install Docker."
else
    if docker build -t rl-data-cleaning-env-test "${REPO_ROOT}" 2>&1; then
        pass "docker build succeeded."
    else
        fail "docker build failed. Check the Dockerfile and requirements.txt."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Step 3/3 — openenv validate
# ---------------------------------------------------------------------------
echo "Step 3/3: Running 'openenv validate' in ${REPO_ROOT}…"

if ! command -v openenv &>/dev/null; then
    info "'openenv' CLI not found. Skipping openenv validate."
    info "Install with: pip install openenv  (or follow HF course instructions)"
    # Do not count as a failure — tool may not be installed locally
else
    if (cd "${REPO_ROOT}" && openenv validate) 2>&1; then
        pass "openenv validate passed."
    else
        fail "openenv validate reported errors. Check openenv.yaml and endpoint responses."
    fi
fi
echo ""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo "======================================================================"
if [[ $FAILURES -eq 0 ]]; then
    echo -e "${GREEN}All checks passed! Ready for submission.${NC}"
    exit 0
else
    echo -e "${RED}${FAILURES} check(s) failed. Please fix the issues above before submitting.${NC}"
    exit 1
fi
