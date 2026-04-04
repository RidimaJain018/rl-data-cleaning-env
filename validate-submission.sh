#!/usr/bin/env bash
# =============================================================================
# validate-submission.sh
# Pre-submission validator for DataCleaningEnv (OpenEnv Hackathon)
#
# Usage:
#   chmod +x scripts/validate-submission.sh
#   ./scripts/validate-submission.sh <SPACE_URL> <PROJECT_ROOT>
#
# Examples:
#   ./scripts/validate-submission.sh https://myuser-rl-cleaning.hf.space .
#   ./scripts/validate-submission.sh http://localhost:8000 .
#
# Steps performed:
#   1/5 — POST <SPACE_URL>/reset → asserts HTTP 200 + valid JSON
#   2/5 — GET  <SPACE_URL>/health → asserts {"status":"ok"}
#   3/5 — docker build <PROJECT_ROOT> → asserts exit 0
#   4/5 — python inference.py (baseline) → asserts exit 0 + score output
#   5/5 — Required env variables present (API_BASE_URL, MODEL_NAME, HF_TOKEN)
#
# Exit codes:
#   0  all checks passed
#   1  one or more checks failed
# =============================================================================

set -euo pipefail

SPACE_URL="${1:-http://localhost:8000}"
PROJECT_ROOT="${2:-.}"

# Strip trailing slash
SPACE_URL="${SPACE_URL%/}"

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

PASS=0
FAIL=0

pass() { echo -e "  ${GREEN}✓${NC} $1"; ((PASS++)) || true; }
fail() { echo -e "  ${RED}✗${NC} $1"; ((FAIL++)) || true; }
warn() { echo -e "  ${YELLOW}!${NC} $1"; }
header() { echo -e "\n${CYAN}${BOLD}$1${NC}"; }

echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}║     DataCleaningEnv — Pre-Submission Validator           ║${NC}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "  Space URL    : ${CYAN}${SPACE_URL}${NC}"
echo -e "  Project root : ${CYAN}${PROJECT_ROOT}${NC}"

# =============================================================================
# Step 1/5 — POST /reset (the automated ping check)
# =============================================================================
header "Step 1/5 — POST /reset → HTTP 200"

RESET_STATUS=$(curl -s -o /tmp/vc_reset_body.json -w "%{http_code}" \
    -X POST "${SPACE_URL}/reset" \
    -H "Content-Type: application/json" \
    -d '{"task_level":"easy"}' 2>/dev/null || echo "000")

if [[ "${RESET_STATUS}" == "200" ]]; then
    pass "POST /reset returned HTTP ${RESET_STATUS}"
    # Check observation field exists in response
    if grep -q '"observation"' /tmp/vc_reset_body.json 2>/dev/null; then
        pass "Response contains 'observation' field"
    else
        fail "Response body missing 'observation' field — got: $(cat /tmp/vc_reset_body.json)"
    fi
else
    fail "POST /reset returned HTTP ${RESET_STATUS} (expected 200)"
    warn "Is the server running? Start with: uvicorn app:app --port 8000"
fi

# =============================================================================
# Step 2/5 — GET /health
# =============================================================================
header "Step 2/5 — GET /health → {\"status\":\"ok\"}"

HEALTH_STATUS=$(curl -s -o /tmp/vc_health_body.json -w "%{http_code}" \
    "${SPACE_URL}/health" 2>/dev/null || echo "000")

if [[ "${HEALTH_STATUS}" == "200" ]]; then
    HEALTH_BODY=$(cat /tmp/vc_health_body.json 2>/dev/null || echo "")
    if echo "${HEALTH_BODY}" | grep -q '"ok"'; then
        pass "GET /health returned HTTP 200 with {\"status\":\"ok\"}"
    else
        fail "GET /health returned HTTP 200 but body was: ${HEALTH_BODY}"
    fi
else
    fail "GET /health returned HTTP ${HEALTH_STATUS} (expected 200)"
fi

# =============================================================================
# Step 3/5 — docker build
# =============================================================================
header "Step 3/5 — docker build ${PROJECT_ROOT}"

if ! command -v docker &>/dev/null; then
    warn "docker not found in PATH — skipping build check"
    warn "Install Docker Desktop: https://docker.com/products/docker-desktop"
else
    if docker build -t rl-data-cleaning-env-validate "${PROJECT_ROOT}" \
        > /tmp/vc_docker_build.log 2>&1; then
        pass "docker build completed successfully"
        # Clean up the validation image
        docker rmi rl-data-cleaning-env-validate >/dev/null 2>&1 || true
    else
        fail "docker build failed — see /tmp/vc_docker_build.log"
        tail -20 /tmp/vc_docker_build.log | sed 's/^/    /'
    fi
fi

# =============================================================================
# Step 4/5 — python inference.py (baseline)
# =============================================================================
header "Step 4/5 — python inference.py (baseline agent)"

INFERENCE_SCRIPT="${PROJECT_ROOT}/inference.py"

if [[ ! -f "${INFERENCE_SCRIPT}" ]]; then
    fail "inference.py not found at ${INFERENCE_SCRIPT}"
else
    cd "${PROJECT_ROOT}"
    if python inference.py --agent baseline > /tmp/vc_inference_out.txt 2>&1; then
        pass "inference.py exited 0"
        # Check it printed scores
        if grep -q "Score" /tmp/vc_inference_out.txt || \
           grep -q "score" /tmp/vc_inference_out.txt || \
           grep -q "1.00"  /tmp/vc_inference_out.txt; then
            pass "Output contains score data"
            echo ""
            sed 's/^/    /' /tmp/vc_inference_out.txt
        else
            warn "Could not find score data in output — manual review recommended"
            sed 's/^/    /' /tmp/vc_inference_out.txt
        fi
    else
        fail "inference.py exited with non-zero status"
        echo ""
        tail -30 /tmp/vc_inference_out.txt | sed 's/^/    /'
    fi
    cd - >/dev/null
fi

# =============================================================================
# Step 5/5 — Required environment variables
# =============================================================================
header "Step 5/5 — Required environment variables"

ENV_PASS=true

check_env() {
    local var="$1"
    if [[ -n "${!var:-}" ]]; then
        pass "${var} is set"
    else
        fail "${var} is NOT set (required by hackathon spec)"
        ENV_PASS=false
    fi
}

check_env "API_BASE_URL"
check_env "MODEL_NAME"
check_env "HF_TOKEN"

if [[ "${ENV_PASS}" == "false" ]]; then
    warn "Set missing variables before submitting, e.g.:"
    warn "  export API_BASE_URL='https://api-inference.huggingface.co/v1'"
    warn "  export MODEL_NAME='meta-llama/Llama-3.2-3B-Instruct'"
    warn "  export HF_TOKEN='hf_...'"
fi

# =============================================================================
# Summary
# =============================================================================
echo ""
echo -e "${BOLD}══════════════════════════════════════════════════════════${NC}"
TOTAL=$((PASS + FAIL))
echo -e "  Results: ${GREEN}${PASS} passed${NC} / ${RED}${FAIL} failed${NC} / ${TOTAL} total"

if [[ ${FAIL} -eq 0 ]]; then
    echo -e "\n  ${GREEN}${BOLD}✓ All checks passed — ready to submit!${NC}"
    echo ""
    exit 0
else
    echo -e "\n  ${RED}${BOLD}✗ ${FAIL} check(s) failed — fix before submitting.${NC}"
    echo ""
    exit 1
fi
