#!/bin/sh
# start.sh — production entrypoint for HF Spaces Docker runtime
# =============================================================
# Starts both servers inside the single container:
#   FastAPI  on port 8000  — OpenEnv API  (hackathon checker pings this)
#   Gradio   on port 7860  — demo UI      (HF Spaces routes this to visitors)
#
# FastAPI is launched in the background first; Gradio runs in the foreground
# so Docker keeps the container alive.  If FastAPI dies the healthcheck will
# restart the container; if Gradio dies Docker exits and Spaces restarts it.

set -e

echo "[start.sh] Starting FastAPI on port 8000 ..."
uvicorn app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 1 \
    --log-level info &

FASTAPI_PID=$!
echo "[start.sh] FastAPI pid=${FASTAPI_PID}"

# Brief pause so FastAPI is ready before Gradio tries to import env/agent
sleep 2

echo "[start.sh] Starting Gradio UI on port 7860 ..."
GRADIO_SERVER_PORT=7860 python gradio_ui.py
