# ── Base ──────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir \
        pandas>=2.0 \
        numpy>=1.24 \
        fastapi>=0.110 \
        "uvicorn[standard]>=0.29" \
        pydantic>=2.0 \
        python-multipart>=0.0.9 \
        httpx>=0.27 \
        openai>=1.0 \
        gradio>=4.0 \
        openenv>=0.2.0

COPY models.py     .
COPY env.py        .
COPY agent.py      .
COPY client.py     .
COPY app.py        .
COPY inference.py  .
COPY gradio_ui.py  .
COPY openenv.yaml  .
COPY start.sh      .

RUN chmod +x start.sh

ENV API_BASE_URL="" \
    MODEL_NAME="meta-llama/Llama-3.2-3B-Instruct" \
    HF_TOKEN="" \
    AGENT_TYPE="baseline" \
    GRADIO_SERVER_PORT=7860

# Port 8000 — FastAPI OpenEnv API  (hackathon checker, openenv validate)
# Port 7860 — Gradio demo UI       (HF Spaces visitor traffic)
EXPOSE 8000 7860

# Healthcheck targets the FastAPI server only (the critical endpoint)
HEALTHCHECK --interval=15s --timeout=5s --start-period=15s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["./start.sh"]
