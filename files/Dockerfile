# ── Base image ───────────────────────────────────────────────────────────────
FROM python:3.11-slim

# ── System deps ──────────────────────────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# ── Working directory ─────────────────────────────────────────────────────────
WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── App source ────────────────────────────────────────────────────────────────
COPY app.py .

# ── Model artifacts (must exist before docker build) ─────────────────────────
# Run export_artifacts.py first, then copy the outputs here:
COPY lending_club_model.keras .
COPY scaler.pkl .

# ── Runtime config ────────────────────────────────────────────────────────────
ENV MODEL_PATH=lending_club_model.keras
ENV SCALER_PATH=scaler.pkl

EXPOSE 8000

# ── Entrypoint ────────────────────────────────────────────────────────────────
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
