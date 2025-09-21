# Use a stable Python base (3.11 for best wheel availability)
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Install system deps if needed for grpc / others
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Only copy requirements first for better layer caching
COPY requirements.txt ./
RUN pip install --upgrade pip setuptools wheel && \
    pip install -r requirements.txt

# Copy application code
COPY . .

# Cloud Run expects to listen on $PORT, default 8080
ENV PORT=8080
EXPOSE 8080

# Start the Flask app via gunicorn
# server:app means use `app` from `server.py`
CMD ["gunicorn", "-w", "2", "-k", "gthread", "-b", "0.0.0.0:8080", "server:app"]
