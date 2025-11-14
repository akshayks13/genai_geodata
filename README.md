# Gemma Server Deployment

This repo contains a Flask service (`server.py`) that queries BigQuery, enriches the query via Vertex AI/Gemma, and returns an answer using Gemini.

Note: `server.py` currently contains hard-coded API keys and a BigQuery project ID (as per your original script). For production, you should replace these with environment variables and/or Google Application Default Credentials (ADC).

## Local run (localhost only)

### 1. Create and activate a virtual environment (macOS/Linux)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Set required environment variables

Minimal variables (replace the placeholder values):

```bash
export GENAI_API_KEY="<gemma-or-google-genai-key>"        # Used by Gemma enrichment
export GEMINI_API_KEY="<gemini-api-key>"                  # Used for final answer & RAG
export BQ_PROJECT_ID="<bigquery-project-id>"              # Project that owns embedding table
export DISCOVERY_ENGINE_ID="<discovery-engine-id>"        # For /discovery/search route
export VERTEX_INDEX_ENDPOINT="<vertex-index-endpoint>"    # e.g. projects/123/locations/us-central1/indexEndpoints/456
export VERTEX_DEPLOYED_INDEX_ID="<deployed-index-id>"     # e.g. 7890123456789012345
# Optional / tuning
export CORS_ORIGINS="http://localhost:3000"               # Comma-separated list; defaults allow local dev
export VERTEX_API_ENDPOINT="us-central1-aiplatform.googleapis.com"  # Override if needed
```

Data/key files (if you use them) expected by the hybrid search:

* `gemini_cleaned_text.json` – local corpus for TF-IDF (place in repo root or same dir as `server.py`).
* `vertexmanager-key.json` – service account key (optional; prefer ADC). If omitted, Application Default Credentials are used.

### 3. Run the server

```bash
python3 server.py
```

The app listens on `http://0.0.0.0:8080`.

### 4. Available endpoints (all POST except /health)

| Endpoint | Purpose | Sample Payload |
|----------|---------|----------------|
| `GET /health` | Liveness check | n/a |
| `POST /query` | BigQuery + enrichment + Gemini answer | `{ "question": "What are the cybersecurity risks for AI adoption in Indian cities?" }` |
| `POST /discovery/search` | Discovery Engine search passthrough | `{ "query": "renewable energy" }` |
| `POST /hybrid_search` | Dense + sparse (TF-IDF) + Vertex neighbors + RAG | `{ "question": "Impact of AI on transport safety", "neighbor_count": 8, "alpha": 0.5 }` |

Field notes:
* `neighbor_count` (int) – how many neighbors to request from Vertex (defaults inside code if omitted).
* `alpha` (float 0–1) – balancing parameter for sparse vs dense retrieval (RRF style); optional.

### 5. Sample curl commands (localhost)

Health:
```bash
curl -s http://localhost:8080/health | jq
```

Query:
```bash
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the cybersecurity risks for AI adoption in Indian cities?"}' | jq
```

Discovery search:
```bash
curl -s -X POST http://localhost:8080/discovery/search \
  -H 'Content-Type: application/json' \
  -d '{"query": "renewable energy"}' | jq
```

Hybrid search (dense + sparse + RAG):
```bash
curl -s -X POST http://localhost:8080/hybrid_search \
  -H 'Content-Type: application/json' \
  -d '{"question": "Impact of AI on transport safety", "neighbor_count": 8, "alpha": 0.55}' | jq
```

If you do not have `jq` installed, you can omit the final pipe or install it:

```bash
brew install jq
```

## Docker build & run

```bash
# Build image
docker build -t gemma-server:latest .

# Run container (map port 8080)
docker run --rm -p 8080:8080 \
  gemma-server:latest
```

## Credentials & IAM

- If running locally or in Docker, your script will use the values in `server.py`. For production, prefer using environment variables and Google ADC.
- For BigQuery, ensure the runtime identity (user or service account) has `roles/bigquery.jobUser` (or `roles/bigquery.user`) on the target project.

## Next steps (recommended hardening)

- Parameterize secrets: Replace hard-coded API keys in `server.py` with environment variables (e.g., `os.environ["GENAI_API_KEY"]`, `os.environ["GEMINI_API_KEY"]`).
- Remove keys from the repo and rotate them.
- Optionally switch to explicit service account JSON (or workload identity on Cloud Run) instead of user credentials.
