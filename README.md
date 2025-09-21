# Gemma Server Deployment

This repo contains a Flask service (`server.py`) that queries BigQuery, enriches the query via Vertex AI/Gemma, and returns an answer using Gemini.

Note: `server.py` currently contains hard-coded API keys and a BigQuery project ID (as per your original script). For production, you should replace these with environment variables and/or Google Application Default Credentials (ADC).

## Local run

1) Create and activate a virtual environment (macOS/Linux):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python server.py
```

App will serve on `http://0.0.0.0:8080` by default.

## Docker build & run

```bash
# Build image
docker build -t gemma-server:latest .

# Run container (map port 8080)
docker run --rm -p 8080:8080 \
  gemma-server:latest
```

Then call the service:

```bash
curl -s -X POST http://localhost:8080/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What are the cybersecurity risks for AI adoption in Indian cities?"}' | jq
```

## Deploy to Cloud Run

Requirements:
- `gcloud` CLI authenticated to your project
- A Google Cloud project with billing enabled
- BigQuery dataset/table accessible to the runtime service account

Set env variables (replace placeholders):

```bash
export PROJECT_ID="your-gcp-project-id"
export REGION="us-central1"
```

Build the image with Cloud Build and push:

```bash
gcloud builds submit --tag gcr.io/$PROJECT_ID/gemma-server
```

Deploy to Cloud Run:

```bash
gcloud run deploy gemma-server \
  --image gcr.io/$PROJECT_ID/gemma-server \
  --platform managed \
  --region $REGION \
  --allow-unauthenticated
```

After deploy, Cloud Run will print a service URL. You can test it with curl as above.

## Credentials & IAM

- If running locally or in Docker, your script will use the values in `server.py`. For production, prefer using environment variables and Google ADC.
- For BigQuery, ensure the runtime identity (user or service account) has `roles/bigquery.jobUser` (or `roles/bigquery.user`) on the target project.

## Next steps (recommended hardening)

- Parameterize secrets: Replace hard-coded API keys in `server.py` with environment variables (e.g., `os.environ["GENAI_API_KEY"]`, `os.environ["GEMINI_API_KEY"]`).
- Remove keys from the repo and rotate them.
- Optionally switch to explicit service account JSON (or workload identity on Cloud Run) instead of user credentials.
