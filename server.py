from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai
from google.cloud import bigquery
from vertexai.preview.language_models import TextEmbeddingModel
import requests
import os
import json
from dotenv import load_dotenv
from google.oauth2 import service_account
import google.auth.transport.requests as google_auth_requests
from sklearn.feature_extraction.text import TfidfVectorizer  # new for hybrid search
from google.cloud import aiplatform_v1  # vertex vector search


# --- Config ---
load_dotenv()  # load variables from .env if present

GENAI_API_KEY = os.getenv("GENAI_API_KEY")  # for google.genai (Gemma)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # for Gemini REST
BQ_PROJECT_ID = os.getenv("BQ_PROJECT_ID", "")  # BigQuery project id

# Discovery Engine defaults (can be overridden per-request)
DISCOVERY_PROJECT_ID = os.getenv("DISCOVERY_PROJECT_ID", "")
DISCOVERY_ENGINE_ID = os.getenv("DISCOVERY_ENGINE_ID", "")
DISCOVERY_LOCATION = os.getenv("DISCOVERY_LOCATION", "global")
DISCOVERY_COLLECTION = os.getenv("DISCOVERY_COLLECTION", "default_collection")
DISCOVERY_SERVING_CONFIG = os.getenv("DISCOVERY_SERVING_CONFIG", "default_search")
def _resolve_existing_path(basename: str, override: str | None = None):
    """Return the first existing path among sensible locations.
    Order: explicit override -> alongside this file -> CWD -> parent of this file.
    """
    candidates = []
    if override:
        candidates.append(override)
    base_dir = os.path.dirname(__file__)
    candidates.extend([
        os.path.join(base_dir, basename),
        os.path.join(os.getcwd(), basename),
        os.path.join(os.path.dirname(base_dir), basename),
    ])
    for p in candidates:
        try:
            if p and os.path.exists(p):
                return p
        except Exception:
            pass
    return candidates[0] if candidates else None

DISCOVERY_SA_PATH = _resolve_existing_path(
    "vertexmanager-key.json", os.getenv("DISCOVERY_SA_PATH")
)

# --- Helpers ---
def getGemmaResponse(prompt):
    if not GENAI_API_KEY:
        raise RuntimeError("Missing GENAI_API_KEY in environment/.env")
    client = genai.Client(api_key=GENAI_API_KEY)
    response = client.models.generate_content(
        model="gemma-3-27b-it",
        contents=prompt,
    )
    return response.text


def getGeminiResponse(prompt: str) -> str:
    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"
    headers = {
        "Content-Type": "application/json",
        "X-goog-api-key": GEMINI_API_KEY or "",
    }
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        result = response.json()
        try:
            return result["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError):
            return "No text found in response."
    else:
        return f"Error {response.status_code}: {response.text}"


# --- Initialize clients like app.py ---
if not BQ_PROJECT_ID:
    # Allow ADC to infer default project; else require explicit project id
    bq_client = bigquery.Client()
else:
    bq_client = bigquery.Client(project=BQ_PROJECT_ID)
embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-004")

# --- Hybrid Search Globals ---
VERTEX_API_ENDPOINT = os.getenv("VERTEX_API_ENDPOINT", "")  # optional override
VERTEX_INDEX_ENDPOINT = os.getenv("VERTEX_INDEX_ENDPOINT", "")  # projects/..../indexEndpoints/ID
VERTEX_DEPLOYED_INDEX_ID = os.getenv("VERTEX_DEPLOYED_INDEX_ID", "")  # deployed index id
SERVICE_ACCOUNT_KEY_PATH = _resolve_existing_path(
    "vertexmanager-key.json", os.getenv("VERTEX_SA_PATH")
)

vectorizer = TfidfVectorizer(max_features=5000, stop_words="english")
_tfidf_fitted = False
_corpus_loaded = False
_corpus_texts = []

def _load_corpus():
    """Load local corpus for TF-IDF fitting. Expects gemini_cleaned_text.json mapping id->text."""
    global _corpus_loaded, _corpus_texts
    if _corpus_loaded:
        return
    corpus_path = _resolve_existing_path("gemini_cleaned_text.json")
    if not corpus_path or not os.path.exists(corpus_path):
        return  # handled later if needed
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        _corpus_texts = list(data.values())
        _corpus_loaded = True
    except Exception:
        # Leave as not loaded; error surfaced on use
        pass

def _fit_tfidf_if_needed():
    global _tfidf_fitted
    if _tfidf_fitted:
        return
    _load_corpus()
    if not _corpus_texts:
        raise RuntimeError("Corpus file gemini_cleaned_text.json not found or unreadable in expected locations (cwd, app dir, parent). Set absolute path if needed.")
    vectorizer.fit(_corpus_texts)
    _tfidf_fitted = True

def _sparse_embedding(text: str):
    if not _tfidf_fitted:
        raise RuntimeError("TF-IDF not fitted yet.")
    tfidf_vector = vectorizer.transform([text])
    values = [float(v) for v in tfidf_vector.data]
    dims = [int(i) for i in tfidf_vector.indices]
    return {"values": values, "dimensions": dims}

def _dense_query_embedding(text: str):
    if not GEMINI_API_KEY:
        raise RuntimeError("Missing GEMINI_API_KEY for embedding model.")
    client_local = genai.Client(api_key=GEMINI_API_KEY)
    response = client_local.models.embed_content(
        model="models/embedding-001",
        contents=[text],
        config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768},
    )
    if hasattr(response, "embeddings"):
        return response.embeddings[0].values
    return response.embedding.values

def _vertex_match_client():
    # Try service account key else ADC
    if SERVICE_ACCOUNT_KEY_PATH and os.path.exists(SERVICE_ACCOUNT_KEY_PATH):
        creds = service_account.Credentials.from_service_account_file(
            SERVICE_ACCOUNT_KEY_PATH,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
    else:
        creds = None  # let library pick ADC
    client_options = {}
    if VERTEX_API_ENDPOINT:
        client_options["api_endpoint"] = VERTEX_API_ENDPOINT
    return aiplatform_v1.MatchServiceClient(client_options=client_options, credentials=creds)


app = Flask(__name__)

# Configure CORS. Allow origins from CORS_ORIGINS env (comma separated) or default to '*'.
_cors_origins = os.getenv("CORS_ORIGINS", "*")
origins_list = [o.strip() for o in _cors_origins.split(",") if o.strip()] or ["*"]
CORS(app, resources={r"/*": {"origins": origins_list}}, supports_credentials=False)


@app.get("/health")
def health():
    return {"status": "ok"}, 200


@app.post("/discovery/search")
def discovery_search():
    """Proxy route to Google Discovery Engine Search API.

    Body JSON fields:
    - query: string (required)
    - Optional overrides: project_id, engine_id, discovery_location, collection, serving_config
    - Any other fields will be forwarded to the Discovery Engine search payload.
    Auth:
    - Uses GOOGLE_APPLICATION_CREDENTIALS service account if set; otherwise attempts ADC.
    """
    data = request.get_json(silent=True) or {}
    if not isinstance(data, dict):
        return jsonify(error="Invalid JSON body"), 400

    # Required query
    if not data.get("query"):
        return jsonify(error="'query' is required in body"), 400

    # Obtain access token using the Discovery SA file; also capture project_id from that file
    scopes = ["https://www.googleapis.com/auth/cloud-platform"]
    try:
        if not DISCOVERY_SA_PATH or not os.path.exists(DISCOVERY_SA_PATH):
            return jsonify(error="Discovery service account file not found",
                           detail=f"Tried at {DISCOVERY_SA_PATH}. Set DISCOVERY_SA_PATH env or place vertexmanager-key.json in project root or alongside server.py."), 500
        with open(DISCOVERY_SA_PATH, "r", encoding="utf-8") as f:
            sa_info = json.load(f)
        creds = service_account.Credentials.from_service_account_info(sa_info, scopes=scopes)
        auth_req = google_auth_requests.Request()
        creds.refresh(auth_req)
        token = creds.token
        sa_project = sa_info.get("project_id")
    except Exception as e:
        return jsonify(error="Failed to load Discovery service account or obtain token", detail=str(e)), 500

    project_id = data.get("project_id") or DISCOVERY_PROJECT_ID or sa_project
    engine_id = data.get("engine_id") or DISCOVERY_ENGINE_ID
    location = data.get("discovery_location") or DISCOVERY_LOCATION
    collection = data.get("collection") or DISCOVERY_COLLECTION
    serving_config = data.get("serving_config") or DISCOVERY_SERVING_CONFIG

    missing = []
    if not project_id:
        missing.append("project_id")
    if not engine_id:
        missing.append("engine_id")
    if missing:
        hint = "Set env (e.g., DISCOVERY_ENGINE_ID) or pass in request body." if "engine_id" in missing else "Provide required identifier(s)."
        return jsonify(
            error=f"Missing required field(s): {', '.join(missing)}",
            hint=hint,
            resolved={
                "project_id": project_id,
                "engine_id": engine_id,
                "location": location,
                "collection": collection,
                "serving_config": serving_config,
                "sa_path": DISCOVERY_SA_PATH,
            },
        ), 400

    url = (
        f"https://discoveryengine.googleapis.com/v1alpha/projects/{project_id}/"
        f"locations/{location}/collections/{collection}/engines/{engine_id}/"
        f"servingConfigs/{serving_config}:search"
    )

    # Build payload by forwarding all fields except our routing overrides
    exclude_keys = {"project_id", "engine_id", "discovery_location", "collection", "serving_config"}
    payload = {k: v for k, v in data.items() if k not in exclude_keys}

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }

    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=60)
    except requests.RequestException as re:
        return jsonify(error="Network error calling Discovery Engine", detail=str(re)), 502

    # Relay response
    try:
        body = resp.json()
    except ValueError:
        body = resp.text
    return jsonify(status=resp.status_code, url=url, request=payload, response=body), resp.status_code


@app.post("/query")
def query():
    data = request.get_json(silent=True) or {}
    query_text = data.get("question") or data.get("query") or ""
    if not query_text:
        return jsonify(error="Provide 'question' in JSON body."), 400

    wrapper_prompt = f"""
You are an advanced research analyst AI with expertise in geopolitics, economics, technology, cybersecurity, defense, urban development, and industry-specific domains.

Available datasets for reference:
- Bharat Matters
- Bilateral Ties
- DRDO
- Geopolitics
- Google Products
- India in Business
- IndiaAI reports (National Intelligence strategies for AI)
- Industry Developments (Agriculture/Fishery, AI_development, biology, economics, fashion, military, finance, health, mining, pharmaceuticals, trade, semiconductors, culture, hardware, cybersecurity, tourism)
- Ministry of External Affairs (Annual Reports, Other publications)
- MeiTy
- Ministry of Defence reports
- NASSCOM
- Niti Aayog
- Work Culture in other nations

Task:
1. Take the naive user query below.
2. Expand and enrich it into a **comprehensive multi-dimensional research query** of around **700 words**.
3. Look into the provided user prompt and select certain categories which align with the user request query:
   - **Urban development & regional dynamics** (cities like Chennai, Kolkata, Bengaluru, Delhi, etc.)
   - **Technology & AI** (IndiaAI, semiconductors, Google Products, NASSCOM, Niti Aayog)
   - **Cybersecurity & ransomware** (Industry Developments: cybersecurity, MeiTy, DRDO reports)
   - **Company portfolio analysis**
   
    Once a category is decided, start embedding more words related to that category. For example, if AI is selected,
   embed words like LLM, compute, mistral, deepseek, etc.
   
   It is compulsory for you to include embedding words from all three of the below categories in the answer as well
   
   - **Economics & trade** (India in Business, agriculture, pharmaceuticals, finance)
   - **Geopolitics & bilateral ties** (Bharat Matters, Bilateral Ties, Ministry of External Affairs, Niti Aayog)
   - **Defense & strategic considerations** (DRDO, Ministry of Defence reports)
   
  
   
4. Reference multiple datasets when building the enriched version.
5. The final enriched query should resemble a **professional research problem statement or policy question**, not just keywords.

Constraints:
- Ensure ~100 words with well-structured paragraphs.
- Keep the tone analytical, policy-oriented, and grounded in the datasets.
- Highlight interconnections: e.g., how urban vulnerabilities (Chennai floods, Kolkata port trade routes) intersect with AI deployment, ransomware risks, and global economic dependencies.

Naive User Query: {query_text}

Output:
Provide the **100-word enriched query** integrating cities, AI, cybersecurity, ransomware, economics, geopolitics, and datasets, ready for downstream BigQuery knowledge analysis.
"""
    try:
        enriched_query = getGemmaResponse(wrapper_prompt)

        # Generate embedding for the enriched query
        query_embedding = embedding_model.get_embeddings([enriched_query])[0].values
        embedding_str = "[" + ",".join(map(str, query_embedding)) + "]"

        # SQL (same as in app.py)
        sql = f"""
WITH query_vector AS (
  SELECT ARRAY<FLOAT64>{embedding_str} AS q_embedding
)
SELECT
  t.chunk_text,
  (SELECT SUM(q_val * t_val)
   FROM UNNEST(q.q_embedding) AS q_val WITH OFFSET
   JOIN UNNEST(t.embedding) AS t_val WITH OFFSET USING(OFFSET))
  /
  (SQRT((SELECT SUM(POW(q_val,2)) FROM UNNEST(q.q_embedding) AS q_val)) *
   SQRT((SELECT SUM(POW(t_val,2)) FROM UNNEST(t.embedding) AS t_val))) AS cosine_similarity
FROM `genai-knowledge-graph.content_pipeline.raw_documents_direct` AS t
CROSS JOIN query_vector AS q
ORDER BY cosine_similarity DESC
LIMIT 5
"""
        # Run query
        query_job = bq_client.query(sql)
        result = query_job.result()

        total_data = ""
        for row in result:
            total_data += row.chunk_text

        final_prompt = f"""
Context: You have access to a set of high-quality resources, including top-level government reports, defence analyses, think tank studies, research papers, and other authoritative sources.

Role: You are an expert in the field relevant to the question, capable of analyzing complex data and synthesizing it into a clear, comprehensive, and professional answer.

Audience: The answer should be suitable for policymakers, researchers, university students or professionals who require in-depth, evidence-based insights to plan their career paths.

Task: Carefully read and understand all the provided resources. Extract every relevant fact, statistic, research finding, report analysis, bibliographic reference, and perspective from the data. Use this information to craft a detailed, authoritative, and well-structured response to the given question. Ensure all statements are precise, factual, and supported by the sources.

Format:
- Present your answer in coherent, professional sentences.
- Include citations, percentages, and future projections wherever mentioned in the resources.
- Maintain proper grammar and readability.
- Organize the answer logically with sections or bullet points if necessary to convey clarity.
-Ensure to give priority to statistics

Question: {query_text}

Resources: {total_data}

You must use the provided resources as much as possible to give an in-depth answer that not only answers user request, but also goes
deeper than what they had expected. If it is impossible to use, quietly ignore them
We are on a professional platform and must not hint at any irrelavance of underlying data. Avoid any data that damages
India's reputation like "covert Indian operations targeting government critics".

After this, add in your own background knowledge to answer the question effectively

Instruction: Generate a comprehensive, expert-level answer to the above question based entirely on the provided resources, demonstrating deep understanding and thorough analysis.

Further notes: Avoid statements in the final answer like :

*(Avoid direct mentions in the provided text. However, the .... original prompt would be relevant here, if accessible.)

Instead, fill those gaps with your own knowledge
"""
        answer = getGeminiResponse(final_prompt)

        return jsonify(
            question=query_text,
            enriched_query=enriched_query,
            answer=answer,
        )
    except Exception as e:
        return jsonify(error="Internal error", detail=str(e)), 500


@app.post("/hybrid_search")
def hybrid_search():
    """Hybrid dense + sparse vector search with Vertex AI Index + RAG answer.

    Body JSON:
      question: str (required)
      neighbor_count: int (optional, default 10)
      alpha: float (0..1, dense weight for RRF, default 0.8)
    Environment required:
      VERTEX_INDEX_ENDPOINT, VERTEX_DEPLOYED_INDEX_ID (and optionally VERTEX_API_ENDPOINT)
      gemini_cleaned_text.json file for TF-IDF corpus.
    """
    body = request.get_json(silent=True) or {}
    question = body.get("question")
    if not question:
        return jsonify(error="'question' is required."), 400
    neighbor_count = int(body.get("neighbor_count", 10))
    alpha = float(body.get("alpha", 0.8))
    try:
        _fit_tfidf_if_needed()
    except Exception as e:
        return jsonify(error="TF-IDF setup failed", detail=str(e)), 500
    if not VERTEX_INDEX_ENDPOINT or not VERTEX_DEPLOYED_INDEX_ID:
        return jsonify(error="Missing Vertex index config", detail="Set VERTEX_INDEX_ENDPOINT and VERTEX_DEPLOYED_INDEX_ID env vars."), 500
    try:
        dense_vec = _dense_query_embedding(question)
    except Exception as e:
        return jsonify(error="Dense embedding failed", detail=str(e)), 500
    try:
        sparse = _sparse_embedding(question)
    except Exception as e:
        return jsonify(error="Sparse embedding failed", detail=str(e)), 500
    # Build request
    try:
        datapoint = aiplatform_v1.IndexDatapoint(
            feature_vector=dense_vec,
            sparse_embedding=aiplatform_v1.IndexDatapoint.SparseEmbedding(
                values=sparse["values"], dimensions=sparse["dimensions"]
            ),
        )
        query_obj = aiplatform_v1.FindNeighborsRequest.Query(
            datapoint=datapoint,
            neighbor_count=neighbor_count,
            rrf=aiplatform_v1.FindNeighborsRequest.Query.RRF(alpha=alpha),
        )
        request_v = aiplatform_v1.FindNeighborsRequest(
            index_endpoint=VERTEX_INDEX_ENDPOINT,
            deployed_index_id=VERTEX_DEPLOYED_INDEX_ID,
            queries=[query_obj],
            return_full_datapoint=False,
        )
        match_client = _vertex_match_client()
        response_v = match_client.find_neighbors(request_v)
    except Exception as e:
        return jsonify(error="Vertex find_neighbors failed", detail=str(e)), 500
    # Load corpus mapping id->text
    corpus_path = _resolve_existing_path("gemini_cleaned_text.json")
    try:
        with open(corpus_path, "r", encoding="utf-8") as f:
            corpus_data = json.load(f)
    except Exception as e:
        return jsonify(error="Failed to load corpus for RAG", detail=f"{e}. Expected gemini_cleaned_text.json near app or project root."), 500
    neighbors_out = []
    collected_texts = []
    for nset in response_v.nearest_neighbors:
        for n in nset.neighbors:
            dp_id = n.datapoint.datapoint_id
            distance = getattr(n, "distance", None)
            raw_text = corpus_data.get(dp_id, "")
            snippet = raw_text[:400].strip() if raw_text else ""
            neighbors_out.append({"id": dp_id, "distance": distance, "snippet": snippet})
            if raw_text:
                collected_texts.append(snippet)
    # RAG answer
    rag_answer = None
    normal_answer = None
    try:
        raw_info = "\n".join(collected_texts)
        rag_prompt = (
            f"Provided Question: {question}\n\n"
            f"Embed the provided context snippets into a credible analytical answer.\n\n"
            f"Raw Info:\n{raw_info}"
        )
        rag_answer = genai.Client(api_key=GEMINI_API_KEY).models.generate_content(
            model="gemini-2.0-flash",
            contents=rag_prompt,
        ).text.strip()
    except Exception as e:
        rag_answer = f"RAG generation failed: {e}"  # surface error inline
    try:
        normal_answer = genai.Client(api_key=GEMINI_API_KEY).models.generate_content(
            model="gemini-2.0-flash",
            contents=(
                f"Provided Question: {question}\n\nGive a professional research-quality answer."
            ),
        ).text.strip()
    except Exception as e:
        normal_answer = f"Normal generation failed: {e}"
    return jsonify(
        question=question,
        neighbor_count=neighbor_count,
        alpha=alpha,
        neighbors=neighbors_out,
        rag_answer=rag_answer,
        normal_answer=normal_answer,
    )


if __name__ == "__main__":
    # Simple run: source .venv/bin/activate && pip install Flask && python server.py
    app.run(host="0.0.0.0", port=8080, debug=True)
