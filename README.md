# Insight Copilot

Insight Copilot is a conversational AI tool that generates document insights from natural-language questions.

## Repo layout

```text
ui/       React/Vite frontend
backend/ FastAPI API, auth, sessions, quotas, and persistence
engine/  RAG retrieval, model clients, chunking, embeddings, and pipelines
```

The Python folders are linked through the `app.*` namespace package. Run Python commands from the repo root so `sitecustomize.py` adds both `backend/` and `engine/` to the import path.

## Project source scope

The POC is intentionally scoped to the two indexed inputs: the Conversational AI meeting transcript and its recording. This keeps the user-facing source list and retrieval results focused on the validated project data. Set `RAG_SOURCE_DOC_IDS` to a comma-separated list to change the scope; set it to `*` (or `all`) only when all packaged sources should be enabled.

## Run the app locally

Start the FastAPI backend:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

In another PowerShell window, start the React UI:

```powershell
$env:Path = "$PWD\tools\node-v24.18.0-win-x64;$env:Path"
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
.\tools\node-v24.18.0-win-x64\npm.cmd --prefix ui run dev -- --host 127.0.0.1
```

Open:

```text
http://127.0.0.1:5173
```

Health check:

```text
http://127.0.0.1:8000/health
```

## Run on Databricks

The Databricks app is named:

```text
insight-copilot
```

Current app URL:

```text
https://insight-copilot-7474659683601153.aws.databricksapps.com
```

Deploy updates from this repo:

```powershell
$dbx = "C:\Users\Ankur_Kumar\AppData\Local\Microsoft\WinGet\Packages\Databricks.DatabricksCLI_Microsoft.Winget.Source_8wekyb3d8bbwe\databricks.exe"
$profile = "dbc-4d180757-761e"
$appName = "insight-copilot"
$workspacePath = "/Workspace/Shared/insight-copilot"

& $dbx sync . $workspacePath --profile $profile --full --exclude-from deploy/databricks/.databricksignore
& $dbx apps deploy $appName --profile $profile --source-code-path $workspacePath --mode SNAPSHOT --auto-approve
```

Check deployment:

```powershell
& $dbx apps get $appName --profile $profile
```

Open:

```text
https://insight-copilot-7474659683601153.aws.databricksapps.com/health
https://insight-copilot-7474659683601153.aws.databricksapps.com/me
```

Databricks-specific details are in `deploy/databricks/README.md`.

Databricks hosting uses:

```text
LLM_PROVIDER=databricks
DATABRICKS_CHAT_ENDPOINT=databricks-claude-sonnet-4
DATABRICKS_EMBEDDING_ENDPOINT=databricks-bge-large-en
DATABRICKS_TRANSCRIPTION_ENDPOINT=databricks-gemini-3-5-flash
DATABRICKS_VISION_ENDPOINT=databricks-gemini-3-5-flash
QUALITY_ENRICHMENT_PROVIDER=databricks
QUALITY_ENRICHMENT_MODEL=databricks-claude-sonnet-4
```

This removes the need for `OPENAI_API_KEY` for chat answers, query embeddings, scheduled video audio transcription, visual extraction, and quality cleanup. The serving endpoint names must exist in your Databricks workspace.

## RAG pipeline

The current RAG path is:

1. Extract raw layout, transcript, OCR, and visual signals.
2. Build DI-aware chunks from headings, body paragraphs, tables, transcript windows, and visual summaries.
3. Enrich chunk quality by preserving raw text, normalizing mixed Hindi-English/Hinglish to clear English, applying the domain glossary, and scoring extraction quality.
4. Embed the enriched chunks.
5. Store/query vectors locally with LanceDB.
6. Generate answers from retrieved chunks.

Useful commands:

```bash
PYTHONPATH=backend:engine:. python3 -m app.pipelines.build_chunks \
  "output/document_intelligence/2002.08909v1 - REALM Retrieval-Augmented Language Model Pre-Training-document-intelligence.json" \
  --multimodal-json "output/multimodal_analysis/2002.08909v1 - REALM Retrieval-Augmented Language Model Pre-Training-multimodal-analysis.json"

PYTHONPATH=backend:engine:. python3 - <<'PY'
from pathlib import Path
from app.services.embed_chunks import embed_chunks_file
embed_chunks_file(Path("output/chunks/2002-08909v1-realm-retrieval-augmented-language-model-pre-training-chunks.jsonl"))
PY

PYTHONPATH=backend:engine:. python3 -m app.pipelines.build_lancedb_index \
  output/embeddings/2002-08909v1-realm-retrieval-augmented-language-model-pre-training-embedded.jsonl

# Or build one combined LanceDB index from every embedded file:
PYTHONPATH=backend:engine:. python3 -m app.pipelines.build_combined_lancedb_index

PYTHONPATH=backend:engine:. python3 -m app.pipelines.query_rag \
  "What is REALM and how does retrieval help it?"
```

LanceDB writes the local vector database to `output/vector_db/lancedb`. No cloud vector database or Docker service is required.

Embedding defaults:

```bash
OPENAI_EMBEDDING_MODEL=text-embedding-3-large
OPENAI_EMBEDDING_DIMENSIONS=3072
```

## Video insight pipeline

Videos are processed into timestamped multimodal chunks:

```text
video -> audio transcript -> sampled frames -> Azure DI frame OCR -> optional vision summaries -> timestamped chunks -> quality enrichment -> embeddings -> LanceDB
```

Requirements:

- `ffmpeg` and `ffprobe` available on the machine, or use the API Docker image.
- Azure Document Intelligence env vars for frame OCR.
- Databricks deployment uses `DATABRICKS_TRANSCRIPTION_ENDPOINT` for spoken audio, `DATABRICKS_VISION_ENDPOINT` for frame visual summaries, `QUALITY_ENRICHMENT_MODEL` for retrieval-quality cleanup, and `DATABRICKS_EMBEDDING_ENDPOINT` for embeddings.
- Local OpenAI mode can still use `OPENAI_API_KEY` for transcription, frame visual summaries, embeddings, and final answers.

Process every video in `data/Videos`, embed the chunks, and rebuild LanceDB:

```bash
PYTHONPATH=backend:engine:. python3 -m app.pipelines.process_video \
  "data/Videos" \
  --frame-interval-seconds 5 \
  --chunk-window-seconds 30 \
  --embed \
  --rebuild-index
```

Outputs are written to:

```text
output/videos/{video_id}/transcript.json
output/videos/{video_id}/frames/
output/videos/{video_id}/frame_ocr.json
output/videos/{video_id}/visual_analysis.json
output/chunks/{video_id}-video-chunks.jsonl
output/quality/{video_id}-video-quality-chunks.jsonl
output/embeddings/{video_id}-video-embedded.jsonl
```

Every video chunk includes:

```text
source_type=video
start_time
end_time
start_time_label
end_time_label
key_frame_path
transcript + Azure OCR text + visual summary
raw_content + quality score in metadata
```

## React chatbot + FastAPI

The React UI calls a FastAPI backend that uses the main RAG pipeline:

```text
React UI -> FastAPI -> LanceDB vector retrieval -> OpenAI answer generation -> PostgreSQL history
```

If LanceDB/OpenAI is unavailable, the API falls back to local keyword retrieval so the UI still responds.

The UI has two retrieval tabs:

- `Talk to text`: searches PDF/text chunks.
- `Talk to video`: searches timestamped video chunks and returns exact source time ranges.

Run the full container stack:

```bash
docker compose up --build
```

Open:

- React UI: http://localhost:8080
- FastAPI health: http://localhost:8000/health
- PostgreSQL: localhost:5432

The API creates these tables automatically on startup:

- `chat_sessions`
- `chat_messages`

Useful API routes:

```text
GET    /health
GET    /documents
GET    /sessions
GET    /sessions/{session_id}/messages
POST   /chat
DELETE /sessions/{session_id}
```

## Retrieval evaluation

The versioned seed set in `evaluation/golden_questions.v1.jsonl` covers PDF, video, and
cross-source questions. Run the provider-independent BM25 baseline from the repo root:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py `
  --method bm25 `
  --top-k 10 `
  --output output\evaluation\retrieval-baseline.json
```

The report includes hit rate, mean reciprocal rank, per-case latency, and the ranked
document IDs. Chat responses also include a `telemetry` object with pipeline provider,
mode, cache/fallback flags, source count, total time, and per-stage timings.

Production controls, load testing, alerts, and release gates are documented in
[`docs/RAG_ENGINE_OPERATIONS.md`](docs/RAG_ENGINE_OPERATIONS.md).
