# RAG Engine Operations

## User-visible progress

The engine keeps technical search and ranking stages in response telemetry. Streaming
clients receive only these stable, user-facing phases:

1. Getting ready
2. Understanding your question
3. Finding the most relevant information
4. Reviewing the best supporting information
5. Preparing a clear answer
6. Adding helpful next questions
7. Your answer is ready

Do not expose implementation terms in SSE messages. `public_progress_payload()` in
`backend/app/api/main.py` is the translation boundary.

## Quality controls

- Hybrid retrieval combines lexical and semantic rankings for every provider.
- Document and source-type filters are enforced before prompt construction.
- Context selection applies chunk and token limits, per-document diversity,
  near-duplicate removal, and adjacent evidence expansion.
- Citations are checked against source indexes, PDF pages, video timestamps, and text.
- Confidence is calibrated from model confidence, citation coverage, and source strength.
- Source text is untrusted. The answer prompt prohibits following source instructions.

## Runtime diagnostics

`GET /diagnostics/runtime` includes `rag_metrics` for the latest 500 requests:

- request count and response modes
- cache hits, fallbacks, and errors
- p50, p95, and maximum engine latency
- model calls and tokens when reported by the provider

Each response contains `telemetry` with per-stage timing, provider, mode, cache/fallback
flags, source count, error type, and usage. It is also stored in message metadata.

## Evaluation and release gates

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_retrieval.py `
  --method bm25 --top-k 10 `
  --output output\evaluation\retrieval-report.json

.\.venv\Scripts\python.exe scripts\evaluate_rag.py `
  --output output\evaluation\rag-report.json

.\.venv\Scripts\python.exe scripts\check_release_gates.py `
  --retrieval-report output\evaluation\retrieval-report.json `
  --rag-report output\evaluation\rag-report.json
```

Default gates:

- retrieval hit rate >= 0.85
- mean reciprocal rank >= 0.65
- expected-document recall >= 0.75
- grounded-answer rate >= 0.90
- citation precision >= 0.95
- p95 response latency <= 8 seconds

The end-to-end run calls the configured model and may incur provider cost.

## Load testing

Use a dedicated test identity:

```powershell
$env:LOAD_TEST_BEARER_TOKEN = "<test token>"
.\.venv\Scripts\python.exe scripts\load_test_chat.py `
  --base-url http://127.0.0.1:8000 `
  --requests 20 --concurrency 5
```

The command fails when success rate is below 98%.

## Alert recommendations

| Signal | Warning | Critical | First action |
|---|---:|---:|---|
| Chat error rate | 2% / 10 min | 5% / 5 min | Check provider and DB health |
| Fallback rate | 10% / 15 min | 25% / 10 min | Check embeddings and LanceDB |
| Insufficient answers | 20% / 30 min | 40% / 15 min | Validate index/model compatibility |
| p95 latency | 8 seconds | 15 seconds | Inspect stage timings |
| Citation rejection | 5% | 15% | Inspect prompt/schema drift |
| Provider circuit open | any | repeated | Check permissions and endpoint status |

## Rollback

1. Record the last successful Git revision and Databricks deployment ID.
2. If quality or security gates fail, redeploy the last successful snapshot.
3. Verify `/health`, `/diagnostics/runtime`, one PDF question, and one timestamped
   video question after rollback.
