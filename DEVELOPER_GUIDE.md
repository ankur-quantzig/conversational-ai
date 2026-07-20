# Insight Copilot / Shell Conversational AI — Developer & AI-Agent Guide

> A single onboarding document for developers **and** the AI coding agents they use.
> Read this before making changes. It captures how the system fits together, how to run and
> deploy it, and the non-obvious gotchas that have already bitten us (and that a build/typecheck
> will **not** catch).

**App:** a conversational AI ("Insight Copilot", branded in-product as **Shell Conversational AI**)
that answers natural-language questions over an indexed corpus of **documents and videos** using a
RAG pipeline. Deployed as a **Databricks App**.

- **Live app:** https://insight-copilot-7474659683601153.aws.databricksapps.com
- **Repo:** https://github.com/ankur-quantzig/conversational-ai
- **Default branch:** `main`

---

## 1. Architecture at a glance

```
React UI  ─►  FastAPI backend  ─►  RAG engine  ─►  LanceDB (vector search)  ─►  LLM answer
 (ui/)         (backend/)          (engine/)        + keyword fallback          (Databricks or OpenAI)
                                                                                      │
                                                              PostgreSQL / SQLite (chat history)
```

| Folder | Role |
|--------|------|
| `ui/` | React + Vite frontend (chat UI, citations, text/video tabs, streaming, sessions, print/PDF). Single big component in `ui/src/App.jsx`. |
| `backend/` | FastAPI API: routes, role-based auth, quotas, rate limiting, guardrails, audit log, DB persistence. |
| `engine/` | RAG: retrieval, model clients, chunking, embeddings, quality enrichment, and ingestion pipelines (PDF / video / medallion / Databricks volume). |
| `deploy/databricks/` | Databricks Apps deploy config (`app.yaml`), startup script, ingestion jobs. |
| `docs/` | Architecture diagrams (`.drawio`). |
| `tests/` | Smoke tests (thin — see §11). |
| `scripts/` | One-off data/tooling scripts. |

### The split-package import model (read this first)
`backend/` and `engine/` are **linked through the `app.*` namespace package**. `sitecustomize.py`
adds both `backend/` and `engine/` to the import path. **Always run Python commands from the repo
root** so imports like `app.api.main`, `app.rag.answer`, `app.clients.lancedb_store` resolve.

```bash
# correct — from repo root
python -m app.pipelines.build_combined_lancedb_index
# or set it explicitly
PYTHONPATH=backend:engine:. python -m app.rag.query_rag "..."
```

---

## 2. Local development

Prereqs: Python 3.11, Node 20+ (repo has been run with Node 24). A Python venv at `.venv/`.

**Backend** (from repo root):
```powershell
.\.venv\Scripts\python.exe -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000
```

**Frontend** (separate terminal):
```powershell
$env:VITE_API_BASE_URL = "http://127.0.0.1:8000"
npm --prefix ui run dev -- --host 127.0.0.1
```

Open http://127.0.0.1:5173 · health check http://127.0.0.1:8000/health

> **Vite first-load quirk:** if the dev page opens blank on first load, hard-refresh
> (`Ctrl+Shift+R`). Vite pre-bundles deps on first hit and forces one reload.

**Docker (full stack incl. Postgres):** `docker compose up --build` → UI :8080, API :8000, Postgres :5432.

---

## 3. Configuration

Config is read via `backend/app/config.py`, which loads a local `.env` (for local dev) and reads
environment variables. On Databricks, config comes from `deploy/databricks/app.yaml` `env:`.

Key variables:

| Variable | Meaning |
|----------|---------|
| `APP_ENV` | `local` / `databricks`. Controls env-specific behavior (e.g. forced UI rebuild on Databricks). |
| `LLM_PROVIDER` | `openai` (default) or `databricks`. Selects the answer/embedding pipeline. |
| `DATABRICKS_HOST` / `DATABRICKS_TOKEN` | Workspace URL + PAT (local/CI). **Not set in app.yaml** — see §9 auth. |
| `DATABRICKS_CHAT_ENDPOINT` | e.g. `databricks-claude-sonnet-4`. |
| `DATABRICKS_EMBEDDING_ENDPOINT` | e.g. `databricks-bge-large-en` (**1024-dim** — must match stored embeddings). |
| `DATABRICKS_VISION_ENDPOINT` / `DATABRICKS_TRANSCRIPTION_ENDPOINT` | Video frame/audio models. |
| `QUALITY_ENRICHMENT_PROVIDER` / `_MODEL` | Retrieval-quality cleanup model. |
| `INSIGHT_OUTPUT_ROOT` | Where the app reads/writes `output/` (chunks, embeddings, LanceDB). See §7. |
| `DATABRICKS_DATA_VOLUME` / `DATABRICKS_OUTPUT_VOLUME` | UC Volume paths for ingestion data/artifacts. |
| `RETRIEVAL_TOP_K`, `RATE_LIMIT_PER_MINUTE`, `BASIC_USER_QUESTION_LIMIT` | Tuning/limits. |
| `POWER_USERS` / `BASIC_USERS` | Comma-separated emails → role + quota mapping. |

> **Never commit secrets.** `.env` is git-ignored **and** excluded from the Databricks sync
> (`deploy/databricks/.databricksignore`). `app.yaml` **is** committed — so never put a raw token in
> it; use OAuth (§9) or a Databricks secret reference.

---

## 4. RAG pipeline

```
extract (layout / transcript / OCR / visual)
  → chunk (DI-aware: headings, paragraphs, tables, transcript windows, visual summaries)
  → quality-enrich (preserve raw text, normalize Hinglish→English, apply glossary, score quality)
  → embed
  → store/query in LanceDB (output/vector_db/lancedb, table `rag_chunks`)
  → retrieve top-k
  → generate structured answer (bullets + citations + confidence)
```

- **Retrieval** ([`engine/app/rag/retriever.py`](engine/app/rag/retriever.py)): embeds the query, runs
  LanceDB vector search. If LanceDB/LLM is unavailable, the API **falls back to local keyword search**.
- **Answer generation** ([`engine/app/rag/answer.py`](engine/app/rag/answer.py)): question rephrasing/
  clarification, sub-question generation, structured answer (heading, bullets, citations,
  `confidence_score`), follow-up questions, optional response diagram.
- **Confidence gate:** if the model isn't confident the evidence answers the question, it returns
  exactly: *"I could not find enough information about this in the indexed documents/videos. Can you
  specify the document, video, or topic?"* (see §10 — this is also what you see when retrieval is
  broken/empty).
- **Video chunks** carry `start_time` / `end_time` / labels / `key_frame_path` so the UI can deep-link
  to exact source time ranges.

The UI has two retrieval modes: **Talk to text** (PDF/doc chunks) and **Talk to video** (timestamped
video chunks).

---

## 5. Data & index locations (important)

`output_dir()` in [`engine/app/utils/files.py`](engine/app/utils/files.py) resolves the output root in
this precedence: **`INSIGHT_OUTPUT_ROOT` → `DATABRICKS_OUTPUT_VOLUME` → `OUTPUT_ROOT` → `./output`.**

On Databricks, `app.yaml` sets `INSIGHT_OUTPUT_ROOT=output`, so **the app reads embeddings and the
LanceDB index from the deployed source folder's `output/`**, *not* the UC Volume. The weekly ingestion
job writes to the Volume; artifacts must reach `output/` for the app to serve them.

Layout under the output root:
```
output/embeddings/*-embedded.jsonl        # embedded chunks (each row: content + embedding + metadata)
output/vector_db/lancedb/rag_chunks.lance # LanceDB table the app queries
output/chunks/, output/quality/, ...      # intermediate pipeline outputs
```

> **Empty index ⇒ "insufficient evidence".** If `rag_chunks` is empty (or the query-embedding call
> fails), retrieval returns nothing and every question yields the insufficient-evidence message in a
> few **milliseconds** (no LLM call). A real answer takes **seconds**. Use response latency as your
> first diagnostic signal.

Startup ([`deploy/databricks/start_databricks_app.py`](deploy/databricks/start_databricks_app.py)):
restores packaged artifacts → **builds the LanceDB table only if it doesn't already exist** → rebuilds
the frontend from source → starts uvicorn. Note: it **won't rebuild an existing-but-empty table**; to
force a rebuild, remove `output/vector_db/lancedb` before deploy.

---

## 6. Backend API surface

Base: FastAPI app in [`backend/app/api/main.py`](backend/app/api/main.py).

```
GET    /health                      # {ok, env, chunks}  ← chunks = indexed row count
GET    /me                          # user identity, role, question_quota
GET    /diagnostics/runtime
GET    /documents
GET    /sessions  ·  POST /sessions ·  PATCH/DELETE /sessions/{id}
GET    /sessions/{id}/messages ·  /export ·  /export.txt
POST   /chat                        # main Q&A
POST   /chat/stream                 # streaming Q&A (progress events)
POST   /messages/{id}/feedback ·  /share ·  /retry
GET    /shares/{share_id}
GET    /media/videos/{doc_id} ·  /media/video-clips/{doc_id}
```

Auth/quota/guardrails live in `backend/app/security/` (`auth.py`, `quota.py`, `rate_limit.py`,
`guardrails.py`, `audit.py`). Roles derive from `POWER_USERS` / `BASIC_USERS`.

---

## 7. Databricks deployment

The app is deployed by (1) syncing the repo to a workspace folder and (2) deploying the app from it.
The startup script **rebuilds the UI from source** on Databricks, so source-only changes take effect.

```powershell
$dbx = "<path>\databricks.exe"
# Auth: either a configured --profile, or env vars:
$env:DATABRICKS_HOST  = "<workspace-url>"
$env:DATABRICKS_TOKEN = "<pat>"

# 1) sync (incremental — omit --full unless you intend a full reconcile)
& $dbx sync . /Workspace/Shared/insight-copilot --exclude-from deploy/databricks/.databricksignore
# 2) deploy
& $dbx apps deploy insight-copilot --source-code-path /Workspace/Shared/insight-copilot --mode SNAPSHOT --auto-approve
# check
& $dbx apps get insight-copilot
```

> ⚠️ **`sync --full` reconciles the whole tree.** `.databricksignore` excludes `.env`, `.venv`,
> `node_modules`, `ui/dist`, etc., but **not** `output/`. Avoid syncing a local `output/` over the
> workspace's good index — prefer an incremental sync, or ensure your local `output/` matches intent.
> Binary LanceDB files can corrupt under partial overwrite.

> **Run the CLI from PowerShell, not Git Bash**, for `/Workspace/...` paths — Git Bash's POSIX path
> conversion mangles `/Workspace/...` into a Windows path and the sync fails.

### How the app authenticates to model endpoints (the subtle part)
A Databricks App is a plain uvicorn process — **no `dbutils`, no notebook token.** `app.yaml` sets no
`DATABRICKS_TOKEN`. So the serving client
([`engine/app/clients/databricks_model_serving.py`](engine/app/clients/databricks_model_serving.py))
mints an **OAuth token from the `DATABRICKS_CLIENT_ID` / `DATABRICKS_CLIENT_SECRET` that Databricks
Apps inject** for the app's service principal (OIDC client-credentials grant). If serving calls 401/403:
the app's **service principal needs `CAN QUERY`** on the Foundation Model endpoints.

---

## 8. Known gotchas & lessons (high-value for agents)

1. **Runtime-only JS errors pass `npm run build`.** A call to an undefined function (e.g. a helper
   deleted in a merge) builds fine but blanks the app at runtime with a `ReferenceError`. The UI has
   **no ESLint `no-undef`** yet — adding it would catch this class of bug. Always load the page after
   UI changes, don't trust the build alone.
2. **`App.jsx` is one large component.** Merges conflict often around the topbar/empty-chat. The app
   name is the `APP_NAME` constant — don't reintroduce hardcoded product names.
3. **Fast "insufficient evidence" = infra, not content.** Milliseconds ⇒ retrieval/embedding failed
   or index empty (see §5). Seconds ⇒ the model genuinely judged the evidence weak.
4. **Embedding dimensions must match.** Stored vectors and the query embedding endpoint must agree
   (bge-large-en = 1024). A mismatch makes a populated index return nothing.
5. **Answers are cached** (`_TtlCache`, 300s) keyed by question+scope. When testing a fix, start a
   **New chat** / vary the question, or you'll see a stale cached answer.
6. **The live app is behind Databricks SSO** — you can't curl its `/health` with a PAT. To validate
   the RAG pipeline, reproduce locally with `LLM_PROVIDER=databricks` + the same endpoints + the same
   embeddings, or test in the browser.

---

## 9. Testing & CI

- CI ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): repo-hygiene + secret scan → Python
  compile/import + smoke tests → UI production build.
- Tests are **thin** (a few smoke tests under `tests/`). There is **no** API/auth/RAG-quality coverage
  and no retrieval evaluation harness yet — treat adding these as high-value work.
- Before committing nontrivial changes: run the relevant smoke tests, build the UI, and **exercise the
  actual flow** (load the page / hit the endpoint), because build ≠ runtime here.

---

## 10. Conventions for AI agents working in this repo

- **Run Python from the repo root** (namespace package — §1). On Windows, prefer **PowerShell** for
  Databricks CLI and `/Workspace` paths.
- **Don't commit secrets;** `.env` and `app.yaml` are the sensitive spots (app.yaml is git-tracked).
- **Deploying is production + outward-facing** — confirm scope before `apps deploy`; it targets the
  live Shell app.
- **Verify at runtime,** not just via build/typecheck (§8.1).
- **Branch off `main`** for changes; open a PR rather than pushing straight to `main`.
- Keep new code in the style of the surrounding file (match naming, structure, comment density).
```

---

*Maintainers: keep this file current when the architecture, deploy flow, or a hard-won gotcha changes.
It is the shared context for humans and agents alike.*
