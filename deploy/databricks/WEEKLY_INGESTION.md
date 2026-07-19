# Weekly Databricks Volume Ingestion

This pipeline scans a Databricks Volume for new or updated files, processes only changed files, and writes retrieval artifacts to a shared output Volume. The deployed Databricks App should read the same output Volume through `DATABRICKS_OUTPUT_VOLUME`.

## What It Does

```text
Databricks Volume input
  -> incremental manifest check
  -> PDF OCR/layout extraction
  -> video frame OCR extraction
  -> plain text ingestion
  -> optional Office conversion to PDF when LibreOffice is available
  -> chunks
  -> Databricks embedding endpoint
  -> embedded JSONL
  -> LanceDB rag_chunks index
  -> shared output Volume
```

The manifest is stored at:

```text
<DATABRICKS_OUTPUT_VOLUME>/ingestion/volume-manifest.json
```

The job is restartable. Successful files are skipped until their size, modified time, or SHA-256 changes.

## Required Runtime Config

Use Databricks app/job environment variables or secrets:

```text
APP_ENV=databricks
LLM_PROVIDER=databricks
DATABRICKS_HOST=https://dbc-4d180757-761e.cloud.databricks.com
DATABRICKS_TOKEN=<secret>
DATABRICKS_CHAT_ENDPOINT=databricks-claude-sonnet-4
DATABRICKS_EMBEDDING_ENDPOINT=databricks-bge-large-en
DATABRICKS_DATA_VOLUME=/Volumes/<catalog>/<schema>/<input_volume>
DATABRICKS_OUTPUT_VOLUME=/Volumes/<catalog>/<schema>/<output_volume>/insight-copilot-output
AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT=<secret or env>
AZURE_DOCUMENT_INTELLIGENCE_KEY=<secret>
```

The app must also have `DATABRICKS_OUTPUT_VOLUME` set, otherwise it will read the packaged demo `output/` folder instead of the weekly pipeline output.

The scheduled job uses `--databricks-models-only`, which forces `LLM_PROVIDER=databricks`, uses the Databricks embedding endpoint for chunks, and disables OpenAI-only ingestion steps. In this mode `OPENAI_API_KEY` is not required or used.

For this workspace, the discovered managed Volume is:

```text
DATABRICKS_DATA_VOLUME=/Volumes/insight-copilot/bronze/shell-bronze-insight-copilot
DATABRICKS_OUTPUT_VOLUME=/Volumes/insight-copilot/bronze/shell-bronze-insight-copilot/_insight_copilot_output
```

## Supported Inputs

- PDFs: Azure Document Intelligence OCR/layout extraction. OpenAI page vision is skipped in Databricks-only mode.
- Videos: frame OCR and timestamped chunks. OpenAI audio transcription and frame vision are skipped in Databricks-only mode.
- Text-like docs: `.txt`, `.md`, `.csv`, `.json`, `.jsonl`, `.html`, `.log`.
- Office docs: `.docx`, `.pptx`, `.xlsx`, legacy Office files if `libreoffice` or `soffice` exists on the job cluster.

Video processing uses system `ffmpeg` when present and falls back to the `imageio-ffmpeg` Python package on Databricks serverless. `ffprobe` is optional because the pipeline can parse basic metadata through `ffmpeg`. Office conversion requires LibreOffice.

## Local Smoke Test

This checks scanning and change detection without OCR/model calls:

```powershell
.venv\Scripts\python.exe -m app.pipelines.databricks_volume_ingest `
  --input-volume data `
  --output-root output `
  --dry-run `
  --max-files 5
```

## Databricks Dry Run

Upload the repo to:

```text
/Workspace/Shared/insight-copilot
```

Then run:

```bash
python /Workspace/Shared/insight-copilot/engine/app/pipelines/databricks_volume_ingest.py \
  --input-volume /Volumes/<catalog>/<schema>/<input_volume> \
  --output-root /Volumes/<catalog>/<schema>/<output_volume>/insight-copilot-output \
  --databricks-models-only \
  --dry-run
```

## Databricks Health Checks

Run diagnostics to confirm the job identity can see the Volume, Databricks serving auth, Azure OCR config, and Databricks model mode:

```powershell
databricks jobs submit --json "@deploy/databricks/jobs/volume_ingestion_diagnostics_submit.json" --profile dbc-insight-copilot-deployer
```

Expected diagnostic signals:

```text
llm_provider: databricks
databricks_models_only: true
databricks_embedding_endpoint: databricks-bge-large-en
DATABRICKS_RUNTIME_AUTH: true
OPENAI_API_KEY: false
```

Run the smoke test to process one tiny file, call the Databricks embedding endpoint, and rebuild LanceDB:

```powershell
databricks jobs submit --json "@deploy/databricks/jobs/volume_ingestion_smoke_submit.json" --profile dbc-insight-copilot-deployer
```

Expected smoke output includes `failed: []`, `active_embedding_model: databricks-bge-large-en`, and `target_vector_length: 1024`.

## Create Or Update The Weekly Job

`deploy/databricks/jobs/weekly_volume_ingestion_job.json` is already configured for the workspace Volume above. Change the paths only if you move the data.

Then create the Databricks Workflow:

```powershell
$dbx = "C:\Users\Ankur_Kumar\AppData\Local\Microsoft\WinGet\Packages\Databricks.DatabricksCLI_Microsoft.Winget.Source_8wekyb3d8bbwe\databricks.exe"
$profile = "dbc-4d180757-761e"
& $dbx jobs create --json "@deploy/databricks/jobs/weekly_volume_ingestion_job.json" --profile $profile
```

For updates, use `jobs reset` with the job id:

```powershell
& $dbx jobs reset <job-id> --json "@deploy/databricks/jobs/weekly_volume_ingestion_job.json" --profile $profile
```

## Run Once Manually

After creating the job:

```powershell
& $dbx jobs run-now <job-id> --profile $profile
```

Watch the run in Databricks Workflows. When it succeeds, the app should read the newly generated chunks, embeddings, and LanceDB index from the shared output Volume.

## Operational Notes

- Keep `max_concurrent_runs` at `1` to avoid two jobs rebuilding the same LanceDB table at once.
- Use `--dry-run` first for any new Volume.
- Use `--max-files 1` for the first real run if OCR credentials or video dependencies are still being validated.
- Use `--force` only when you intentionally want to reprocess all matching files.
- The default schedule is Sunday 02:00 Asia/Kolkata.
