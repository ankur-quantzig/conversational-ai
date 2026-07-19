from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SCRIPT_FILE = globals().get("__file__")
ROOT = Path(SCRIPT_FILE).resolve().parents[3] if SCRIPT_FILE else Path(os.getenv("INSIGHT_REPO_ROOT", "/Workspace/Shared/insight-copilot"))
for path in (ROOT / "backend", ROOT / "engine", ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from app.services.chunk_document import estimate_tokens, load_chunks_jsonl, slugify, stable_id, write_jsonl
from app.utils.files import data_root, display_path, output_dir
from app.utils.logging import dump_json


MANIFEST_VERSION = 1
DOCUMENT_EXTENSIONS = {".pdf", ".txt", ".md", ".markdown", ".csv", ".json", ".jsonl", ".html", ".htm", ".log"}
OFFICE_EXTENSIONS = {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v", ".mkv", ".webm"}
DEFAULT_INCLUDE = [f"**/*{extension}" for extension in sorted(DOCUMENT_EXTENSIONS | OFFICE_EXTENSIONS | VIDEO_EXTENSIONS)]
DEFAULT_EXCLUDE = [
    "**/.DS_Store",
    "**/~$*",
    "**/.trash/**",
    "**/_delta_log/**",
    "_insight_copilot_output/**",
    "**/_insight_copilot_output/**",
    "insight-copilot-output/**",
    "**/insight-copilot-output/**",
]
SECRET_ENV_KEYS = [
    "DATABRICKS_HOST",
    "DATABRICKS_TOKEN",
    "DATABRICKS_CHAT_ENDPOINT",
    "DATABRICKS_EMBEDDING_ENDPOINT",
    "DATABRICKS_TRANSCRIPTION_ENDPOINT",
    "LLM_PROVIDER",
    "VIDEO_TRANSCRIPTION_PROVIDER",
    "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
    "AZURE_DOCUMENT_INTELLIGENCE_KEY",
]
OPENAI_ENV_KEYS = [
    "OPENAI_API_KEY",
    "OPENAI_ANSWER_MODEL",
    "OPENAI_EMBEDDING_MODEL",
    "OPENAI_EMBEDDING_DIMENSIONS",
    "OPENAI_TRANSCRIPTION_MODEL",
    "OPENAI_VISION_MODEL",
    "OPENAI_GUARDRAIL_MODEL",
    "OPANAI_API_KEY",
]


@dataclass(frozen=True)
class FileFingerprint:
    path: str
    size_bytes: int
    modified_ns: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "size_bytes": self.size_bytes,
            "modified_ns": self.modified_ns,
            "sha256": self.sha256,
        }


def log(message: str) -> None:
    print(f"[volume-ingest] {message}", flush=True)


def read_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": MANIFEST_VERSION, "files": {}, "runs": []}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        backup_path = path.with_suffix(path.suffix + f".corrupt-{int(time.time())}")
        shutil.copy2(path, backup_path)
        log(f"Manifest was corrupt. Backed it up to {backup_path}")
        return {"version": MANIFEST_VERSION, "files": {}, "runs": []}
    payload.setdefault("version", MANIFEST_VERSION)
    payload.setdefault("files", {})
    payload.setdefault("runs", [])
    return payload


def write_manifest(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(dump_json(payload), encoding="utf-8")
    temporary_path.replace(path)


def sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(block_size), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(path: Path) -> FileFingerprint:
    stat = path.stat()
    return FileFingerprint(
        path=str(path),
        size_bytes=stat.st_size,
        modified_ns=stat.st_mtime_ns,
        sha256=sha256_file(path),
    )


def matches_any(path: Path, patterns: list[str], root: Path) -> bool:
    relative = path.relative_to(root).as_posix()
    return any(
        fnmatch.fnmatch(relative, pattern)
        or fnmatch.fnmatch(f"./{relative}", pattern)
        or fnmatch.fnmatch(path.name, pattern)
        for pattern in patterns
    )


def is_under(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def discover_files(root: Path, include: list[str], exclude: list[str], excluded_roots: list[Path] | None = None) -> list[Path]:
    if not root.exists():
        raise FileNotFoundError(f"Input Volume path does not exist: {root}")
    if root.is_file():
        return [root]
    excluded_resolved = [path.resolve() for path in excluded_roots or []]
    files = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        resolved_path = path.resolve()
        if any(is_under(resolved_path, excluded_root) for excluded_root in excluded_resolved):
            continue
        if not matches_any(path, include, root):
            continue
        if matches_any(path, exclude, root):
            continue
        if path.suffix.lower() not in DOCUMENT_EXTENSIONS | OFFICE_EXTENSIONS | VIDEO_EXTENSIONS:
            continue
        files.append(path)
    return sorted(files)


def file_changed(file_key: str, current: FileFingerprint, manifest: dict[str, Any], force: bool) -> bool:
    if force:
        return True
    previous = manifest.get("files", {}).get(file_key)
    if not previous or previous.get("status") != "success":
        return True
    old = previous.get("fingerprint") or {}
    return (
        old.get("sha256") != current.sha256
        or int(old.get("size_bytes") or -1) != current.size_bytes
        or int(old.get("modified_ns") or -1) != current.modified_ns
    )


def is_fatal_external_error(exc: Exception) -> bool:
    message = f"{type(exc).__name__}: {exc}".lower()
    fatal_markers = (
        "insufficient_quota",
        "exceeded your current quota",
        "billing details",
        "invalid_api_key",
        "incorrect api key",
        "authenticationerror",
    )
    return any(marker in message for marker in fatal_markers)


def apply_model_mode(args: argparse.Namespace) -> None:
    if not args.databricks_models_only:
        return
    os.environ["LLM_PROVIDER"] = "databricks"
    args.skip_pdf_vision = True
    if args.video_transcription_provider == "auto":
        args.video_transcription_provider = "databricks"
    args.skip_video_vision = True
    for key in OPENAI_ENV_KEYS:
        os.environ.pop(key, None)


def output_summary_path() -> Path:
    path = output_dir("ingestion", f"volume-ingestion-run-{int(time.time())}.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def convert_office_to_pdf(source_path: Path, converted_root: Path) -> Path:
    libreoffice = shutil.which("libreoffice") or shutil.which("soffice")
    if not libreoffice:
        raise RuntimeError("LibreOffice/soffice is required to convert Office files before OCR.")
    converted_root.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            libreoffice,
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(converted_root),
            str(source_path),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=900,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Office conversion failed for {source_path}: {completed.stderr or completed.stdout}")
    converted_path = converted_root / f"{source_path.stem}.pdf"
    if not converted_path.exists():
        matches = sorted(converted_root.glob(f"{source_path.stem}*.pdf"))
        if not matches:
            raise RuntimeError(f"Office conversion did not create a PDF for {source_path}")
        converted_path = matches[0]
    return converted_path


def process_pdf_file(source_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from app.pipelines.process_pdf import analyze_pdf
    from app.services.chunk_document import chunk_document
    from app.services.embed_chunks import embed_chunks_file

    payload = analyze_pdf(
        source_path,
        all_pages=True,
        force_all_visual=args.force_all_visual,
        vision_model=args.pdf_vision_model,
        skip_vision=args.skip_pdf_vision,
    )
    di_json = Path(payload["document_intelligence_output"])
    mm_json = Path(payload["multimodal_output"])
    chunks, chunk_path = chunk_document(di_json, mm_json)
    _, embedded_path = embed_chunks_file(chunk_path)
    return {
        "kind": "pdf",
        "source": str(source_path),
        "document_intelligence_output": str(di_json),
        "multimodal_output": str(mm_json),
        "chunk_path": str(chunk_path),
        "embedded_path": str(embedded_path),
        "chunks": len(chunks),
    }


def process_video_file(source_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    from app.services.embed_chunks import embed_chunks_file
    from app.services.video_processing import process_video

    chunks, chunk_path = process_video(
        video_path=source_path,
        frame_interval_seconds=args.frame_interval_seconds,
        chunk_window_seconds=args.chunk_window_seconds,
        skip_transcription=args.skip_video_transcription,
        transcription_provider=args.video_transcription_provider,
        audio_segment_seconds=args.audio_segment_seconds,
        transcription_max_tokens=args.transcription_max_tokens,
        skip_vision=args.skip_video_vision,
        ocr_workers=args.ocr_workers,
        vision_workers=args.vision_workers,
        vision_timeout_seconds=args.vision_timeout_seconds,
    )
    _, embedded_path = embed_chunks_file(chunk_path)
    return {
        "kind": "video",
        "source": str(source_path),
        "chunk_path": str(chunk_path),
        "embedded_path": str(embedded_path),
        "chunks": len(chunks),
    }


def read_text_file(path: Path, max_bytes: int = 20 * 1024 * 1024) -> str:
    if path.stat().st_size > max_bytes:
        raise RuntimeError(f"Text file is larger than {max_bytes} bytes. Split it before ingestion: {path}")
    return path.read_text(encoding="utf-8", errors="replace")


def process_text_file(source_path: Path) -> dict[str, Any]:
    from app.services.embed_chunks import embed_chunks_file

    text = read_text_file(source_path)
    doc_id = slugify(source_path.stem)
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n") if paragraph.strip()]
    chunks = []
    buffer: list[str] = []
    buffer_chars = 0

    def flush() -> None:
        nonlocal buffer_chars
        if not buffer:
            return
        content = "\n\n".join(buffer).strip()
        if not content:
            buffer.clear()
            buffer_chars = 0
            return
        chunks.append(
            {
                "id": f"{doc_id}-text-{len(chunks) + 1:06d}-{stable_id(content)}",
                "doc_id": doc_id,
                "source_pdf": str(source_path),
                "source_path": str(source_path),
                "source_type": "document",
                "content": content,
                "content_type": "text",
                "page_numbers": [],
                "section_path": [source_path.name],
                "section": source_path.name,
                "role": "body",
                "token_count": estimate_tokens(content),
                "metadata": {"source": "plain_text_volume_ingestion", "file_extension": source_path.suffix.lower()},
            }
        )
        buffer.clear()
        buffer_chars = 0

    for paragraph in paragraphs or [text.strip()]:
        if buffer and buffer_chars + len(paragraph) > 3200:
            flush()
        if len(paragraph) > 3200:
            for start in range(0, len(paragraph), 3200):
                buffer.append(paragraph[start : start + 3200])
                flush()
            continue
        buffer.append(paragraph)
        buffer_chars += len(paragraph)
    flush()

    chunk_path = output_dir("chunks") / f"{doc_id}-chunks.jsonl"
    write_jsonl(chunk_path, chunks)
    _, embedded_path = embed_chunks_file(chunk_path)
    return {
        "kind": "text",
        "source": str(source_path),
        "chunk_path": str(chunk_path),
        "embedded_path": str(embedded_path),
        "chunks": len(chunks),
    }


def process_file(source_path: Path, args: argparse.Namespace) -> dict[str, Any]:
    suffix = source_path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return process_video_file(source_path, args)
    if suffix == ".pdf":
        return process_pdf_file(source_path, args)
    if suffix in OFFICE_EXTENSIONS:
        converted_root = output_dir("ingestion", "converted")
        pdf_path = convert_office_to_pdf(source_path, converted_root)
        result = process_pdf_file(pdf_path, args)
        result["kind"] = "office"
        result["original_source"] = str(source_path)
        result["converted_pdf"] = str(pdf_path)
        return result
    if suffix in DOCUMENT_EXTENSIONS:
        return process_text_file(source_path)
    raise RuntimeError(f"Unsupported file extension: {suffix}")


def rebuild_index() -> dict[str, Any]:
    from app.clients.lancedb_store import DEFAULT_TABLE_NAME, DB_DIR, create_or_replace_index
    from app.services.embed_chunks import embedding_config

    embedded_files = sorted(output_dir("embeddings").glob("*-embedded.jsonl"))
    chunks: list[dict[str, Any]] = []
    for path in embedded_files:
        chunks.extend(load_chunks_jsonl(path))
    discovered_chunks = len(chunks)
    active_model, _ = embedding_config()
    if active_model:
        matching_chunks = [
            chunk
            for chunk in chunks
            if active_model.lower() in str(chunk.get("embedding_model") or "").lower()
        ]
        if matching_chunks:
            chunks = matching_chunks
    vector_lengths: dict[int, int] = {}
    for chunk in chunks:
        vector = chunk.get("embedding")
        if isinstance(vector, list) and vector:
            vector_lengths[len(vector)] = vector_lengths.get(len(vector), 0) + 1
    target_vector_length = max(vector_lengths, key=vector_lengths.get) if vector_lengths else 0
    if target_vector_length:
        chunks = [
            chunk
            for chunk in chunks
            if isinstance(chunk.get("embedding"), list) and len(chunk["embedding"]) == target_vector_length
        ]
    if not chunks:
        return {
            "indexed_chunks": 0,
            "discovered_chunks": discovered_chunks,
            "active_embedding_model": active_model,
            "target_vector_length": target_vector_length,
            "embedded_files": [],
            "table_name": DEFAULT_TABLE_NAME,
            "vector_db": str(DB_DIR),
        }
    if str(DB_DIR).startswith("/Volumes/"):
        with tempfile.TemporaryDirectory(prefix="insight-copilot-lancedb-") as temporary_dir:
            local_db_dir = Path(temporary_dir) / "lancedb"
            create_or_replace_index(chunks, table_name=DEFAULT_TABLE_NAME, db_dir=local_db_dir)
            if DB_DIR.exists():
                shutil.rmtree(DB_DIR)
            DB_DIR.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(local_db_dir, DB_DIR)
    else:
        create_or_replace_index(chunks, table_name=DEFAULT_TABLE_NAME)
    return {
        "indexed_chunks": len(chunks),
        "discovered_chunks": discovered_chunks,
        "active_embedding_model": active_model,
        "target_vector_length": target_vector_length,
        "embedded_files": [str(path) for path in embedded_files],
        "table_name": DEFAULT_TABLE_NAME,
        "vector_db": str(DB_DIR),
    }


def parse_patterns(values: list[str] | None, default: list[str]) -> list[str]:
    if not values:
        return default
    patterns: list[str] = []
    for value in values:
        patterns.extend(item.strip() for item in value.split(",") if item.strip())
    return patterns or default


def configure_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.input_volume:
        os.environ["INSIGHT_DATA_ROOT"] = args.input_volume
    if args.output_root:
        os.environ["INSIGHT_OUTPUT_ROOT"] = args.output_root
    input_root = Path(args.input_volume).expanduser() if args.input_volume else data_root()
    manifest_path = Path(args.manifest_path).expanduser() if args.manifest_path else output_dir("ingestion", "volume-manifest.json")
    return input_root, manifest_path


def load_secret_scope(scope: str | None) -> None:
    if not scope:
        return
    try:
        import IPython

        shell = IPython.get_ipython()
        dbutils = shell.user_ns.get("dbutils") if shell else None
    except Exception:
        dbutils = None
    if not dbutils:
        log(f"Secret scope `{scope}` requested, but dbutils is not available in this runtime.")
        return

    for key in SECRET_ENV_KEYS:
        if os.getenv(key):
            continue
        try:
            value = dbutils.secrets.get(scope=scope, key=key)
        except Exception:
            continue
        if value:
            os.environ[key] = value


def run(args: argparse.Namespace) -> dict[str, Any]:
    input_root, manifest_path = configure_paths(args)
    load_secret_scope(args.secret_scope)
    apply_model_mode(args)
    if args.diagnostics:
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            try:
                import imageio_ffmpeg

                ffmpeg_path = imageio_ffmpeg.get_ffmpeg_exe()
            except Exception:
                ffmpeg_path = None
        runtime_host, runtime_token = "", ""
        serving_auth_error = ""
        config_error = ""
        provider = os.getenv("LLM_PROVIDER", "")
        chat_endpoint = os.getenv("DATABRICKS_CHAT_ENDPOINT", "")
        embedding_endpoint = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "")
        transcription_endpoint = os.getenv("DATABRICKS_TRANSCRIPTION_ENDPOINT", "")
        try:
            from app.config import (
                databricks_chat_endpoint,
                databricks_embedding_endpoint,
                databricks_transcription_endpoint,
                llm_provider,
            )

            provider = llm_provider()
            chat_endpoint = databricks_chat_endpoint()
            embedding_endpoint = databricks_embedding_endpoint()
            transcription_endpoint = databricks_transcription_endpoint()
        except Exception as exc:
            config_error = f"{type(exc).__name__}: {exc}"
        try:
            from app.clients.databricks_model_serving import serving_auth

            runtime_host, runtime_token = serving_auth()
        except Exception as exc:
            serving_auth_error = f"{type(exc).__name__}: {exc}"

        return {
            "diagnostics": True,
            "model_mode": {
                "llm_provider": provider,
                "databricks_models_only": args.databricks_models_only,
                "databricks_chat_endpoint": chat_endpoint,
                "databricks_embedding_endpoint": embedding_endpoint,
                "databricks_transcription_endpoint": transcription_endpoint,
                "video_transcription_provider": args.video_transcription_provider,
                "audio_segment_seconds": args.audio_segment_seconds,
                "skip_pdf_vision": args.skip_pdf_vision,
                "skip_video_transcription": args.skip_video_transcription,
                "skip_video_vision": args.skip_video_vision,
                "config_error": config_error,
                "serving_auth_error": serving_auth_error,
            },
            "input_root": str(input_root),
            "input_root_exists": input_root.exists(),
            "output_root": str(output_dir()),
            "output_root_exists": output_dir().exists(),
            "manifest_path": str(manifest_path),
            "binaries": {
                "ffmpeg": bool(ffmpeg_path),
                "ffprobe": bool(shutil.which("ffprobe")),
                "libreoffice": bool(shutil.which("libreoffice") or shutil.which("soffice")),
            },
            "env": {
                "DATABRICKS_HOST": bool(os.getenv("DATABRICKS_HOST")),
                "DATABRICKS_TOKEN": bool(os.getenv("DATABRICKS_TOKEN")),
                "DATABRICKS_RUNTIME_AUTH": bool(runtime_host and runtime_token),
                "DATABRICKS_TRANSCRIPTION_ENDPOINT": bool(os.getenv("DATABRICKS_TRANSCRIPTION_ENDPOINT")),
                "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT": bool(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT")),
                "AZURE_DOCUMENT_INTELLIGENCE_KEY": bool(os.getenv("AZURE_DOCUMENT_INTELLIGENCE_KEY")),
                "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
            },
        }

    include = parse_patterns(args.include, DEFAULT_INCLUDE)
    exclude = parse_patterns(args.exclude, DEFAULT_EXCLUDE)
    manifest = read_manifest(manifest_path)
    discovered = discover_files(input_root, include=include, exclude=exclude, excluded_roots=[output_dir()])
    log(f"Discovered {len(discovered)} supported files under {input_root}")

    candidates = []
    for path in discovered:
        current = fingerprint(path)
        file_key = str(path.resolve())
        if file_changed(file_key, current, manifest, force=args.force):
            candidates.append((file_key, path, current))
    if args.max_files:
        candidates = candidates[: args.max_files]

    run_summary: dict[str, Any] = {
        "started_at_epoch": int(time.time()),
        "input_root": str(input_root),
        "output_root": str(output_dir()),
        "manifest_path": str(manifest_path),
        "dry_run": args.dry_run,
        "discovered_files": len(discovered),
        "changed_files": len(candidates),
        "processed": [],
        "failed": [],
        "skipped": len(discovered) - len(candidates),
    }

    if args.dry_run:
        run_summary["changed_preview"] = [str(path) for _, path, _ in candidates]
        log(f"Dry run found {len(candidates)} new or updated files")
        summary_path = output_summary_path()
        summary_path.write_text(dump_json(run_summary), encoding="utf-8")
        run_summary["summary_path"] = str(summary_path)
        return run_summary

    for index, (file_key, path, current) in enumerate(candidates, 1):
        started = time.perf_counter()
        log(f"Processing {index}/{len(candidates)}: {path}")
        try:
            result = process_file(path, args)
            result["duration_seconds"] = round(time.perf_counter() - started, 3)
            result["fingerprint"] = current.to_dict()
            manifest["files"][file_key] = {
                "status": "success",
                "processed_at_epoch": int(time.time()),
                "fingerprint": current.to_dict(),
                "result": result,
            }
            run_summary["processed"].append(result)
            write_manifest(manifest_path, manifest)
        except Exception as exc:
            error = {
                "source": str(path),
                "error_type": type(exc).__name__,
                "message": str(exc),
                "traceback": traceback.format_exc(limit=8),
                "fingerprint": current.to_dict(),
            }
            manifest["files"][file_key] = {
                "status": "failed",
                "processed_at_epoch": int(time.time()),
                "fingerprint": current.to_dict(),
                "error": error,
            }
            run_summary["failed"].append(error)
            write_manifest(manifest_path, manifest)
            log(f"Failed {path}: {type(exc).__name__}: {exc}")
            if is_fatal_external_error(exc):
                run_summary["fatal_error"] = error
                log("Stopping ingestion early because a fatal external service error was detected.")
                break
            if args.fail_fast:
                break

    if run_summary["processed"] and not args.no_rebuild_index:
        log("Rebuilding combined LanceDB index from all embedded files")
        run_summary["index"] = rebuild_index()
    else:
        run_summary["index"] = {"skipped": True}

    run_summary["finished_at_epoch"] = int(time.time())
    manifest.setdefault("runs", []).append(
        {
            "started_at_epoch": run_summary["started_at_epoch"],
            "finished_at_epoch": run_summary["finished_at_epoch"],
            "processed": len(run_summary["processed"]),
            "failed": len(run_summary["failed"]),
            "dry_run": args.dry_run,
        }
    )
    manifest["runs"] = manifest["runs"][-50:]
    write_manifest(manifest_path, manifest)
    summary_path = output_summary_path()
    summary_path.write_text(dump_json(run_summary), encoding="utf-8")
    run_summary["summary_path"] = str(summary_path)
    return run_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally ingest new/updated Databricks Volume files into RAG artifacts.")
    parser.add_argument("--input-volume", default=os.getenv("DATABRICKS_DATA_VOLUME") or os.getenv("INSIGHT_DATA_ROOT") or os.getenv("DATA_ROOT"))
    parser.add_argument("--output-root", default=os.getenv("DATABRICKS_OUTPUT_VOLUME") or os.getenv("INSIGHT_OUTPUT_ROOT") or os.getenv("OUTPUT_ROOT"))
    parser.add_argument("--manifest-path", default=os.getenv("INGESTION_MANIFEST_PATH", ""))
    parser.add_argument("--secret-scope", default=os.getenv("INGESTION_SECRET_SCOPE", ""))
    parser.add_argument("--include", action="append", help="Glob include patterns, comma-separated or repeatable.")
    parser.add_argument("--exclude", action="append", help="Glob exclude patterns, comma-separated or repeatable.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="Print non-secret runtime readiness checks and exit.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--max-files", type=int, default=0)
    parser.add_argument("--no-rebuild-index", action="store_true")
    parser.add_argument("--databricks-models-only", action="store_true", help="Use Databricks model serving for embeddings and disable OpenAI-only ingestion steps.")
    parser.add_argument("--force-all-visual", action="store_true")
    parser.add_argument("--skip-pdf-vision", action="store_true", help="Skip OpenAI PDF page vision and rely on Azure OCR/layout.")
    parser.add_argument("--pdf-vision-model", default=os.getenv("OPENAI_VISION_MODEL") or "gpt-4o-mini")
    parser.add_argument("--frame-interval-seconds", type=float, default=float(os.getenv("VIDEO_FRAME_INTERVAL_SECONDS") or 5.0))
    parser.add_argument("--chunk-window-seconds", type=float, default=float(os.getenv("VIDEO_CHUNK_WINDOW_SECONDS") or 30.0))
    parser.add_argument("--skip-video-transcription", action="store_true", help="Skip audio transcription and rely on visual OCR.")
    parser.add_argument(
        "--video-transcription-provider",
        default=os.getenv("VIDEO_TRANSCRIPTION_PROVIDER") or "auto",
        choices=["auto", "openai", "databricks", "none"],
        help="Audio transcription provider. Databricks-only mode maps auto to databricks.",
    )
    parser.add_argument("--audio-segment-seconds", type=float, default=float(os.getenv("VIDEO_AUDIO_SEGMENT_SECONDS") or 600.0))
    parser.add_argument("--transcription-max-tokens", type=int, default=int(os.getenv("VIDEO_TRANSCRIPTION_MAX_TOKENS") or 4096))
    parser.add_argument("--skip-video-vision", action="store_true")
    parser.add_argument("--ocr-workers", type=int, default=int(os.getenv("VIDEO_OCR_WORKERS") or 2))
    parser.add_argument("--vision-workers", type=int, default=int(os.getenv("VIDEO_VISION_WORKERS") or 2))
    parser.add_argument("--vision-timeout-seconds", type=int, default=int(os.getenv("VIDEO_VISION_TIMEOUT_SECONDS") or 120))
    args = parser.parse_args()

    try:
        summary = run(args)
    except Exception as exc:
        print(json.dumps({"status": "failed", "error_type": type(exc).__name__, "message": str(exc)}, indent=2), file=sys.stderr)
        raise
    print(json.dumps(summary, indent=2))
    if summary.get("failed"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
