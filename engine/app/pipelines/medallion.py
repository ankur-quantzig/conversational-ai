from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.services.chunk_document import load_chunks_jsonl
from app.utils.logging import dump_json


DEFAULT_CATALOG = "insight-copilot"
DEFAULT_BRONZE_SCHEMA = "bronze"
DEFAULT_SILVER_SCHEMA = "silver"
DEFAULT_GOLD_SCHEMA = "gold"
DEFAULT_BRONZE_TABLE = "insight_copilot_raw_files"
DEFAULT_SILVER_TABLE = "insight_copilot_extracted_content"
DEFAULT_GOLD_TABLE = "insight_copilot_rag_chunks"


def epoch() -> int:
    return int(time.time())


def json_text(value: Any) -> str:
    if value is None:
        return ""
    return dump_json(value) if isinstance(value, (dict, list)) else str(value)


def quote_identifier(value: str) -> str:
    return f"`{value.replace('`', '``')}`"


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def source_type_for_path(path: str, result: dict[str, Any] | None = None) -> str:
    kind = (result or {}).get("kind")
    if kind:
        return str(kind)
    suffix = Path(path).suffix.lower()
    if suffix in {".mp4", ".mov", ".m4v", ".mkv", ".webm"}:
        return "video"
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".doc", ".docx", ".ppt", ".pptx", ".xls", ".xlsx"}:
        return "office"
    return "document"


def spark_session():
    try:
        from pyspark.sql import SparkSession

        return SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()
    except Exception:
        return None


@dataclass
class MedallionConfig:
    enabled: bool
    required: bool
    catalog: str
    bronze_schema: str
    silver_schema: str
    gold_schema: str
    bronze_table: str
    silver_table: str
    gold_table: str

    @classmethod
    def from_args(cls, args: Any) -> "MedallionConfig":
        return cls(
            enabled=bool(getattr(args, "enable_medallion", False)),
            required=bool(getattr(args, "medallion_required", False)),
            catalog=getattr(args, "medallion_catalog", "") or os.getenv("MEDALLION_CATALOG") or DEFAULT_CATALOG,
            bronze_schema=getattr(args, "medallion_bronze_schema", "")
            or os.getenv("MEDALLION_BRONZE_SCHEMA")
            or DEFAULT_BRONZE_SCHEMA,
            silver_schema=getattr(args, "medallion_silver_schema", "")
            or os.getenv("MEDALLION_SILVER_SCHEMA")
            or DEFAULT_SILVER_SCHEMA,
            gold_schema=getattr(args, "medallion_gold_schema", "")
            or os.getenv("MEDALLION_GOLD_SCHEMA")
            or DEFAULT_GOLD_SCHEMA,
            bronze_table=getattr(args, "medallion_bronze_table", "")
            or os.getenv("MEDALLION_BRONZE_TABLE")
            or DEFAULT_BRONZE_TABLE,
            silver_table=getattr(args, "medallion_silver_table", "")
            or os.getenv("MEDALLION_SILVER_TABLE")
            or DEFAULT_SILVER_TABLE,
            gold_table=getattr(args, "medallion_gold_table", "")
            or os.getenv("MEDALLION_GOLD_TABLE")
            or DEFAULT_GOLD_TABLE,
        )


class MedallionWriter:
    def __init__(self, config: MedallionConfig, input_root: Path | None = None) -> None:
        self.config = config
        self.input_root = str(input_root or "")
        self.spark = spark_session() if config.enabled else None
        self.ready = False

    def table_name(self, schema: str, table: str) -> str:
        return ".".join(quote_identifier(part) for part in (self.config.catalog, schema, table))

    @property
    def bronze_table(self) -> str:
        return self.table_name(self.config.bronze_schema, self.config.bronze_table)

    @property
    def silver_table(self) -> str:
        return self.table_name(self.config.silver_schema, self.config.silver_table)

    @property
    def gold_table(self) -> str:
        return self.table_name(self.config.gold_schema, self.config.gold_table)

    def setup(self) -> dict[str, Any]:
        if not self.config.enabled:
            return {"enabled": False}
        if not self.spark:
            message = "SparkSession is not available; cannot write medallion Delta tables."
            if self.config.required:
                raise RuntimeError(message)
            return {"enabled": True, "ready": False, "message": message}

        try:
            for schema in (self.config.bronze_schema, self.config.silver_schema, self.config.gold_schema):
                self.spark.sql(f"CREATE SCHEMA IF NOT EXISTS {quote_identifier(self.config.catalog)}.{quote_identifier(schema)}")
            self.spark.sql(
                f"""
                CREATE TABLE IF NOT EXISTS {self.bronze_table} (
                  run_id STRING,
                  source_path STRING,
                  source_name STRING,
                  source_extension STRING,
                  source_type STRING,
                  input_root STRING,
                  size_bytes BIGINT,
                  modified_ns BIGINT,
                  sha256 STRING,
                  status STRING,
                  discovered_at_epoch BIGINT,
                  processed_at_epoch BIGINT,
                  duration_seconds DOUBLE,
                  result_kind STRING,
                  result_chunks BIGINT,
                  chunk_path STRING,
                  embedded_path STRING,
                  artifact_paths_json STRING,
                  fingerprint_json STRING,
                  result_json STRING,
                  error_json STRING
                )
                USING DELTA
                """
            )
            self.spark.sql(
                f"""
                CREATE TABLE IF NOT EXISTS {self.silver_table} (
                  run_id STRING,
                  source_path STRING,
                  doc_id STRING,
                  chunk_id STRING,
                  source_type STRING,
                  content_type STRING,
                  role STRING,
                  section STRING,
                  content STRING,
                  raw_content STRING,
                  normalized_content STRING,
                  token_count BIGINT,
                  quality_score DOUBLE,
                  needs_review BOOLEAN,
                  detected_languages_json STRING,
                  quality_corrections_json STRING,
                  quality_notes_json STRING,
                  quality_provider STRING,
                  quality_model STRING,
                  page_numbers_json STRING,
                  start_time_seconds DOUBLE,
                  end_time_seconds DOUBLE,
                  start_time_label STRING,
                  end_time_label STRING,
                  key_frame_path STRING,
                  frame_paths_json STRING,
                  metadata_json STRING,
                  chunk_path STRING,
                  artifact_paths_json STRING,
                  processed_at_epoch BIGINT
                )
                USING DELTA
                """
            )
            self.spark.sql(
                f"""
                CREATE TABLE IF NOT EXISTS {self.gold_table} (
                  run_id STRING,
                  source_path STRING,
                  doc_id STRING,
                  chunk_id STRING,
                  source_type STRING,
                  content_type STRING,
                  role STRING,
                  section STRING,
                  content STRING,
                  raw_content STRING,
                  normalized_content STRING,
                  token_count BIGINT,
                  quality_score DOUBLE,
                  needs_review BOOLEAN,
                  detected_languages_json STRING,
                  quality_corrections_json STRING,
                  quality_notes_json STRING,
                  quality_provider STRING,
                  quality_model STRING,
                  page_numbers_json STRING,
                  start_time_seconds DOUBLE,
                  end_time_seconds DOUBLE,
                  start_time_label STRING,
                  end_time_label STRING,
                  key_frame_path STRING,
                  frame_paths_json STRING,
                  metadata_json STRING,
                  embedding ARRAY<DOUBLE>,
                  embedding_model STRING,
                  embedding_dimensions BIGINT,
                  chunk_path STRING,
                  embedded_path STRING,
                  indexed_at_epoch BIGINT
                )
                USING DELTA
                """
            )
        except Exception as exc:
            if self.config.required:
                raise
            return {"enabled": True, "ready": False, "message": f"{type(exc).__name__}: {exc}"}

        self.ready = True
        return {
            "enabled": True,
            "ready": True,
            "bronze_table": self.bronze_table,
            "silver_table": self.silver_table,
            "gold_table": self.gold_table,
        }

    def append(self, table: str, rows: list[dict[str, Any]]) -> None:
        if not self.config.enabled or not rows:
            return
        if not self.ready:
            message = f"Medallion writer is not ready; skipped append to {table}."
            if self.config.required:
                raise RuntimeError(message)
            return
        assert self.spark is not None
        dataframe = self.spark.createDataFrame(rows)
        dataframe.write.format("delta").mode("append").option("mergeSchema", "true").saveAsTable(table)

    def record_file(
        self,
        *,
        run_id: str,
        source_path: Path,
        fingerprint: Any,
        status: str,
        result: dict[str, Any] | None = None,
        error: dict[str, Any] | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        result = result or {}
        row = {
            "run_id": run_id,
            "source_path": str(source_path),
            "source_name": source_path.name,
            "source_extension": source_path.suffix.lower(),
            "source_type": source_type_for_path(str(source_path), result),
            "input_root": self.input_root,
            "size_bytes": safe_int(getattr(fingerprint, "size_bytes", 0)),
            "modified_ns": safe_int(getattr(fingerprint, "modified_ns", 0)),
            "sha256": str(getattr(fingerprint, "sha256", "")),
            "status": status,
            "discovered_at_epoch": safe_int(getattr(fingerprint, "discovered_at_epoch", 0), epoch()),
            "processed_at_epoch": epoch(),
            "duration_seconds": safe_float(duration_seconds, -1.0),
            "result_kind": str(result.get("kind") or ""),
            "result_chunks": safe_int(result.get("chunks")),
            "chunk_path": str(result.get("chunk_path") or ""),
            "embedded_path": str(result.get("embedded_path") or ""),
            "artifact_paths_json": json_text(artifact_paths(result)),
            "fingerprint_json": json_text(fingerprint.to_dict() if hasattr(fingerprint, "to_dict") else {}),
            "result_json": json_text(result),
            "error_json": json_text(error or {}),
        }
        self.append(self.bronze_table, [row])

    def record_extractions(self, *, run_id: str, result: dict[str, Any]) -> None:
        chunk_path = Path(str(result.get("chunk_path") or ""))
        if not chunk_path.exists():
            return
        processed_at = epoch()
        artifacts = artifact_paths(result)
        rows = [
            silver_row(
                run_id=run_id,
                source_path=str(result.get("source") or result.get("original_source") or ""),
                chunk=chunk,
                chunk_path=str(chunk_path),
                artifact_paths=artifacts,
                processed_at_epoch=processed_at,
            )
            for chunk in load_chunks_jsonl(chunk_path)
        ]
        self.append(self.silver_table, rows)

    def record_embeddings(self, *, run_id: str, result: dict[str, Any]) -> None:
        embedded_path = Path(str(result.get("embedded_path") or ""))
        if not embedded_path.exists():
            return
        indexed_at = epoch()
        chunk_path = str(result.get("chunk_path") or "")
        rows = [
            gold_row(
                run_id=run_id,
                source_path=str(result.get("source") or result.get("original_source") or ""),
                chunk=chunk,
                chunk_path=chunk_path,
                embedded_path=str(embedded_path),
                indexed_at_epoch=indexed_at,
            )
            for chunk in load_chunks_jsonl(embedded_path)
        ]
        self.append(self.gold_table, rows)


def artifact_paths(result: dict[str, Any]) -> dict[str, str]:
    keys = (
        "document_intelligence_output",
        "multimodal_output",
        "raw_chunk_path",
        "chunk_path",
        "embedded_path",
        "quality_summary_path",
        "converted_pdf",
    )
    return {key: str(result.get(key) or "") for key in keys if result.get(key)}


def metadata_value(chunk: dict[str, Any], key: str, default: Any = None) -> Any:
    metadata = chunk.get("metadata") or {}
    return metadata.get(key, default)


def base_chunk_row(run_id: str, source_path: str, chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    quality = metadata.get("quality") or {}
    raw_content = str(metadata.get("raw_content") or "")
    normalized_content = str(chunk.get("content") or "")
    return {
        "run_id": run_id,
        "source_path": source_path or str(chunk.get("source_path") or chunk.get("source_pdf") or ""),
        "doc_id": str(chunk.get("doc_id") or ""),
        "chunk_id": str(chunk.get("id") or ""),
        "source_type": str(chunk.get("source_type") or metadata.get("source_type") or "document"),
        "content_type": str(chunk.get("content_type") or ""),
        "role": str(chunk.get("role") or ""),
        "section": str(chunk.get("section") or " > ".join(chunk.get("section_path") or [])),
        "content": normalized_content,
        "raw_content": raw_content,
        "normalized_content": normalized_content,
        "token_count": safe_int(chunk.get("token_count")),
        "quality_score": safe_float(chunk.get("quality_score") or quality.get("quality_score"), -1.0),
        "needs_review": safe_bool(chunk.get("needs_review") or quality.get("needs_review")),
        "detected_languages_json": json_text(chunk.get("detected_languages") or quality.get("detected_languages") or []),
        "quality_corrections_json": json_text(quality.get("corrections") or []),
        "quality_notes_json": json_text(quality.get("quality_notes") or []),
        "quality_provider": str(quality.get("provider") or ""),
        "quality_model": str(quality.get("model") or ""),
        "page_numbers_json": json_text(chunk.get("page_numbers") or []),
        "start_time_seconds": safe_float(metadata.get("start_time"), -1.0),
        "end_time_seconds": safe_float(metadata.get("end_time"), -1.0),
        "start_time_label": str(metadata.get("start_time_label") or ""),
        "end_time_label": str(metadata.get("end_time_label") or ""),
        "key_frame_path": str(metadata.get("key_frame_path") or ""),
        "frame_paths_json": json_text(metadata.get("frame_paths") or []),
        "metadata_json": json_text(metadata),
    }


def silver_row(
    *,
    run_id: str,
    source_path: str,
    chunk: dict[str, Any],
    chunk_path: str,
    artifact_paths: dict[str, str],
    processed_at_epoch: int,
) -> dict[str, Any]:
    row = base_chunk_row(run_id, source_path, chunk)
    row.update(
        {
            "chunk_path": chunk_path,
            "artifact_paths_json": json_text(artifact_paths),
            "processed_at_epoch": processed_at_epoch,
        }
    )
    return row


def gold_row(
    *,
    run_id: str,
    source_path: str,
    chunk: dict[str, Any],
    chunk_path: str,
    embedded_path: str,
    indexed_at_epoch: int,
) -> dict[str, Any]:
    row = base_chunk_row(run_id, source_path, chunk)
    vector = chunk.get("embedding") or []
    row.update(
        {
            "embedding": [float(value) for value in vector],
            "embedding_model": str(chunk.get("embedding_model") or ""),
            "embedding_dimensions": safe_int(chunk.get("embedding_dimensions") or len(vector)),
            "chunk_path": chunk_path,
            "embedded_path": embedded_path,
            "indexed_at_epoch": indexed_at_epoch,
        }
    )
    return row
