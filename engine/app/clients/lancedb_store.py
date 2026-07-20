from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lancedb

from app.utils.files import output_dir


DEFAULT_TABLE_NAME = "rag_chunks"
DB_DIR = output_dir("vector_db", "lancedb")
MIN_INDEX_COVERAGE = 0.8


def index_coverage_status(
    chunk_count: int,
    indexed_count: int,
    minimum_coverage: float = MIN_INDEX_COVERAGE,
) -> dict[str, Any]:
    coverage = 1.0 if chunk_count == 0 else indexed_count / chunk_count
    return {
        "chunks": max(0, int(chunk_count)),
        "indexed_rows": max(0, int(indexed_count)),
        "coverage": round(max(0.0, min(1.0, coverage)), 4),
        "minimum_coverage": minimum_coverage,
        "consistent": chunk_count > 0 and indexed_count >= chunk_count * minimum_coverage,
    }


def vector_index_status(chunks: list[dict[str, Any]]) -> dict[str, Any]:
    unique_chunk_ids = {str(chunk.get("id") or "") for chunk in chunks}
    unique_chunk_ids.discard("")
    try:
        indexed_count = open_table(DEFAULT_TABLE_NAME).count_rows()
        status = index_coverage_status(len(unique_chunk_ids), indexed_count)
        status["error"] = ""
        return status
    except Exception as exc:
        status = index_coverage_status(len(unique_chunk_ids), 0)
        status["error"] = f"{type(exc).__name__}: {exc}"
        return status


def connect(db_dir: Path | None = None):
    path = db_dir or DB_DIR
    path.mkdir(parents=True, exist_ok=True)
    return lancedb.connect(path)


def table_exists(table_name: str = DEFAULT_TABLE_NAME) -> bool:
    db = connect()
    return table_name in db.table_names()


def to_lancedb_record(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata", {}) or {}
    return {
        "id": chunk["id"],
        "doc_id": chunk.get("doc_id", ""),
        "source_pdf": chunk.get("source_pdf", ""),
        "source_path": chunk.get("source_path") or chunk.get("source_pdf", ""),
        "source_type": chunk.get("source_type") or metadata.get("source_type") or "document",
        "content": chunk.get("content", ""),
        "content_type": chunk.get("content_type", ""),
        "role": chunk.get("role", ""),
        "section": chunk.get("section") or " > ".join(chunk.get("section_path", [])),
        "page_numbers_json": json.dumps(chunk.get("page_numbers", [])),
        "start_time": float(metadata.get("start_time", -1.0)),
        "end_time": float(metadata.get("end_time", -1.0)),
        "start_time_label": metadata.get("start_time_label", ""),
        "end_time_label": metadata.get("end_time_label", ""),
        "key_frame_path": metadata.get("key_frame_path", ""),
        "token_count": chunk.get("token_count", 0),
        "metadata_json": json.dumps(metadata, ensure_ascii=False),
        "embedding_model": chunk.get("embedding_model", ""),
        "embedding_dimensions": chunk.get("embedding_dimensions", 0),
        "vector": chunk["embedding"],
    }


def active_embedding_model_filter() -> str:
    try:
        from app.config import databricks_embedding_endpoint, llm_provider

        if llm_provider() == "databricks":
            return databricks_embedding_endpoint().lower()
    except Exception:
        return ""
    return ""


def filter_compatible_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    active_model = active_embedding_model_filter()
    if active_model:
        model_records = [
            record
            for record in records
            if active_model in str(record.get("embedding_model") or "").lower()
        ]
        if model_records:
            records = model_records

    dimension_counts: dict[int, int] = {}
    for record in records:
        vector = record.get("vector")
        if isinstance(vector, list) and vector:
            dimension_counts[len(vector)] = dimension_counts.get(len(vector), 0) + 1
    if not dimension_counts:
        return []
    target_dimension = max(dimension_counts, key=dimension_counts.get)
    return [
        record
        for record in records
        if isinstance(record.get("vector"), list) and len(record["vector"]) == target_dimension
    ]


def quality_rank(chunk: dict[str, Any]) -> tuple[int, float]:
    metadata = chunk.get("metadata") or {}
    quality = metadata.get("quality") or {}
    try:
        score = float(chunk.get("quality_score") or quality.get("quality_score") or 0.0)
    except Exception:
        score = 0.0
    return int(bool(quality)) + int(bool(metadata.get("raw_content"))), score


def dedupe_chunks_by_id(chunks: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: dict[str, dict[str, Any]] = {}
    anonymous: list[dict[str, Any]] = []
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        if not chunk_id:
            anonymous.append(chunk)
            continue
        previous = deduped.get(chunk_id)
        if previous is None or quality_rank(chunk) >= quality_rank(previous):
            deduped[chunk_id] = chunk
    return [*deduped.values(), *anonymous]


def create_or_replace_index(chunks: list[dict[str, Any]], table_name: str = DEFAULT_TABLE_NAME, db_dir: Path | None = None):
    chunks = dedupe_chunks_by_id(chunks)
    records = filter_compatible_records([to_lancedb_record(chunk) for chunk in chunks if chunk.get("embedding")])
    if not records:
        raise ValueError("No embedded chunks found. Embed chunks before building the vector DB.")
    db = connect(db_dir)
    if table_name in db.table_names():
        db.drop_table(table_name)
    return db.create_table(table_name, data=records)


def open_table(table_name: str = DEFAULT_TABLE_NAME):
    db = connect()
    if table_name not in db.table_names():
        raise RuntimeError(f"LanceDB table `{table_name}` does not exist. Build the vector DB first.")
    return db.open_table(table_name)


def _normalize_query_vector(query_vector: list[float], table) -> list[float]:
    """Normalize an incoming query vector to the dimension of the table's `vector` column.

    If the table's vector column reports a fixed size, truncate or pad with zeros
    so the query matches that dimension. If we can't determine the dimension,
    return the original vector unchanged.
    """
    if not query_vector:
        return query_vector

    try:
        # lancedb's schema.field(...).type may expose a list size for fixed_size_list
        vector_field = table.schema.field("vector")
        vector_type = getattr(vector_field, "type", None)
        # try common attributes that indicate list size
        dimension = None
        if hasattr(vector_type, "list_size"):
            dimension = getattr(vector_type, "list_size")
        elif hasattr(vector_type, "shape"):
            # some versions may expose shape-like info
            shape = getattr(vector_type, "shape")
            if isinstance(shape, (list, tuple)) and shape:
                dimension = int(shape[-1])
    except Exception:
        return query_vector

    if dimension is None:
        return query_vector

    try:
        dimension = int(dimension)
    except Exception:
        return query_vector

    length = len(query_vector)
    if length == dimension:
        return query_vector
    if length > dimension:
        return query_vector[:dimension]
    # pad with zeros
    return query_vector + [0.0] * (dimension - length)


def document_indexed(doc_id: str, table_name: str = DEFAULT_TABLE_NAME) -> bool:
    if not table_exists(table_name):
        return False
    table = open_table(table_name)
    try:
        return table.search().where(f"doc_id = '{doc_id}'").limit(1).to_list() != []
    except Exception:
        return any(row.get("doc_id") == doc_id for row in table.to_list())


def vector_search(query_vector: list[float], top_k: int = 5, table_name: str = DEFAULT_TABLE_NAME) -> list[dict[str, Any]]:
    table = open_table(table_name)
    # ensure the query vector matches the table's stored vector dimension
    normalized_query_vector = _normalize_query_vector(query_vector, table)

    try:
        rows = (
            table.search(
                normalized_query_vector,
                vector_column_name="vector",
                query_type="vector",
            )
            .limit(top_k)
            .to_list()
        )
    except Exception as vector_exc:
        print(f"LanceDB search using vector column `vector` failed: {vector_exc}")

        # fallback to older column name if present
        rows = (
            table.search(
                normalized_query_vector,
                vector_column_name="embedding",
                query_type="vector",
            )
            .limit(top_k)
            .to_list()
        )

    results = []
    for row in rows:
        results.append(
            {
                "id": row["id"],
                "doc_id": row.get("doc_id"),
                "source_pdf": row.get("source_pdf"),
                "source_path": row.get("source_path") or row.get("source_pdf"),
                "source_type": row.get("source_type") or "document",
                "content": row.get("content"),
                "content_type": row.get("content_type"),
                "role": row.get("role"),
                "section": row.get("section"),
                "page_numbers": json.loads(row.get("page_numbers_json") or "[]"),
                "start_time": row.get("start_time"),
                "end_time": row.get("end_time"),
                "start_time_label": row.get("start_time_label"),
                "end_time_label": row.get("end_time_label"),
                "key_frame_path": row.get("key_frame_path"),
                "score": 1 / (1 + float(row.get("_distance", 0.0))),
                "distance": row.get("_distance"),
                "metadata": json.loads(row.get("metadata_json") or "{}"),
                "embedding_model": row.get("embedding_model") or "",
                "embedding_dimensions": row.get("embedding_dimensions") or 0,
            }
        )

    return results
