from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from app.schemas.chunk import RagChunk
from app.utils.files import output_dir
from app.utils.logging import dump_json


MAX_CHARS = 3_200
MIN_CHARS = 250
STRUCTURAL_ROLES = {"title", "sectionHeading"}
SKIP_ROLES = {"pageHeader", "pageFooter", "pageNumber"}


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


def slugify(value: str) -> str:
    value = re.sub(r"[^a-zA-Z0-9]+", "-", value.lower()).strip("-")
    return value[:80] or "document"


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("||".join(parts).encode("utf-8")).hexdigest()[:12]
    return digest


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + "\n", encoding="utf-8")


def table_to_markdown(table: dict[str, Any]) -> str:
    rows = table.get("rows") or []
    if not rows:
        return ""
    max_cols = max(len(row) for row in rows)
    normalized = [row + [""] * (max_cols - len(row)) for row in rows]
    header = normalized[0]
    body = normalized[1:] if len(normalized) > 1 else []
    lines = [
        "| " + " | ".join(str(cell).replace("\n", " ") for cell in header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for row in body:
        lines.append("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |")
    return "\n".join(lines)


def flush_text_chunk(
    chunks: list[RagChunk],
    buffer: list[dict[str, Any]],
    doc_id: str,
    source_pdf: str,
    section_path: list[str],
) -> None:
    if not buffer:
        return
    content = "\n\n".join(item["content"].strip() for item in buffer if item.get("content", "").strip())
    if not content:
        buffer.clear()
        return
    pages = sorted({page for item in buffer for page in item.get("page_numbers", [])})
    first_page = pages[0] if pages else 0
    chunk_index = len(chunks) + 1
    chunks.append(
        RagChunk(
            id=f"{doc_id}-text-{chunk_index:06d}-{stable_id(content)}",
            doc_id=doc_id,
            source_pdf=source_pdf,
            content=content,
            content_type="text",
            page_numbers=pages,
            section_path=list(section_path),
            role="body",
            token_count=estimate_tokens(content),
            metadata={
                "first_page": first_page,
                "paragraph_count": len(buffer),
                "source": "azure_document_intelligence",
            },
        )
    )
    buffer.clear()


def build_text_chunks(di: dict[str, Any], doc_id: str, source_pdf: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    section_path: list[str] = []
    buffer: list[dict[str, Any]] = []
    buffer_chars = 0

    for paragraph in di.get("paragraphs_full", []) or []:
        role = paragraph.get("role") or "body"
        content = (paragraph.get("content") or "").strip()
        if not content or role in SKIP_ROLES:
            continue

        if role in STRUCTURAL_ROLES:
            flush_text_chunk(chunks, buffer, doc_id, source_pdf, section_path)
            if role == "title":
                section_path = [content]
            else:
                section_path = section_path[:1] + [content] if section_path else [content]
            chunks.append(
                RagChunk(
                    id=f"{doc_id}-{role}-{len(chunks) + 1:06d}-{stable_id(content)}",
                    doc_id=doc_id,
                    source_pdf=source_pdf,
                    content=content,
                    content_type="heading",
                    page_numbers=paragraph.get("page_numbers", []),
                    section_path=list(section_path),
                    role=role,
                    token_count=estimate_tokens(content),
                    metadata={
                        "source": "azure_document_intelligence",
                        "bounding_regions": paragraph.get("bounding_regions", []),
                    },
                )
            )
            continue

        paragraph_chars = len(content)
        if buffer and buffer_chars + paragraph_chars > MAX_CHARS and buffer_chars >= MIN_CHARS:
            flush_text_chunk(chunks, buffer, doc_id, source_pdf, section_path)
            buffer_chars = 0
        buffer.append(paragraph)
        buffer_chars += paragraph_chars

    flush_text_chunk(chunks, buffer, doc_id, source_pdf, section_path)
    return chunks


def build_table_chunks(di: dict[str, Any], doc_id: str, source_pdf: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for index, table in enumerate(di.get("tables_full", []) or [], 1):
        markdown = table_to_markdown(table)
        if not markdown:
            continue
        page_number = table.get("page_number")
        content = f"Table {index}"
        if page_number:
            content += f" on page {page_number}"
        content += f"\n\n{markdown}"
        chunks.append(
            RagChunk(
                id=f"{doc_id}-table-{index:06d}-{stable_id(content)}",
                doc_id=doc_id,
                source_pdf=source_pdf,
                content=content,
                content_type="table",
                page_numbers=[page_number] if page_number else [],
                section_path=[],
                role="table",
                token_count=estimate_tokens(content),
                metadata={
                    "source": "azure_document_intelligence",
                    "row_count": table.get("row_count"),
                    "column_count": table.get("column_count"),
                    "bounding_regions": table.get("bounding_regions", []),
                    "rows": table.get("rows", []),
                },
            )
        )
    return chunks


def build_visual_chunks(mm: dict[str, Any], doc_id: str, source_pdf: str) -> list[RagChunk]:
    chunks: list[RagChunk] = []
    for index, visual in enumerate(mm.get("visual_analysis", []) or [], 1):
        analysis = (visual.get("analysis") or "").strip()
        if not analysis:
            continue
        page_number = visual.get("page_number")
        chunks.append(
            RagChunk(
                id=f"{doc_id}-visual-{index:06d}-{stable_id(analysis)}",
                doc_id=doc_id,
                source_pdf=source_pdf,
                content=f"Visual analysis for page {page_number}\n\n{analysis}",
                content_type="figure_summary",
                page_numbers=[page_number] if page_number else [],
                section_path=[],
                role="figure_summary",
                token_count=estimate_tokens(analysis),
                metadata={
                    "source": "openai_vision",
                    "model": visual.get("model"),
                    "status": visual.get("status"),
                    "reason": visual.get("reason"),
                    "image_path": visual.get("image_path"),
                },
            )
        )
    return chunks


def split_large_text(text: str, max_chars: int = MAX_CHARS) -> list[str]:
    text = text.strip()
    if len(text) <= max_chars:
        return [text] if text else []
    parts = []
    buffer = []
    buffer_chars = 0
    for paragraph in [item.strip() for item in text.split("\n\n") if item.strip()]:
        paragraph_chars = len(paragraph)
        if buffer and buffer_chars + paragraph_chars > max_chars:
            parts.append("\n\n".join(buffer))
            buffer = []
            buffer_chars = 0
        if paragraph_chars > max_chars:
            for index in range(0, paragraph_chars, max_chars):
                parts.append(paragraph[index : index + max_chars])
            continue
        buffer.append(paragraph)
        buffer_chars += paragraph_chars
    if buffer:
        parts.append("\n\n".join(buffer))
    return parts


def build_merged_page_chunks(mm: dict[str, Any], doc_id: str, source_pdf: str) -> list[RagChunk]:
    pages = (mm.get("merged_extraction") or {}).get("pages") or []
    chunks: list[RagChunk] = []
    for page in pages:
        page_number = page.get("page_number")
        merged_text = (page.get("merged_text") or "").strip()
        if not merged_text:
            continue
        page_prefix = f"Page {page_number} merged extraction"
        content_base = f"{page_prefix}\n\n{merged_text}"
        for part_index, content in enumerate(split_large_text(content_base), 1):
            chunks.append(
                RagChunk(
                    id=f"{doc_id}-merged-page-{page_number or 0:04d}-{part_index:03d}-{stable_id(content)}",
                    doc_id=doc_id,
                    source_pdf=source_pdf,
                    content=content,
                    content_type="merged_page_extraction",
                    page_numbers=[page_number] if page_number else [],
                    section_path=[Path(source_pdf).stem, f"Page {page_number}"],
                    role="merged_page",
                    token_count=estimate_tokens(content),
                    metadata={
                        "source": "merged_page_extraction",
                        "page_number": page_number,
                        "merge_sources": page.get("sources", []),
                        "dedupe": page.get("dedupe", {}),
                        "tables": page.get("tables", []),
                    },
                )
            )
    return chunks


def chunk_document(di_path: Path, mm_path: Path | None = None) -> tuple[list[RagChunk], Path]:
    di = load_json(di_path)
    mm = load_json(mm_path) if mm_path and mm_path.exists() else {}
    source_pdf = di.get("input_pdf") or di_path.stem
    doc_id = slugify(Path(source_pdf).stem)
    merged_chunks = build_merged_page_chunks(mm, doc_id, source_pdf)
    chunks = merged_chunks or [
        *build_text_chunks(di, doc_id, source_pdf),
        *build_table_chunks(di, doc_id, source_pdf),
        *build_visual_chunks(mm, doc_id, source_pdf),
    ]
    chunk_path = output_dir("chunks") / f"{doc_id}-chunks.jsonl"
    write_jsonl(chunk_path, [chunk.to_dict(include_embedding=False) for chunk in chunks])
    summary_path = output_dir("chunks") / f"{doc_id}-chunk-summary.json"
    summary = {
        "doc_id": doc_id,
        "source_pdf": source_pdf,
        "chunk_count": len(chunks),
        "by_content_type": {},
        "output": str(chunk_path),
    }
    for chunk in chunks:
        summary["by_content_type"][chunk.content_type] = summary["by_content_type"].get(chunk.content_type, 0) + 1
    summary_path.write_text(dump_json(summary), encoding="utf-8")
    return chunks, chunk_path


def load_chunks_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
