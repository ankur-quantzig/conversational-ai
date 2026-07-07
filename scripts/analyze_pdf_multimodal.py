from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
for deps in (Path("/private/tmp/azdeps"), Path("/private/tmp/multimodal_deps")):
    if deps.exists():
        sys.path.insert(0, str(deps))

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    load_dotenv = None

import fitz
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.core.credentials import AzureKeyCredential
from openai import OpenAI


OUTPUT_DIR = ROOT / "output" / "multimodal_analysis"
IMAGE_DIR = OUTPUT_DIR / "page_images"
DEFAULT_DI_MODEL = "prebuilt-layout"
DEFAULT_VISION_MODEL = "gpt-4o-mini"
FIGURE_PATTERN = re.compile(
    r"\b(fig(?:ure)?\.?\s*\d+|diagram|flow\s*chart|flowchart|architecture|pipeline|schematic|"
    r"chart|plot|graph|illustration|visualization)\b",
    re.IGNORECASE,
)
VISUAL_HINT_PATTERN = re.compile(r"\b(figure|fig\.|diagram|table|chart|graph|plot|image)\b", re.IGNORECASE)
TABLE_CAPTION_PATTERN = re.compile(r"^Table\s+\d+\.", re.I)


def load_env_file() -> None:
    if load_dotenv is not None:
        load_dotenv(ROOT / ".env")

    # Be forgiving of spaces around keys and the current OPANAI_API_KEY typo.
    env_path = ROOT / ".env"
    if env_path.exists():
        for line in env_path.read_text(errors="replace").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or "=" not in stripped:
                continue
            key, value = stripped.split("=", 1)
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            os.environ.setdefault(key, value)
            if key == "OPANAI_API_KEY":
                os.environ.setdefault("OPENAI_API_KEY", value)


def env_value(*names: str) -> str:
    for name in names:
        value = os.getenv(name)
        if value:
            return value
    return ""


def pick_pdf() -> Path:
    preferred = ROOT / "data/research_papers/rag/2002.08909v1 - REALM Retrieval-Augmented Language Model Pre-Training.pdf"
    if preferred.exists():
        return preferred
    pdfs = sorted((ROOT / "data" / "research_papers").glob("*/*.pdf"))
    if not pdfs:
        raise FileNotFoundError("No PDFs found under data/research_papers/*/*.pdf")
    return pdfs[0]


def content_for_spans(content: str, spans) -> str:
    if not content or not spans:
        return ""
    pieces = []
    for span in spans:
        offset = getattr(span, "offset", None)
        length = getattr(span, "length", None)
        if isinstance(offset, int) and isinstance(length, int):
            pieces.append(content[offset : offset + length])
    return "".join(pieces)


def table_to_dict(table) -> dict[str, Any]:
    return {
        "row_count": table.row_count,
        "column_count": table.column_count,
        "cells": [
            {
                "row_index": cell.row_index,
                "column_index": cell.column_index,
                "row_span": getattr(cell, "row_span", None),
                "column_span": getattr(cell, "column_span", None),
                "kind": getattr(cell, "kind", None),
                "content": cell.content,
            }
            for cell in table.cells
        ],
    }


def reconstruct_tables_from_text(page_text: str) -> list[dict[str, Any]]:
    if not page_text:
        return []

    tables = []
    lines = [line.rstrip() for line in page_text.splitlines()]
    idx = 0
    while idx < len(lines):
        line = lines[idx].strip()
        if not TABLE_CAPTION_PATTERN.match(line):
            idx += 1
            continue

        caption = line
        idx += 1
        body_lines = []
        while idx < len(lines):
            current = lines[idx].strip()
            if not current:
                if body_lines:
                    idx += 1
                    break
                idx += 1
                continue
            if current.startswith("Table ") and body_lines:
                break
            if current.startswith("Figure ") and body_lines:
                break
            if re.match(r"^\d+\.\s+[A-Z]", current):
                break
            body_lines.append(current)
            idx += 1

        rows = []
        for body_line in body_lines:
            parts = [part.strip() for part in re.split(r"\s{2,}|\t+", body_line) if part.strip()]
            rows.append(parts if len(parts) >= 2 else [body_line])
        if rows:
            max_cols = max(len(row) for row in rows)
            normalized_rows = [row + [""] * (max_cols - len(row)) for row in rows]
            tables.append(
                {
                    "caption": caption,
                    "row_count": len(normalized_rows),
                    "column_count": max_cols,
                    "rows": normalized_rows,
                    "source": "text_fallback",
                    "confidence": "low",
                }
            )
    return tables


def analyze_with_document_intelligence(pdf_path: Path, model_id: str) -> dict[str, Any]:
    endpoint = env_value(
        "AZURE_DOCUMENT_INTELLIGENCE_ENDPOINT",
        "DOCUMENT_INTELLIGENCE_ENDPOINT",
        "AZURE_FORM_RECOGNIZER_ENDPOINT",
    )
    key = env_value(
        "AZURE_DOCUMENT_INTELLIGENCE_KEY",
        "DOCUMENT_INTELLIGENCE_KEY",
        "AZURE_FORM_RECOGNIZER_KEY",
    )
    if not endpoint or not key:
        raise RuntimeError("Missing Azure Document Intelligence endpoint/key in .env")

    client = DocumentIntelligenceClient(endpoint=endpoint, credential=AzureKeyCredential(key))
    with pdf_path.open("rb") as handle:
        poller = client.begin_analyze_document(model_id, handle, content_type="application/pdf")
    result = poller.result()

    content = getattr(result, "content", "") or ""
    pages = getattr(result, "pages", []) or []
    tables = getattr(result, "tables", []) or []
    paragraphs = getattr(result, "paragraphs", []) or []
    return {
        "model_id": model_id,
        "api_model_id": getattr(result, "model_id", None),
        "content": content,
        "content_chars": len(content),
        "paragraphs": len(paragraphs),
        "tables": [table_to_dict(table) for table in tables],
        "pages": [
            {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "lines": len(page.lines or []),
                "words": len(page.words or []),
                "text": content_for_spans(content, getattr(page, "spans", []) or []),
            }
            for page in pages
        ],
    }


def pdf_page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def local_page_text(pdf_path: Path, page_number: int) -> str:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        return page.get_text("text") or ""
    finally:
        doc.close()


def page_has_visuals(pdf_path: Path, page_number: int) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    try:
        page = doc.load_page(page_number - 1)
        text = page.get_text("text") or ""
        image_count = len(page.get_images(full=True))
        drawing_count = len(page.get_drawings())
        has_visual = image_count > 0 or drawing_count > 0 or bool(VISUAL_HINT_PATTERN.search(text))
        return {
            "has_visual": has_visual,
            "image_count": image_count,
            "drawing_count": drawing_count,
            "text_hint": bool(VISUAL_HINT_PATTERN.search(text)),
        }
    finally:
        doc.close()


def page_records(
    pdf_path: Path,
    di_result: dict[str, Any],
    force_pages: list[int] | None = None,
    all_pages: bool = False,
) -> list[dict[str, Any]]:
    di_pages = {page["page_number"]: page for page in di_result["pages"]}
    total_pages = pdf_page_count(pdf_path)

    if force_pages:
        selected = force_pages
    elif all_pages:
        selected = list(range(1, total_pages + 1))
    else:
        selected = list(range(1, total_pages + 1))

    records = []
    for page_number in selected:
        page = di_pages.get(page_number)
        if page is None:
            text = local_page_text(pdf_path, page_number)
            source = "local"
        else:
            text = page.get("text", "")
            source = "azure_di"
        visual = page_has_visuals(pdf_path, page_number)
        matches = FIGURE_PATTERN.findall(text)
        if visual["has_visual"]:
            reason_parts = []
            if visual["image_count"]:
                reason_parts.append(f"{visual['image_count']} embedded images")
            if visual["drawing_count"]:
                reason_parts.append(f"{visual['drawing_count']} drawings")
            if visual["text_hint"] or matches:
                reason_parts.append("visual keywords in text")
            reason = ", ".join(reason_parts) if reason_parts else "visual content detected"
        elif matches:
            reason = "figure/caption keywords: " + ", ".join(sorted(set(m[:40] for m in matches))[:6])
        elif source == "local":
            reason = "local fallback page text"
        else:
            reason = "azure_di page text"
        records.append(
            {
                "page_number": page_number,
                "reason": reason,
                "text": text,
                "source": source,
                "has_visual": visual["has_visual"],
                "image_count": visual["image_count"],
                "drawing_count": visual["drawing_count"],
                "tables_fallback": reconstruct_tables_from_text(text),
            }
        )
    return records


def render_page(pdf_path: Path, page_number: int, dpi: int = 160) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    page = doc[page_number - 1]
    pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
    out_path = IMAGE_DIR / f"{pdf_path.stem[:70]}-page-{page_number}.png"
    pix.save(out_path)
    doc.close()
    return out_path


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def analyze_page_with_openai(
    client: OpenAI,
    model_candidates: list[str],
    pdf_path: Path,
    page_number: int,
    image_path: Path,
    page_text: str,
) -> dict[str, Any]:
    prompt = f"""
You are analyzing a rendered PDF page from an academic paper for a RAG preprocessing pipeline.

PDF: {pdf_path.name}
Page: {page_number}

Task:
1. Identify whether the page contains any figures, flow charts, diagrams, architecture drawings, plots, graphs, tables, equations, or other visual elements.
2. For each visual element, provide:
   - type
   - concise title or caption if visible
   - detailed description of what the visual shows
   - important labels, arrows, axes, boxes, or relationships
   - why the visual matters for retrieval/search
3. If there is no meaningful visual element, say that clearly.

Nearby extracted page text:
{page_text[:3500]}
""".strip()

    last_error = ""
    for model in model_candidates:
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "input_text", "text": prompt},
                            {"type": "input_image", "image_url": image_to_data_url(image_path)},
                        ],
                    }
                ],
            )
            return {
                "page_number": page_number,
                "image_path": str(image_path.relative_to(ROOT)),
                "status": "ok",
                "model": model,
                "analysis": response.output_text,
                "error": "",
            }
        except Exception as exc:  # pragma: no cover - keeps batch jobs from losing partial outputs
            last_error = f"{model}: {type(exc).__name__}: {exc}"
            continue
    return {
        "page_number": page_number,
        "image_path": str(image_path.relative_to(ROOT)),
        "status": "error",
        "model": None,
        "analysis": "",
        "error": last_error,
    }


def write_outputs(pdf_path: Path, payload: dict[str, Any]) -> Path:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pdf_path.stem[:80]}-multimodal-analysis.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def parse_pages(value: str | None) -> list[int] | None:
    if not value:
        return None
    pages = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        pages.append(int(part))
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PDF text/layout and figure pages with OpenAI vision.")
    parser.add_argument("pdf", nargs="?", help="PDF path. Defaults to the REALM sample PDF.")
    parser.add_argument("--max-vision-pages", type=int, default=3, help="Maximum candidate pages sent to OpenAI vision.")
    parser.add_argument("--pages", help="Comma-separated page numbers to force through vision, e.g. 1,3,8.")
    parser.add_argument("--all-pages", action="store_true", help="Analyze every page in the PDF.")
    parser.add_argument("--force-all-visual", action="store_true", help="Ignore page screening and send all pages to vision.")
    parser.add_argument("--di-model", default=DEFAULT_DI_MODEL)
    parser.add_argument(
        "--vision-model",
        default=env_value("OPENAI_VISION_MODEL") or DEFAULT_VISION_MODEL,
        help="Primary vision model; fallback models are tried automatically.",
    )
    parser.add_argument("--skip-openai", action="store_true", help="Run Document Intelligence and render pages only.")
    args = parser.parse_args()

    load_env_file()
    pdf_path = Path(args.pdf).resolve() if args.pdf else pick_pdf()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    di_result = analyze_with_document_intelligence(pdf_path, args.di_model)
    all_page_records = page_records(pdf_path, di_result, parse_pages(args.pages), True)
    candidates = list(all_page_records)
    if not args.force_all_visual:
        candidates = [item for item in candidates if item.get("has_visual")]
    if not args.all_pages and not args.pages and not args.force_all_visual:
        candidates = candidates[: args.max_vision_pages]
    client = None if args.skip_openai else OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
    vision_candidates = [m.strip() for m in args.vision_model.split(",") if m.strip()]
    if "gpt-4o-mini" not in vision_candidates:
        vision_candidates.append("gpt-4o-mini")
    if "gpt-4.1-mini" not in vision_candidates:
        vision_candidates.append("gpt-4.1-mini")
    if "gpt-4o" not in vision_candidates:
        vision_candidates.append("gpt-4o")

    visual_pages = []
    for candidate in candidates:
        page_number = candidate["page_number"]
        image_path = render_page(pdf_path, page_number)
        if args.skip_openai:
            visual_pages.append(
                {
                    "page_number": page_number,
                    "reason": candidate["reason"],
                    "image_path": str(image_path.relative_to(ROOT)),
                    "status": "skipped",
                    "analysis": "",
                    "error": "",
                }
            )
        else:
            result = analyze_page_with_openai(
                client=client,
                model_candidates=vision_candidates,
                pdf_path=pdf_path,
                page_number=page_number,
                image_path=image_path,
                page_text=candidate["text"],
            )
            result["reason"] = candidate["reason"]
            result["has_visual"] = candidate.get("has_visual", False)
            result["image_count"] = candidate.get("image_count", 0)
            result["drawing_count"] = candidate.get("drawing_count", 0)
            visual_pages.append(result)

    payload = {
        "input_pdf": str(pdf_path.relative_to(ROOT)),
        "document_intelligence": di_result,
        "vision_model": None if args.skip_openai else vision_candidates[0],
        "vision_model_candidates": None if args.skip_openai else vision_candidates,
        "vision_candidate_pages": [
            {
                "page_number": item["page_number"],
                "reason": item["reason"],
                "has_visual": item.get("has_visual", False),
                "image_count": item.get("image_count", 0),
                "drawing_count": item.get("drawing_count", 0),
            }
            for item in candidates
        ],
        "visual_analysis": visual_pages,
        "tables_fallback": [
            {
                "page_number": item["page_number"],
                "caption": table["caption"],
                "row_count": table["row_count"],
                "column_count": table["column_count"],
                "rows": table["rows"],
                "source": table["source"],
                "confidence": table["confidence"],
            }
            for item in all_page_records
            for table in item.get("tables_fallback", [])
        ],
    }
    output_path = write_outputs(pdf_path, payload)

    summary = {
        "input_pdf": payload["input_pdf"],
        "pages": len(di_result["pages"]),
        "content_chars": di_result["content_chars"],
        "tables": len(di_result["tables"]),
        "vision_model": payload["vision_model"],
        "vision_pages": [item["page_number"] for item in visual_pages],
        "output": str(output_path.relative_to(ROOT)),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
