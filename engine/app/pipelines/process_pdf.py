from __future__ import annotations

import argparse
import base64
import json
import re
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

import fitz
from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.document_intelligence import client_from_env, env_value, load_dotenv_file
from app.rag.answer import require_all_properties
from app.services.extract_layout import normalize_di_result
from app.services.extract_tables import reconstruct_tables_from_text
from app.utils.files import data_root, display_path, output_dir, project_root
from app.utils.logging import dump_json


DEFAULT_DI_MODEL = "prebuilt-layout"
DEFAULT_VISION_MODEL = "gpt-4o-mini"
OUTPUT_DIR = output_dir("multimodal_analysis")
IMAGE_DIR = OUTPUT_DIR / "page_images"
DI_OUTPUT_DIR = output_dir("document_intelligence")
FIGURE_PATTERN = re.compile(
    r"\b(fig(?:ure)?\.?\s*\d+|diagram|flow\s*chart|flowchart|architecture|pipeline|schematic|chart|plot|graph|illustration|visualization)\b",
    re.IGNORECASE,
)
VISUAL_HINT_PATTERN = re.compile(r"\b(figure|fig\.|diagram|table|chart|graph|plot|image)\b", re.IGNORECASE)
SKIP_ROLES = {"pageHeader", "pageFooter", "pageNumber"}


class PageVisualElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_type: str = Field(description="Figure, chart, table, equation, diagram, screenshot, logo, or other visual type.")
    title: str = Field(default="", description="Visible title/caption when present.")
    description: str = Field(description="Detailed description of what the element shows.")
    important_labels: list[str] = Field(default_factory=list)
    relationships: list[str] = Field(default_factory=list)


class PageVisualTable(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(default="")
    description: str = Field(default="")
    rows_markdown: str = Field(default="", description="Markdown table when readable, otherwise concise text.")


class PageVisionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int
    page_summary: str = Field(description="Dense summary of all important content visible on this page.")
    visible_text: list[str] = Field(default_factory=list, description="Important visible text not already obvious from OCR.")
    headings: list[str] = Field(default_factory=list)
    key_points: list[str] = Field(default_factory=list)
    visual_elements: list[PageVisualElement] = Field(default_factory=list)
    tables: list[PageVisualTable] = Field(default_factory=list)
    equations: list[str] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


def page_vision_schema() -> dict[str, Any]:
    return require_all_properties(PageVisionExtraction.model_json_schema())


def pick_pdf() -> Path:
    root = data_root()
    preferred_name = "2002.08909v1 - REALM Retrieval-Augmented Language Model Pre-Training.pdf"
    pdfs = sorted(root.rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {root}")
    for pdf in pdfs:
        if pdf.name == preferred_name:
            return pdf
    return pdfs[0]


def page_count(pdf_path: Path) -> int:
    doc = fitz.open(pdf_path)
    try:
        return doc.page_count
    finally:
        doc.close()


def render_page(pdf_path: Path, page_number: int, dpi: int = 160) -> Path:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    try:
        page = doc[page_number - 1]
        pix = page.get_pixmap(matrix=fitz.Matrix(dpi / 72, dpi / 72), alpha=False)
        out_path = IMAGE_DIR / f"{pdf_path.stem[:70]}-page-{page_number}.png"
        pix.save(out_path)
        return out_path
    finally:
        doc.close()


def page_has_visuals(pdf_path: Path, page_number: int) -> dict:
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
            "text": text,
        }
    finally:
        doc.close()


def image_to_data_url(image_path: Path) -> str:
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def parse_page_vision(text: str) -> PageVisionExtraction:
    try:
        return PageVisionExtraction.model_validate_json(text)
    except ValidationError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return PageVisionExtraction.model_validate_json(text[start : end + 1])
        raise


def analyze_page_with_openai(client: OpenAI, model_candidates: list[str], pdf_path: Path, page_number: int, image_path: Path, page_text: str) -> dict:
    prompt = f"""
You are extracting all useful content from a rendered PDF page.

PDF: {pdf_path.name}
Page: {page_number}

Task:
1. Read the page image and extract all important information that a text parser may miss.
2. Capture visible headings, labels, figures, flow charts, diagrams, architecture drawings, plots, graphs, tables, equations, screenshots, and callouts.
3. Use the nearby Azure Document Intelligence text as supporting context, but do not simply copy it unless it is needed to complete the page extraction.
4. Be exhaustive and precise. Nothing important on the page should be omitted.
5. Return only valid JSON matching the provided schema.

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
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "page_vision_extraction",
                        "schema": page_vision_schema(),
                        "strict": True,
                    }
                },
            )
            structured = parse_page_vision(response.output_text)
            return {
                "page_number": page_number,
                "image_path": display_path(image_path),
                "status": "ok",
                "model": model,
                "analysis": structured.page_summary,
                "structured": structured.model_dump(),
                "error": "",
            }
        except Exception as exc:
            last_error = f"{model}: {type(exc).__name__}: {exc}"
    return {
        "page_number": page_number,
        "image_path": display_path(image_path),
        "status": "error",
        "model": None,
        "analysis": "",
        "structured": {},
        "error": last_error,
    }


def page_records(pdf_path: Path, di_result: dict, all_pages: bool = False) -> list[dict]:
    total_pages = page_count(pdf_path)
    selected = list(range(1, total_pages + 1)) if all_pages else list(range(1, total_pages + 1))
    di_pages = {page["page_number"]: page for page in di_result["pages"]}
    records = []
    for page_number in selected:
        page = di_pages.get(page_number)
        text = page["text"] if page else page_has_visuals(pdf_path, page_number)["text"]
        visual = page_has_visuals(pdf_path, page_number)
        matches = FIGURE_PATTERN.findall(text)
        if visual["has_visual"]:
            parts = []
            if visual["image_count"]:
                parts.append(f"{visual['image_count']} embedded images")
            if visual["drawing_count"]:
                parts.append(f"{visual['drawing_count']} drawings")
            if visual["text_hint"] or matches:
                parts.append("visual keywords in text")
            reason = ", ".join(parts) if parts else "visual content detected"
        elif matches:
            reason = "figure/caption keywords: " + ", ".join(sorted(set(m[:40] for m in matches))[:6])
        else:
            reason = "azure_di page text"
        records.append(
            {
                "page_number": page_number,
                "reason": reason,
                "text": text,
                "has_visual": visual["has_visual"],
                "image_count": visual["image_count"],
                "drawing_count": visual["drawing_count"],
                "tables_fallback": reconstruct_tables_from_text(text),
            }
        )
    return records


def normalize_for_dedupe(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def is_duplicate_text(candidate: str, existing: list[str]) -> bool:
    key = normalize_for_dedupe(candidate)
    if not key:
        return True
    for item in existing:
        item_key = normalize_for_dedupe(item)
        if not item_key:
            continue
        if key == item_key or key in item_key or item_key in key:
            return True
        if SequenceMatcher(None, key, item_key).ratio() >= 0.9:
            return True
    return False


def add_unique_text(values: list[str], value: str) -> None:
    cleaned = " ".join(str(value or "").split())
    if cleaned and not is_duplicate_text(cleaned, values):
        values.append(cleaned)


def markdown_table(table: dict[str, Any]) -> str:
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


def paragraphs_for_page(di_result: dict[str, Any], page_number: int) -> list[dict[str, Any]]:
    paragraphs = []
    for paragraph in di_result.get("paragraphs_full", []) or []:
        if paragraph.get("role") in SKIP_ROLES:
            continue
        if page_number in (paragraph.get("page_numbers") or []):
            paragraphs.append(paragraph)
    return paragraphs


def tables_for_page(di_result: dict[str, Any], page_number: int) -> list[dict[str, Any]]:
    tables = []
    for table in di_result.get("tables_full", []) or []:
        if table.get("page_number") == page_number:
            tables.append(table)
    return tables


def merge_page_extractions(di_result: dict[str, Any], page_records_payload: list[dict[str, Any]], visual_pages: list[dict[str, Any]]) -> dict[str, Any]:
    vision_by_page = {item.get("page_number"): item for item in visual_pages}
    merged_pages = []
    for record in page_records_payload:
        page_number = record["page_number"]
        page_text = " ".join((record.get("text") or "").split())
        paragraphs = paragraphs_for_page(di_result, page_number)
        di_tables = tables_for_page(di_result, page_number)
        vision = vision_by_page.get(page_number) or {}
        structured = vision.get("structured") or {}

        facts: list[str] = []
        for paragraph in paragraphs:
            add_unique_text(facts, paragraph.get("content", ""))
        if not facts:
            add_unique_text(facts, page_text)

        table_blocks = []
        for index, table in enumerate(di_tables, 1):
            table_text = markdown_table(table)
            if table_text:
                table_blocks.append({"source": "azure_document_intelligence", "title": f"Table {index}", "text": table_text})
                add_unique_text(facts, f"Table {index}\n{table_text}")

        if structured:
            add_unique_text(facts, structured.get("page_summary", ""))
            for value in structured.get("headings", []) or []:
                add_unique_text(facts, f"Heading: {value}")
            for value in structured.get("visible_text", []) or []:
                add_unique_text(facts, f"Visible text: {value}")
            for value in structured.get("key_points", []) or []:
                add_unique_text(facts, value)
            for element in structured.get("visual_elements", []) or []:
                parts = [
                    element.get("element_type", "Visual element"),
                    element.get("title", ""),
                    element.get("description", ""),
                    "; ".join(element.get("important_labels", []) or []),
                    "; ".join(element.get("relationships", []) or []),
                ]
                add_unique_text(facts, " - ".join(part for part in parts if part))
            for table in structured.get("tables", []) or []:
                text = "\n".join(
                    part
                    for part in [table.get("title", ""), table.get("description", ""), table.get("rows_markdown", "")]
                    if part
                )
                if text:
                    table_blocks.append({"source": "openai_vision", "title": table.get("title", ""), "text": text})
                    add_unique_text(facts, text)
            for equation in structured.get("equations", []) or []:
                add_unique_text(facts, f"Equation: {equation}")

        merged_text = "\n\n".join(facts)
        merged_pages.append(
            {
                "page_number": page_number,
                "merged_text": merged_text,
                "sources": ["azure_document_intelligence"] + (["openai_vision"] if structured else []),
                "azure_di": {
                    "page_text": page_text,
                    "paragraph_count": len(paragraphs),
                    "table_count": len(di_tables),
                },
                "vision": structured,
                "tables": table_blocks,
                "dedupe": {
                    "merged_items": len(facts),
                    "azure_paragraphs": len(paragraphs),
                    "vision_items_present": bool(structured),
                },
            }
        )
    return {
        "pages": merged_pages,
        "page_count": len(merged_pages),
        "sources": ["azure_document_intelligence", "openai_vision"],
        "dedupe_method": "normalized text containment plus SequenceMatcher >= 0.90",
    }


def analyze_pdf(pdf_path: Path, all_pages: bool = False, force_all_visual: bool = False, vision_model: str | None = None) -> dict:
    load_dotenv_file()
    di_client = client_from_env()
    with pdf_path.open("rb") as handle:
        poller = di_client.begin_analyze_document(DEFAULT_DI_MODEL, handle, content_type="application/pdf")
    relative_pdf = display_path(pdf_path)
    di_result = {
        "input_pdf": relative_pdf,
        "model_id": DEFAULT_DI_MODEL,
        "api_model_id": DEFAULT_DI_MODEL,
        **normalize_di_result(poller.result()),
    }

    all_page_records = page_records(pdf_path, di_result, all_pages=True)
    candidates = list(all_page_records)
    if not all_pages and not force_all_visual:
        candidates = list(all_page_records)

    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
    vision_candidates = [m.strip() for m in (vision_model or DEFAULT_VISION_MODEL).split(",") if m.strip()]
    for extra in ("gpt-4o-mini", "gpt-4.1-mini", "gpt-4o"):
        if extra not in vision_candidates:
            vision_candidates.append(extra)

    visual_pages = []
    for candidate in candidates:
        image_path = render_page(pdf_path, candidate["page_number"])
        result = analyze_page_with_openai(client, vision_candidates, pdf_path, candidate["page_number"], image_path, candidate["text"])
        result["reason"] = candidate["reason"]
        result["has_visual"] = candidate.get("has_visual", False)
        result["image_count"] = candidate.get("image_count", 0)
        result["drawing_count"] = candidate.get("drawing_count", 0)
        visual_pages.append(result)

    merged_extraction = merge_page_extractions(di_result, all_page_records, visual_pages)
    payload = {
        "input_pdf": relative_pdf,
        "document_intelligence": di_result,
        "vision_model": vision_candidates[0],
        "vision_model_candidates": vision_candidates,
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
        "merged_extraction": merged_extraction,
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
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DI_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    di_output_path = DI_OUTPUT_DIR / f"{pdf_path.stem[:80]}-document-intelligence.json"
    di_output_path.write_text(dump_json(di_result), encoding="utf-8")
    payload["document_intelligence_output"] = str(di_output_path)
    output_path = OUTPUT_DIR / f"{pdf_path.stem[:80]}-multimodal-analysis.json"
    payload["multimodal_output"] = str(output_path)
    output_path.write_text(dump_json(payload), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PDF text/layout and figure pages with OpenAI vision.")
    parser.add_argument("pdf", nargs="?", help="PDF path. Defaults to the REALM sample PDF.")
    parser.add_argument("--all-pages", action="store_true", default=True, help="Analyze every page in the PDF.")
    parser.add_argument("--force-all-visual", action="store_true", help="Ignore page screening and send all pages to vision.")
    parser.add_argument("--vision-model", default=DEFAULT_VISION_MODEL)
    args = parser.parse_args()
    pdf_path = Path(args.pdf).resolve() if args.pdf else pick_pdf()
    payload = analyze_pdf(pdf_path, all_pages=args.all_pages, force_all_visual=args.force_all_visual, vision_model=args.vision_model)
    print(json.dumps({"input_pdf": payload["input_pdf"], "pages": len(payload["document_intelligence"]["pages"]), "content_chars": payload["document_intelligence"]["content_chars"], "tables": len(payload["document_intelligence"]["tables"]), "vision_model": payload["vision_model"], "vision_pages": [v["page_number"] for v in payload["visual_analysis"]], "output": display_path(OUTPUT_DIR / f"{pdf_path.stem[:80]}-multimodal-analysis.json")}, indent=2))


if __name__ == "__main__":
    main()
