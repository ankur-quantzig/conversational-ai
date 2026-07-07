from __future__ import annotations

from typing import Any

from app.schemas.extracted_document import normalize_table


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


def normalize_span(span) -> dict[str, int | None]:
    return {
        "offset": getattr(span, "offset", None),
        "length": getattr(span, "length", None),
    }


def normalize_point(point) -> dict[str, float | None] | float | int | str | None:
    if hasattr(point, "x") and hasattr(point, "y"):
        return {"x": getattr(point, "x", None), "y": getattr(point, "y", None)}
    return point


def normalize_bounding_region(region) -> dict[str, Any]:
    polygon = getattr(region, "polygon", None) or []
    return {
        "page_number": getattr(region, "page_number", None),
        "polygon": [normalize_point(point) for point in polygon],
    }


def normalize_paragraph(paragraph, full_content: str) -> dict[str, Any]:
    spans = getattr(paragraph, "spans", []) or []
    bounding_regions = getattr(paragraph, "bounding_regions", []) or []
    content = getattr(paragraph, "content", None) or content_for_spans(full_content, spans)
    page_numbers = sorted(
        {
            getattr(region, "page_number", None)
            for region in bounding_regions
            if getattr(region, "page_number", None) is not None
        }
    )
    return {
        "role": getattr(paragraph, "role", None) or "body",
        "content": content,
        "content_chars": len(content),
        "page_numbers": page_numbers,
        "spans": [normalize_span(span) for span in spans],
        "bounding_regions": [normalize_bounding_region(region) for region in bounding_regions],
    }


def role_counts(paragraphs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for paragraph in paragraphs:
        role = paragraph.get("role") or "body"
        counts[role] = counts.get(role, 0) + 1
    return dict(sorted(counts.items()))


def normalize_di_result(result) -> dict[str, Any]:
    content = getattr(result, "content", "") or ""
    pages = getattr(result, "pages", []) or []
    tables = getattr(result, "tables", []) or []
    paragraphs = getattr(result, "paragraphs", []) or []
    paragraphs_full = [normalize_paragraph(paragraph, content) for paragraph in paragraphs]
    return {
        "model_id": getattr(result, "model_id", None),
        "api_model_id": getattr(result, "model_id", None),
        "content": content,
        "content_chars": len(content),
        "paragraphs": len(paragraphs),
        "paragraph_roles": role_counts(paragraphs_full),
        "paragraphs_full": paragraphs_full,
        "headings": [
            paragraph
            for paragraph in paragraphs_full
            if paragraph.get("role") in {"title", "sectionHeading"}
        ],
        "headers_footers": [
            paragraph
            for paragraph in paragraphs_full
            if paragraph.get("role") in {"pageHeader", "pageFooter", "pageNumber", "footnote"}
        ],
        "tables": [normalize_table(table) for table in tables],
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
