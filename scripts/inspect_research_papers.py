from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PDF_DEPS = Path("/private/tmp/pdfdeps")
if PDF_DEPS.exists():
    sys.path.insert(0, str(PDF_DEPS))

from pypdf import PdfReader  # noqa: E402


DATA_DIR = ROOT / "data" / "research_papers"
MANIFEST_PATH = DATA_DIR / "papers_manifest.json"
OUTPUT_PATH = DATA_DIR / "inspection_summary.json"


def read_manifest() -> dict[str, dict]:
    if not MANIFEST_PATH.exists():
        return {}
    items = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    return {item["arxiv_id"]: item for item in items if item.get("arxiv_id")}


def arxiv_id_from_path(path: Path) -> str:
    return path.name.split(" - ", 1)[0]


def inspect_pdf(path: Path, manifest_by_id: dict[str, dict]) -> dict:
    arxiv_id = arxiv_id_from_path(path)
    manifest = manifest_by_id.get(arxiv_id, {})
    record = {
        "path": str(path.relative_to(ROOT)),
        "topic": path.parent.name,
        "arxiv_id": arxiv_id,
        "title": manifest.get("title") or path.stem,
        "authors": manifest.get("authors", ""),
        "published": manifest.get("published", ""),
        "updated": manifest.get("updated", ""),
        "categories": manifest.get("categories", ""),
        "summary": manifest.get("summary", ""),
        "pages": None,
        "encrypted": False,
        "text_chars_sampled": 0,
        "first_page_chars": 0,
        "first_page_preview": "",
        "extractable_text": False,
        "status": "ok",
        "error": "",
    }
    try:
        reader = PdfReader(str(path))
        record["encrypted"] = bool(reader.is_encrypted)
        record["pages"] = len(reader.pages)
        sampled_text = []
        for index, page in enumerate(reader.pages[: min(3, len(reader.pages))]):
            text = page.extract_text() or ""
            if index == 0:
                record["first_page_chars"] = len(text)
                record["first_page_preview"] = " ".join(text.split())[:500]
            sampled_text.append(text)
        record["text_chars_sampled"] = sum(len(text) for text in sampled_text)
        record["extractable_text"] = record["text_chars_sampled"] > 500
    except Exception as exc:  # pragma: no cover - diagnostic script
        record["status"] = "error"
        record["error"] = f"{type(exc).__name__}: {exc}"
    return record


def main() -> None:
    manifest_by_id = read_manifest()
    pdfs = sorted(DATA_DIR.glob("*/*.pdf"))
    records = [inspect_pdf(path, manifest_by_id) for path in pdfs]

    topic_counts = Counter(record["topic"] for record in records)
    status_counts = Counter(record["status"] for record in records)
    extractable_counts = Counter(str(record["extractable_text"]) for record in records)
    page_counts_by_topic = defaultdict(list)
    for record in records:
        if isinstance(record["pages"], int):
            page_counts_by_topic[record["topic"]].append(record["pages"])

    topic_stats = {}
    for topic, counts in sorted(page_counts_by_topic.items()):
        topic_stats[topic] = {
            "count": len(counts),
            "min_pages": min(counts),
            "max_pages": max(counts),
            "avg_pages": round(sum(counts) / len(counts), 2),
        }

    summary = {
        "pdf_count": len(records),
        "topic_counts": dict(sorted(topic_counts.items())),
        "status_counts": dict(sorted(status_counts.items())),
        "extractable_text_counts": dict(sorted(extractable_counts.items())),
        "topic_page_stats": topic_stats,
        "records": records,
    }
    OUTPUT_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({key: summary[key] for key in summary if key != "records"}, indent=2))
    print(f"wrote {OUTPUT_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
