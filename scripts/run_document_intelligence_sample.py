from __future__ import annotations

import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TMP_DEPS = Path("/private/tmp/azdeps")
if TMP_DEPS.exists():
    sys.path.insert(0, str(TMP_DEPS))
from app.clients.document_intelligence import client_from_env, env_value
from app.services.extract_layout import content_for_spans, normalize_di_result
from app.utils.files import data_root, display_path, output_dir, project_root


OUTPUT_DIR = output_dir("document_intelligence")
DEFAULT_MODEL_ID = "prebuilt-layout"


def pick_pdf() -> Path:
    pdfs = sorted(data_root().rglob("*.pdf"))
    if not pdfs:
        raise FileNotFoundError(f"No PDFs found under {data_root()}")
    for pdf in pdfs:
        if pdf.parent.name == "rag":
            return pdf
    return pdfs[0]


def main() -> None:
    client = client_from_env()
    model_id = env_value("AZURE_DOCUMENT_INTELLIGENCE_MODEL_ID", "DOCUMENT_INTELLIGENCE_MODEL_ID") or DEFAULT_MODEL_ID

    pdf_path = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else pick_pdf()
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    with pdf_path.open("rb") as handle:
        poller = client.begin_analyze_document(model_id, handle, content_type="application/pdf")
    result = poller.result()
    normalized = normalize_di_result(result)

    page_summaries = [
        {
            "page_number": page["page_number"],
            "width": page["width"],
            "height": page["height"],
            "unit": page["unit"],
            "lines": page["lines"],
            "words": page["words"],
            "text_chars": len(page["text"]),
            "text": page["text"],
        }
        for page in normalized["pages"]
    ]

    output = {
        "input_pdf": display_path(pdf_path),
        **normalized,
        "page_count": len(normalized["pages"]),
        "model_id": model_id,
        "styles": len(getattr(result, "styles", []) or []),
        "languages": len(getattr(result, "languages", []) or []),
        "content_preview": " ".join(normalized["content"].split())[:1200],
        "page_summaries": page_summaries,
        "tables_sample": normalized["tables"][:3],
        "tables_full": normalized["tables"],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{pdf_path.stem[:80]}-document-intelligence.json"
    full_output_path = OUTPUT_DIR / f"{pdf_path.stem[:80]}-document-intelligence-full.json"
    output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    full_output_path.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "input_pdf": output["input_pdf"],
                "page_count": output["page_count"],
                "content_chars": output["content_chars"],
                "paragraph_roles": output["paragraph_roles"],
                "headings": len(output["headings"]),
                "headers_footers": len(output["headers_footers"]),
                "tables": len(output["tables_full"]),
                "outputs": [
                    display_path(output_path),
                    display_path(full_output_path),
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
