from __future__ import annotations

import json
import os
import sys
import tempfile
import types
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def install_optional_dependency_stubs() -> None:
    if "openai" not in sys.modules:
        openai = types.ModuleType("openai")

        class OpenAI:
            pass

        openai.OpenAI = OpenAI
        sys.modules.setdefault("openai", openai)

    if "azure.ai.documentintelligence" not in sys.modules:
        azure = types.ModuleType("azure")
        azure_ai = types.ModuleType("azure.ai")
        documentintelligence = types.ModuleType("azure.ai.documentintelligence")
        documentintelligence.DocumentIntelligenceClient = object
        azure_core = types.ModuleType("azure.core")
        credentials = types.ModuleType("azure.core.credentials")
        credentials.AzureKeyCredential = object
        sys.modules.setdefault("azure", azure)
        sys.modules.setdefault("azure.ai", azure_ai)
        sys.modules.setdefault("azure.ai.documentintelligence", documentintelligence)
        sys.modules.setdefault("azure.core", azure_core)
        sys.modules.setdefault("azure.core.credentials", credentials)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def main() -> None:
    install_optional_dependency_stubs()

    from app.services.quality_enrichment import QualityEnrichmentConfig, detected_languages, enrich_chunks_file

    with tempfile.TemporaryDirectory(prefix="quality-enrichment-smoke-") as temporary_dir:
        root = Path(temporary_dir)
        os.environ["INSIGHT_OUTPUT_ROOT"] = str(root / "output")
        chunk_path = root / "chunks" / "kt-video-chunks.jsonl"
        raw_content = "yeh ingestion wala part Bronze me jayega. then embeddings Gold layer mein store hoga."
        write_jsonl(
            chunk_path,
            [
                {
                    "id": "chunk-1",
                    "doc_id": "kt-video",
                    "source_pdf": "kt.mp4",
                    "source_type": "video",
                    "content": raw_content,
                    "content_type": "video_window",
                    "page_numbers": [],
                    "section_path": ["KT", "00:00-01:00"],
                    "role": "video_window",
                    "token_count": 20,
                    "metadata": {"start_time_label": "00:00", "end_time_label": "01:00"},
                }
            ],
        )

        assert detected_languages(raw_content) == ["hi-en"]

        config = QualityEnrichmentConfig(provider="heuristic", min_llm_chars=1)
        chunks, output_path, summary_path, summary = enrich_chunks_file(chunk_path, config=config)
        assert output_path.exists()
        assert summary_path.exists()
        assert summary["chunk_count"] == 1
        assert summary["quality_provider"] == "heuristic"
        assert chunks[0]["metadata"]["raw_content"] == raw_content
        assert chunks[0]["metadata"]["quality"]["detected_languages"] == ["hi-en"]
        assert 0 <= chunks[0]["quality_score"] <= 1

        _, _, _, cached_summary = enrich_chunks_file(chunk_path, config=config)
        assert cached_summary["reused_cache"] is True

    print("quality enrichment smoke ok")


if __name__ == "__main__":
    main()
