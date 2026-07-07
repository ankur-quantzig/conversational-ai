from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.services.chunk_document import chunk_document


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG chunks from DI and multimodal JSON.")
    parser.add_argument("di_json", type=Path)
    parser.add_argument("--multimodal-json", type=Path)
    args = parser.parse_args()

    chunks, output_path = chunk_document(args.di_json, args.multimodal_json)
    summary = {
        "chunk_count": len(chunks),
        "output": str(output_path),
        "content_types": {},
    }
    for chunk in chunks:
        summary["content_types"][chunk.content_type] = summary["content_types"].get(chunk.content_type, 0) + 1
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
