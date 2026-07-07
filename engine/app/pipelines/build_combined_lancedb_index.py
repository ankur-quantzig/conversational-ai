from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.clients.lancedb_store import DB_DIR, DEFAULT_TABLE_NAME, create_or_replace_index
from app.services.chunk_document import load_chunks_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build one LanceDB table from multiple embedded chunk JSONL files.")
    parser.add_argument(
        "embedded_jsonl",
        nargs="*",
        type=Path,
        default=sorted(Path("output/embeddings").glob("*-embedded.jsonl")),
    )
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    args = parser.parse_args()

    chunks = []
    for path in args.embedded_jsonl:
        chunks.extend(load_chunks_jsonl(path))

    create_or_replace_index(chunks, table_name=args.table_name)
    print(
        json.dumps(
            {
                "vector_db": str(DB_DIR),
                "table_name": args.table_name,
                "indexed_chunks": len(chunks),
                "embedded_files": [str(path) for path in args.embedded_jsonl],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
