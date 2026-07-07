from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.clients.lancedb_store import DB_DIR, DEFAULT_TABLE_NAME, create_or_replace_index
from app.services.chunk_document import load_chunks_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a local LanceDB vector index from embedded chunks.")
    parser.add_argument("embedded_jsonl", type=Path)
    parser.add_argument("--table-name", default=DEFAULT_TABLE_NAME)
    args = parser.parse_args()

    chunks = load_chunks_jsonl(args.embedded_jsonl)
    create_or_replace_index(chunks, table_name=args.table_name)
    print(
        json.dumps(
            {
                "vector_db": str(DB_DIR),
                "table_name": args.table_name,
                "indexed_chunks": len(chunks),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
