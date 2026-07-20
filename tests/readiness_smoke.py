from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.api import main as api

    with patch.object(api, "load_chunks", return_value=[{"id": "chunk-1"}]), patch.object(
        api, "vector_index_status", return_value={"consistent": True, "coverage": 1.0}
    ), patch.object(api, "validate_prompt_catalogs"), patch.object(
        api, "get_connection"
    ) as connection:
        connection.return_value.__enter__.return_value.execute.return_value = None
        assert api.readiness_snapshot()["ready"] is True

    with patch.object(api, "load_chunks", return_value=[{"id": "chunk-1"}]), patch.object(
        api, "vector_index_status", return_value={"consistent": False, "coverage": 0.1}
    ), patch.object(api, "validate_prompt_catalogs"), patch.object(
        api, "get_connection"
    ) as connection:
        connection.return_value.__enter__.return_value.execute.return_value = None
        assert api.readiness_snapshot()["ready"] is False

    print("readiness smoke ok")


if __name__ == "__main__":
    main()
