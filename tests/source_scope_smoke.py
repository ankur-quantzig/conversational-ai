from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    from app.config import source_doc_allowlist

    allowed = source_doc_allowlist()
    assert allowed == {
        "conversational-ai-next-steps",
        "conversational-ai-next-steps-20260702-173513-meeting-recording-1",
    }
    print("source scope smoke ok")


if __name__ == "__main__":
    main()
