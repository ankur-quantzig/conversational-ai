from __future__ import annotations

import argparse
import sys
from contextlib import redirect_stdout
from pathlib import Path

from app.services.video_processing import analyze_frame_with_vision
from app.utils.logging import dump_json


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="Analyze one video frame with the configured vision model.")
    parser.add_argument("frame", type=Path)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    with redirect_stdout(sys.stderr):
        result = analyze_frame_with_vision(args.frame, model=args.model)
    print(dump_json(result))


if __name__ == "__main__":
    main()
