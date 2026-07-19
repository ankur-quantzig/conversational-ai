from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    import app.api.main as api

    original_load_chunks = api.load_chunks
    original_data_root = os.environ.get("INSIGHT_DATA_ROOT")
    original_output_root = os.environ.get("INSIGHT_OUTPUT_ROOT")

    try:
        with tempfile.TemporaryDirectory(prefix="video-source-resolution-") as temporary_dir:
            root = Path(temporary_dir)
            data_root = root / "volume"
            output_root = root / "output"
            video_path = data_root / "new_data" / "kt-video.mp4"
            video_path.parent.mkdir(parents=True, exist_ok=True)
            video_path.write_bytes(b"not-a-real-video")

            os.environ["INSIGHT_DATA_ROOT"] = str(data_root)
            os.environ["INSIGHT_OUTPUT_ROOT"] = str(output_root)

            api.load_chunks = lambda: [
                {
                    "doc_id": "kt-video",
                    "source_type": "video",
                    "source_path": str(video_path),
                    "metadata": {"source_type": "video"},
                }
            ]

            assert api.indexed_video_path("kt-video") == video_path.resolve()
    finally:
        api.load_chunks = original_load_chunks
        if original_data_root is None:
            os.environ.pop("INSIGHT_DATA_ROOT", None)
        else:
            os.environ["INSIGHT_DATA_ROOT"] = original_data_root
        if original_output_root is None:
            os.environ.pop("INSIGHT_OUTPUT_ROOT", None)
        else:
            os.environ["INSIGHT_OUTPUT_ROOT"] = original_output_root

    print("video source resolution smoke ok")


if __name__ == "__main__":
    main()
