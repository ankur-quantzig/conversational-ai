from __future__ import annotations

import json
import subprocess
from pathlib import Path


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
CHUNKS_DIR = WORKSPACE_ROOT / "output" / "chunks"
SOURCE_VIDEOS_DIR = WORKSPACE_ROOT / "data" / "Videos"
CLIPS_DIR = WORKSPACE_ROOT / "deploy" / "databricks" / "artifacts" / "video_clips"


def slug(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "-" for ch in value.lower()).strip("-")


def load_video_segments() -> dict[str, tuple[Path, list[tuple[float, float]]]]:
    segments: dict[str, tuple[Path, list[tuple[float, float]]]] = {}
    for chunk_file in sorted(CHUNKS_DIR.glob("*video-chunks.jsonl")):
        for line in chunk_file.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            chunk = json.loads(line)
            metadata = chunk.get("metadata") or {}
            start = metadata.get("start_time")
            end = metadata.get("end_time")
            source_path = Path(chunk.get("source_path") or chunk.get("source_pdf") or "")
            if start is None or end is None or not source_path:
                continue
            doc_id = chunk["doc_id"]
            entry = segments.get(doc_id)
            if not entry:
                segments[doc_id] = (source_path, [(float(start), float(end))])
            else:
                entry[1].append((float(start), float(end)))
    return segments


def transcode_clip(source_path: Path, output_path: Path, start: float, end: float) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and output_path.stat().st_size > 0:
        return
    temp_path = output_path.with_suffix(".tmp.mp4")
    if temp_path.exists():
        temp_path.unlink()
    duration = max(0.1, end - start)
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-vf",
        "scale=640:-2",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "30",
        "-maxrate",
        "700k",
        "-bufsize",
        "1400k",
        "-c:a",
        "aac",
        "-b:a",
        "64k",
        "-movflags",
        "+faststart",
        str(temp_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True)
    temp_path.replace(output_path)


def main() -> None:
    segments = load_video_segments()
    if not segments:
        raise RuntimeError("No video chunks found")

    for doc_id, (source_path, ranges) in segments.items():
        candidate_source = source_path
        if not candidate_source.exists():
            candidate_source = SOURCE_VIDEOS_DIR / source_path.name
        if not candidate_source.exists():
            raise FileNotFoundError(f"Missing source video for {doc_id}: {source_path}")

        safe_doc_id = slug(doc_id)
        for start, end in sorted(set(ranges)):
            clip_name = f"{safe_doc_id}-{int(start * 1000)}-{int(end * 1000)}.mp4"
            output_path = CLIPS_DIR / clip_name
            transcode_clip(candidate_source, output_path, start, end)
            print(f"built {output_path.relative_to(WORKSPACE_ROOT)}")


if __name__ == "__main__":
    main()
