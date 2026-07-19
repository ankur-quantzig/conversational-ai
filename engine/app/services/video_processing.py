from __future__ import annotations

import base64
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.databricks_model_serving import transcribe_audio as databricks_transcribe_audio
from app.clients.document_intelligence import client_from_env, env_value, load_dotenv_file
from app.config import databricks_transcription_endpoint, llm_provider
from app.rag.answer import require_all_properties
from app.services.chunk_document import estimate_tokens, slugify, stable_id, write_jsonl
from app.services.extract_layout import normalize_di_result
from app.utils.files import output_dir
from app.utils.logging import dump_json


DEFAULT_FRAME_INTERVAL_SECONDS = 5.0
DEFAULT_CHUNK_WINDOW_SECONDS = 30.0
DEFAULT_TRANSCRIPTION_MODEL = "whisper-1"
DEFAULT_AUDIO_SEGMENT_SECONDS = 600.0
DEFAULT_TRANSCRIPTION_MAX_TOKENS = 4096
DEFAULT_VISION_MODEL = "gpt-4.1-mini"
DEFAULT_VISION_TIMEOUT_SECONDS = 120


class FrameVisualElement(BaseModel):
    model_config = ConfigDict(extra="forbid")

    element_type: str = Field(default="")
    title: str = Field(default="")
    description: str = Field(default="")
    visible_labels: list[str] = Field(default_factory=list)


class FrameVisionExtraction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    frame_summary: str
    visible_text: list[str] = Field(default_factory=list)
    slide_title: str = Field(default="")
    key_points: list[str] = Field(default_factory=list)
    visual_elements: list[FrameVisualElement] = Field(default_factory=list)
    uncertainty_notes: list[str] = Field(default_factory=list)


def frame_vision_schema() -> dict[str, Any]:
    return require_all_properties(FrameVisionExtraction.model_json_schema())


@dataclass(frozen=True)
class VideoPaths:
    video_id: str
    root: Path
    audio: Path
    frames_dir: Path
    transcript: Path
    frame_ocr: Path
    visual_analysis: Path
    merged_analysis: Path
    chunks: Path
    summary: Path


def binary_path(binary: str) -> str | None:
    found = shutil.which(binary)
    if found:
        return found
    if binary == "ffmpeg":
        try:
            import imageio_ffmpeg

            return imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            return None
    return None


def require_binary(binary: str) -> str:
    path = binary_path(binary)
    if not path:
        raise RuntimeError(
            f"`{binary}` is required for video processing. Install it locally, add it to the Databricks image, "
            "or include imageio-ffmpeg for ffmpeg."
        )
    return path


def log_step(message: str) -> None:
    print(f"[video] {message}", flush=True)


def paths_for_video(video_path: Path) -> VideoPaths:
    video_id = slugify(video_path.stem)
    root = output_dir("videos", video_id)
    return VideoPaths(
        video_id=video_id,
        root=root,
        audio=root / "audio.mp3",
        frames_dir=root / "frames",
        transcript=root / "transcript.json",
        frame_ocr=root / "frame_ocr.json",
        visual_analysis=root / "visual_analysis.json",
        merged_analysis=root / "merged_frame_analysis.json",
        chunks=output_dir("chunks") / f"{video_id}-video-chunks.jsonl",
        summary=root / "video-summary.json",
    )


def run_command(args: list[str]) -> None:
    completed = subprocess.run(args, check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        raise RuntimeError(
            f"Command failed: {' '.join(args)}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
        )


def probe_video(video_path: Path) -> dict[str, Any]:
    ffprobe = binary_path("ffprobe")
    if ffprobe:
        completed = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-print_format",
                "json",
                "-show_format",
                "-show_streams",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        payload = json.loads(completed.stdout)
        video_stream = next((stream for stream in payload.get("streams", []) if stream.get("codec_type") == "video"), {})
        duration = float(payload.get("format", {}).get("duration") or video_stream.get("duration") or 0)
        return {
            "duration_seconds": duration,
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "fps": video_stream.get("r_frame_rate"),
            "format": payload.get("format", {}),
        }

    ffmpeg = require_binary("ffmpeg")
    completed = subprocess.run([ffmpeg, "-hide_banner", "-i", str(video_path)], check=False, capture_output=True, text=True)
    stderr = completed.stderr or completed.stdout
    duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)", stderr)
    width_match = re.search(r"Video:.*?(\d{2,5})x(\d{2,5})", stderr)
    duration = 0.0
    if duration_match:
        hours, minutes, seconds = duration_match.groups()
        duration = int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    return {
        "duration_seconds": duration,
        "width": int(width_match.group(1)) if width_match else None,
        "height": int(width_match.group(2)) if width_match else None,
        "fps": None,
        "format": {"probe_source": "ffmpeg_stderr"},
    }


def extract_audio(video_path: Path, audio_path: Path) -> None:
    ffmpeg = require_binary("ffmpeg")
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="insight-video-audio-") as temporary_dir:
        temporary_audio = Path(temporary_dir) / audio_path.name
        run_command(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_path),
                "-vn",
                "-ac",
                "1",
                "-ar",
                "16000",
                "-b:a",
                "32k",
                str(temporary_audio),
            ]
        )
        if not temporary_audio.exists() or temporary_audio.stat().st_size < 1024:
            raise RuntimeError(f"Audio extraction produced an empty or incomplete file: {temporary_audio}")
        shutil.copy2(temporary_audio, audio_path)


def extract_frames(video_path: Path, frames_dir: Path, interval_seconds: float) -> list[dict[str, Any]]:
    ffmpeg = require_binary("ffmpeg")
    frames_dir.mkdir(parents=True, exist_ok=True)
    frame_pattern = frames_dir / "frame_%06d.jpg"
    run_command(
        [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vf",
            f"fps=1/{interval_seconds}",
            "-q:v",
            "2",
            str(frame_pattern),
        ]
    )
    frames = []
    for index, frame_path in enumerate(sorted(frames_dir.glob("frame_*.jpg")), 1):
        timestamp = round((index - 1) * interval_seconds, 3)
        frames.append({"frame_index": index, "timestamp": timestamp, "image_path": str(frame_path)})
    return frames


def transcribe_audio(
    audio_path: Path,
    model: str | None = None,
    provider: str | None = None,
    duration_seconds: float = 0.0,
    segment_seconds: float = DEFAULT_AUDIO_SEGMENT_SECONDS,
    max_tokens: int = DEFAULT_TRANSCRIPTION_MAX_TOKENS,
    checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    load_dotenv_file()
    resolved_provider = resolve_transcription_provider(provider)
    if resolved_provider == "databricks":
        return transcribe_audio_with_databricks(
            audio_path=audio_path,
            model=model,
            duration_seconds=duration_seconds,
            segment_seconds=segment_seconds,
            max_tokens=max_tokens,
            checkpoint_path=checkpoint_path,
        )
    if resolved_provider == "openai":
        return transcribe_audio_with_openai(audio_path, model=model)
    raise RuntimeError(f"Unsupported video transcription provider: {resolved_provider}")


def resolve_transcription_provider(provider: str | None = None) -> str:
    configured = (provider or env_value("VIDEO_TRANSCRIPTION_PROVIDER") or "").strip().lower()
    if configured in {"", "auto"}:
        return "databricks" if llm_provider() == "databricks" else "openai"
    if configured in {"skip", "none", "disabled"}:
        return "none"
    return configured


def transcribe_audio_with_openai(audio_path: Path, model: str | None = None) -> dict[str, Any]:
    model = model or env_value("OPENAI_TRANSCRIPTION_MODEL") or DEFAULT_TRANSCRIPTION_MODEL
    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"), timeout=180)
    with audio_path.open("rb") as handle:
        response = client.audio.transcriptions.create(
            model=model,
            file=handle,
            response_format="verbose_json",
        )
    payload = response.model_dump() if hasattr(response, "model_dump") else dict(response)
    segments = []
    for index, segment in enumerate(payload.get("segments", []) or [], 1):
        segments.append(
            {
                "segment_index": index,
                "start": float(segment.get("start") or 0),
                "end": float(segment.get("end") or 0),
                "text": (segment.get("text") or "").strip(),
            }
        )
    if not segments and payload.get("text"):
        segments.append({"segment_index": 1, "start": 0.0, "end": 0.0, "text": payload["text"].strip()})
    return {"provider": "openai", "model": model, "text": payload.get("text", ""), "segments": segments}


def transcribe_audio_with_databricks(
    audio_path: Path,
    model: str | None = None,
    duration_seconds: float = 0.0,
    segment_seconds: float = DEFAULT_AUDIO_SEGMENT_SECONDS,
    max_tokens: int = DEFAULT_TRANSCRIPTION_MAX_TOKENS,
    checkpoint_path: Path | None = None,
    retries: int = 3,
) -> dict[str, Any]:
    endpoint = model or databricks_transcription_endpoint()
    segment_ranges = audio_segment_ranges(duration_seconds, segment_seconds)
    payload = load_transcript_checkpoint(checkpoint_path, provider="databricks", model=endpoint)
    completed = {int(segment.get("segment_index") or 0) for segment in payload.get("segments", [])}

    with tempfile.TemporaryDirectory(prefix="insight-audio-segments-") as temporary_dir:
        temporary_root = Path(temporary_dir)
        for segment_index, start, end in segment_ranges:
            if segment_index in completed:
                continue
            log_step(
                f"Databricks audio transcription segment {segment_index}/{len(segment_ranges)} "
                f"({format_time(start)}-{format_time(end)})"
            )
            if len(segment_ranges) == 1:
                segment_path = audio_path
            else:
                segment_path = temporary_root / f"audio_segment_{segment_index:06d}.mp3"
                extract_audio_slice(audio_path, segment_path, start_seconds=start, duration_seconds=max(1.0, end - start))
            text = transcribe_databricks_segment(
                segment_path=segment_path,
                endpoint=endpoint,
                max_tokens=max_tokens,
                retries=retries,
            )
            if text:
                payload.setdefault("segments", []).append(
                    {
                        "segment_index": segment_index,
                        "start": round(start, 3),
                        "end": round(end, 3),
                        "text": text,
                        "provider": "databricks",
                    }
                )
            payload["text"] = "\n".join(segment["text"] for segment in sorted(payload.get("segments", []), key=segment_sort_key))
            payload["segment_seconds"] = segment_seconds
            payload["duration_seconds"] = duration_seconds
            write_transcript_checkpoint(checkpoint_path, payload)

    payload["segments"] = sorted(payload.get("segments", []), key=segment_sort_key)
    payload["text"] = "\n".join(segment["text"] for segment in payload["segments"])
    payload["provider"] = "databricks"
    payload["model"] = endpoint
    return payload


def audio_segment_ranges(duration_seconds: float, segment_seconds: float) -> list[tuple[int, float, float]]:
    segment_seconds = max(30.0, float(segment_seconds or DEFAULT_AUDIO_SEGMENT_SECONDS))
    if duration_seconds <= 0:
        return [(1, 0.0, segment_seconds)]
    segment_count = max(1, math.ceil(duration_seconds / segment_seconds))
    ranges = []
    for index in range(segment_count):
        start = index * segment_seconds
        end = min(duration_seconds, start + segment_seconds)
        if end > start:
            ranges.append((index + 1, start, end))
    return ranges or [(1, 0.0, duration_seconds)]


def extract_audio_slice(audio_path: Path, segment_path: Path, start_seconds: float, duration_seconds: float) -> None:
    ffmpeg = require_binary("ffmpeg")
    segment_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            ffmpeg,
            "-y",
            "-ss",
            f"{max(0.0, start_seconds):.3f}",
            "-t",
            f"{max(1.0, duration_seconds):.3f}",
            "-i",
            str(audio_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            "-b:a",
            "32k",
            str(segment_path),
        ]
    )
    if not segment_path.exists() or segment_path.stat().st_size < 1024:
        raise RuntimeError(f"Audio segment extraction produced an empty or incomplete file: {segment_path}")


def transcribe_databricks_segment(segment_path: Path, endpoint: str, max_tokens: int, retries: int) -> str:
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            raw_text = databricks_transcribe_audio(segment_path, endpoint=endpoint, max_tokens=max_tokens)
            return parse_transcript_text(raw_text)
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
            delay = min(60, 2**attempt)
            log_step(f"Databricks transcription retry {attempt}/{retries - 1} for {segment_path.name}: {exc}. Waiting {delay}s")
            time.sleep(delay)
    raise RuntimeError("Databricks transcription failed") from last_error


def parse_transcript_text(raw_text: str) -> str:
    cleaned = strip_json_fence(raw_text)
    try:
        payload = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                payload = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                payload = None
        else:
            payload = None
    if isinstance(payload, dict):
        if isinstance(payload.get("text"), str):
            return normalize_transcript_whitespace(payload["text"])
        segments = payload.get("segments")
        if isinstance(segments, list):
            values = [segment.get("text", "") for segment in segments if isinstance(segment, dict)]
            return normalize_transcript_whitespace("\n".join(values))
    return normalize_transcript_whitespace(cleaned)


def strip_json_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def normalize_transcript_whitespace(text: str) -> str:
    lines = [" ".join(line.split()) for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def load_transcript_checkpoint(path: Path | None, provider: str, model: str) -> dict[str, Any]:
    if path and path.exists():
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("provider") == provider and payload.get("model") == model and isinstance(payload.get("segments"), list):
            return payload
    return {"provider": provider, "model": model, "text": "", "segments": []}


def write_transcript_checkpoint(path: Path | None, payload: dict[str, Any]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json(payload), encoding="utf-8")


def segment_sort_key(segment: dict[str, Any]) -> tuple[int, float]:
    return int(segment.get("segment_index") or 0), float(segment.get("start") or 0.0)


def ocr_frame_with_azure(frame_path: Path, model_id: str = "prebuilt-read", retries: int = 4) -> dict[str, Any]:
    client = client_from_env()
    last_error: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            with frame_path.open("rb") as handle:
                poller = client.begin_analyze_document(model_id, handle, content_type="image/jpeg")
            result = poller.result()
            break
        except Exception as exc:
            last_error = exc
            if attempt == retries:
                raise
            delay = min(30, 2**attempt)
            log_step(f"Azure OCR retry {attempt}/{retries - 1} for {frame_path.name}: {exc}. Waiting {delay}s")
            time.sleep(delay)
    else:
        raise RuntimeError(f"Azure OCR failed for {frame_path}") from last_error
    normalized = normalize_di_result(result)
    return {
        "image_path": str(frame_path),
        "model_id": model_id,
        "content": normalized.get("content", ""),
        "content_chars": normalized.get("content_chars", 0),
        "pages": normalized.get("pages", []),
        "paragraphs_full": normalized.get("paragraphs_full", []),
    }


def ocr_frames(
    frames: list[dict[str, Any]],
    model_id: str = "prebuilt-read",
    checkpoint_path: Path | None = None,
    workers: int = 1,
) -> list[dict[str, Any]]:
    results = load_frame_checkpoint(checkpoint_path)
    completed = {item.get("image_path") for item in results}
    total = len(frames)
    pending = [(index, frame) for index, frame in enumerate(frames, 1) if frame["image_path"] not in completed]
    if not pending:
        return sorted(results, key=frame_sort_key)
    if workers <= 1:
        for index, frame in pending:
            log_step(f"Azure OCR frame {index}/{total} at {format_time(frame.get('timestamp', 0))}")
            ocr = ocr_frame_with_azure(Path(frame["image_path"]), model_id=model_id)
            results.append({**frame, **ocr})
            results.sort(key=frame_sort_key)
            write_frame_checkpoint(checkpoint_path, results)
        return sorted(results, key=frame_sort_key)

    log_step(f"Azure OCR pending frames: {len(pending)} with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(ocr_frame_with_azure, Path(frame["image_path"]), model_id): (index, frame)
            for index, frame in pending
        }
        for future in as_completed(future_map):
            index, frame = future_map[future]
            ocr = future.result()
            log_step(f"Azure OCR complete frame {index}/{total} at {format_time(frame.get('timestamp', 0))}")
            results.append({**frame, **ocr})
            results.sort(key=frame_sort_key)
            write_frame_checkpoint(checkpoint_path, results)
    return results


def parse_frame_vision(text: str) -> FrameVisionExtraction:
    try:
        return FrameVisionExtraction.model_validate_json(text)
    except ValidationError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return FrameVisionExtraction.model_validate_json(text[start : end + 1])
        raise


def analyze_frame_with_vision(frame_path: Path, model: str | None = None, retries: int = 3) -> dict[str, Any]:
    load_dotenv_file()
    model = model or env_value("OPENAI_VISION_MODEL") or DEFAULT_VISION_MODEL
    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"), timeout=90)
    image_b64 = base64.b64encode(frame_path.read_bytes()).decode("ascii")
    for attempt in range(1, retries + 1):
        try:
            response = client.responses.create(
                model=model,
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "Extract all important information from this video frame. Include visible slide titles, "
                                    "on-screen text, diagrams, charts, UI screens, objects, labels, and concepts. "
                                    "Be exhaustive but concise. Return only valid JSON matching the schema."
                                ),
                            },
                            {"type": "input_image", "image_url": f"data:image/jpeg;base64,{image_b64}"},
                        ],
                    }
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "frame_vision_extraction",
                        "schema": frame_vision_schema(),
                        "strict": True,
                    }
                },
            )
            break
        except Exception as exc:
            if attempt == retries:
                raise
            delay = min(30, 2**attempt)
            log_step(f"Vision retry {attempt}/{retries - 1} for {frame_path.name}: {exc}. Waiting {delay}s")
            time.sleep(delay)
    structured = parse_frame_vision(response.output_text)
    return {
        "image_path": str(frame_path),
        "model": model,
        "analysis": structured.frame_summary,
        "structured": structured.model_dump(),
    }


def analyze_frame_with_vision_subprocess(
    frame_path: Path,
    timeout_seconds: int = DEFAULT_VISION_TIMEOUT_SECONDS,
    model: str | None = None,
    retries: int = 2,
) -> dict[str, Any]:
    command = [sys.executable, "-m", "app.pipelines.analyze_video_frame", str(frame_path)]
    if model:
        command.extend(["--model", model])
    last_error = ""
    for attempt in range(1, retries + 1):
        try:
            env = os.environ.copy()
            env["PYTHONIOENCODING"] = "utf-8"
            completed = subprocess.run(
                command,
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                env=env,
                timeout=timeout_seconds,
            )
        except subprocess.TimeoutExpired:
            last_error = f"vision subprocess timed out after {timeout_seconds}s"
        else:
            if completed.returncode == 0:
                return json.loads(completed.stdout)
            last_error = (completed.stderr or completed.stdout or "vision subprocess failed").strip()
        if attempt < retries:
            delay = min(30, 2**attempt)
            log_step(f"Vision subprocess retry {attempt}/{retries - 1} for {frame_path.name}: {last_error}. Waiting {delay}s")
            time.sleep(delay)
    raise RuntimeError(last_error)


def vision_failure_result(frame_path: Path, error: Exception) -> dict[str, Any]:
    return {
        "image_path": str(frame_path),
        "model": env_value("OPENAI_VISION_MODEL") or DEFAULT_VISION_MODEL,
        "analysis": "",
        "structured": FrameVisionExtraction(
            frame_summary="",
            visible_text=[],
            slide_title="",
            key_points=[],
            visual_elements=[],
            uncertainty_notes=[f"Vision extraction failed for this frame: {str(error)[:300]}"],
        ).model_dump(),
        "error": str(error),
    }


def analyze_frames_with_vision(
    frames: list[dict[str, Any]],
    checkpoint_path: Path | None = None,
    workers: int = 1,
    timeout_seconds: int = 0,
) -> list[dict[str, Any]]:
    results = load_frame_checkpoint(checkpoint_path)
    completed = {item.get("image_path") for item in results}
    total = len(frames)
    pending = [(index, frame) for index, frame in enumerate(frames, 1) if frame["image_path"] not in completed]
    if not pending:
        return sorted(results, key=frame_sort_key)
    if workers <= 1:
        for index, frame in pending:
            log_step(f"Vision analysis frame {index}/{total} at {format_time(frame.get('timestamp', 0))}")
            try:
                analysis = run_frame_vision(frame, timeout_seconds=timeout_seconds)
            except Exception as exc:
                log_step(f"Vision failed frame {index}/{total} at {format_time(frame.get('timestamp', 0))}: {exc}")
                analysis = vision_failure_result(Path(frame["image_path"]), exc)
            results.append({**frame, **analysis})
            results.sort(key=frame_sort_key)
            write_frame_checkpoint(checkpoint_path, results)
        return sorted(results, key=frame_sort_key)

    log_step(f"Vision pending frames: {len(pending)} with {workers} workers")
    with ThreadPoolExecutor(max_workers=workers) as executor:
        future_map = {
            executor.submit(run_frame_vision, frame, timeout_seconds): (index, frame)
            for index, frame in pending
        }
        for future in as_completed(future_map):
            index, frame = future_map[future]
            try:
                analysis = future.result()
            except Exception as exc:
                log_step(f"Vision failed frame {index}/{total} at {format_time(frame.get('timestamp', 0))}: {exc}")
                analysis = vision_failure_result(Path(frame["image_path"]), exc)
            log_step(f"Vision complete frame {index}/{total} at {format_time(frame.get('timestamp', 0))}")
            results.append({**frame, **analysis})
            results.sort(key=frame_sort_key)
            write_frame_checkpoint(checkpoint_path, results)
    return results


def run_frame_vision(frame: dict[str, Any], timeout_seconds: int = 0) -> dict[str, Any]:
    frame_path = Path(frame["image_path"])
    if timeout_seconds > 0:
        return analyze_frame_with_vision_subprocess(frame_path, timeout_seconds=timeout_seconds)
    return analyze_frame_with_vision(frame_path)


def load_frame_checkpoint(path: Path | None) -> list[dict[str, Any]]:
    if not path or not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8")).get("frames", [])


def write_frame_checkpoint(path: Path | None, frames: list[dict[str, Any]]) -> None:
    if not path:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_json({"frames": frames}), encoding="utf-8")


def frame_sort_key(frame: dict[str, Any]) -> tuple[float, int, str]:
    return (
        float(frame.get("timestamp") or 0),
        int(frame.get("frame_index") or 0),
        str(frame.get("image_path") or ""),
    )


def normalize_for_dedupe(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def is_duplicate_text(candidate: str, existing: list[str]) -> bool:
    key = normalize_for_dedupe(candidate)
    if not key:
        return True
    for item in existing:
        item_key = normalize_for_dedupe(item)
        if not item_key:
            continue
        if key == item_key or key in item_key or item_key in key:
            return True
        if SequenceMatcher(None, key, item_key).ratio() >= 0.9:
            return True
    return False


def add_unique_text(values: list[str], value: str) -> None:
    cleaned = " ".join(str(value or "").split())
    if cleaned and not is_duplicate_text(cleaned, values):
        values.append(cleaned)


def merge_frame_extractions(
    frames: list[dict[str, Any]],
    frame_ocr: list[dict[str, Any]],
    visual_analysis: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ocr_by_path = {item.get("image_path"): item for item in frame_ocr}
    vision_by_path = {item.get("image_path"): item for item in visual_analysis}
    merged = []
    for frame in frames:
        image_path = frame.get("image_path")
        ocr = ocr_by_path.get(image_path, {})
        vision = vision_by_path.get(image_path, {})
        structured = vision.get("structured") or {}
        facts: list[str] = []
        add_unique_text(facts, ocr.get("content", ""))
        if structured:
            add_unique_text(facts, structured.get("slide_title", ""))
            add_unique_text(facts, structured.get("frame_summary", ""))
            for value in structured.get("visible_text", []) or []:
                add_unique_text(facts, f"Visible text: {value}")
            for value in structured.get("key_points", []) or []:
                add_unique_text(facts, value)
            for element in structured.get("visual_elements", []) or []:
                parts = [
                    element.get("element_type", ""),
                    element.get("title", ""),
                    element.get("description", ""),
                    "; ".join(element.get("visible_labels", []) or []),
                ]
                add_unique_text(facts, " - ".join(part for part in parts if part))
        elif vision.get("analysis"):
            add_unique_text(facts, vision.get("analysis", ""))
        merged_text = "\n".join(facts)
        merged.append(
            {
                **frame,
                "content": ocr.get("content", ""),
                "vision": structured,
                "analysis": vision.get("analysis", ""),
                "merged_text": merged_text,
                "sources": ["azure_document_intelligence"] + (["openai_vision"] if vision else []),
                "dedupe": {
                    "merged_items": len(facts),
                    "ocr_chars": len(ocr.get("content", "") or ""),
                    "vision_items_present": bool(structured or vision.get("analysis")),
                },
            }
        )
    return merged


def collect_window_items(items: list[dict[str, Any]], start: float, end: float, key: str) -> list[dict[str, Any]]:
    collected = []
    for item in items:
        item_start = float(item.get("start", item.get("timestamp", 0)) or 0)
        item_end = float(item.get("end", item_start) or item_start)
        if item_start < end and item_end >= start:
            value = (item.get(key) or "").strip()
            if value:
                collected.append(item)
    return collected


def build_video_chunks(
    video_path: Path,
    metadata: dict[str, Any],
    transcript: dict[str, Any],
    frame_ocr: list[dict[str, Any]],
    visual_analysis: list[dict[str, Any]],
    merged_frames: list[dict[str, Any]] | None = None,
    chunk_window_seconds: float = DEFAULT_CHUNK_WINDOW_SECONDS,
) -> list[dict[str, Any]]:
    video_id = slugify(video_path.stem)
    duration = float(metadata.get("duration_seconds") or 0)
    if duration <= 0:
        transcript_segments = transcript.get("segments", [])
        duration = max((float(segment.get("end") or 0) for segment in transcript_segments), default=chunk_window_seconds)
    chunks = []
    window_start = 0.0
    while window_start < max(duration, chunk_window_seconds):
        window_end = min(window_start + chunk_window_seconds, duration or window_start + chunk_window_seconds)
        transcript_items = collect_window_items(transcript.get("segments", []), window_start, window_end, "text")
        merged_items = collect_window_items(merged_frames or [], window_start, window_end, "merged_text")
        ocr_items = collect_window_items(frame_ocr, window_start, window_end, "content") if not merged_items else []
        vision_items = collect_window_items(visual_analysis, window_start, window_end, "analysis") if not merged_items else []

        transcript_text = "\n".join(item["text"] for item in transcript_items if item.get("text"))
        merged_text = "\n".join(item["merged_text"] for item in merged_items if item.get("merged_text"))
        ocr_text = "\n".join(item["content"] for item in ocr_items if item.get("content"))
        vision_text = "\n".join(item["analysis"] for item in vision_items if item.get("analysis"))
        if not any([transcript_text.strip(), merged_text.strip(), ocr_text.strip(), vision_text.strip()]):
            window_start += chunk_window_seconds
            continue

        content = "\n\n".join(
            part
            for part in [
                f"Video transcript from {format_time(window_start)} to {format_time(window_end)}:\n{transcript_text}".strip(),
                f"Merged frame extraction:\n{merged_text}".strip() if merged_text.strip() else "",
                f"Azure frame OCR text:\n{ocr_text}".strip() if ocr_text.strip() else "",
                f"Visual frame analysis:\n{vision_text}".strip() if vision_text.strip() else "",
            ]
            if part
        )
        key_frame = (merged_items or vision_items or ocr_items or [{}])[0].get("image_path")
        frame_paths = [
            item.get("image_path")
            for item in [*merged_items, *ocr_items, *vision_items]
            if item.get("image_path")
        ]
        chunks.append(
            {
                "id": f"{video_id}-video-{len(chunks) + 1:06d}-{stable_id(content)}",
                "doc_id": video_id,
                "source_pdf": str(video_path),
                "source_path": str(video_path),
                "source_type": "video",
                "content": content,
                "content_type": "video_window",
                "page_numbers": [],
                "section_path": [Path(video_path).stem, f"{format_time(window_start)}-{format_time(window_end)}"],
                "section": f"{Path(video_path).stem} > {format_time(window_start)}-{format_time(window_end)}",
                "role": "video_window",
                "token_count": estimate_tokens(content),
                "metadata": {
                    "source": "video_pipeline",
                    "video_id": video_id,
                    "start_time": round(window_start, 3),
                    "end_time": round(window_end, 3),
                    "start_time_label": format_time(window_start),
                    "end_time_label": format_time(window_end),
                    "key_frame_path": key_frame,
                    "frame_paths": frame_paths,
                    "transcript_segments": transcript_items,
                    "merged_frames": merged_items,
                },
            }
        )
        window_start += chunk_window_seconds
    return chunks


def format_time(seconds: float) -> str:
    total = max(0, int(round(seconds)))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def process_video(
    video_path: Path,
    frame_interval_seconds: float = DEFAULT_FRAME_INTERVAL_SECONDS,
    chunk_window_seconds: float = DEFAULT_CHUNK_WINDOW_SECONDS,
    skip_transcription: bool = False,
    transcription_provider: str | None = None,
    audio_segment_seconds: float = DEFAULT_AUDIO_SEGMENT_SECONDS,
    transcription_max_tokens: int = DEFAULT_TRANSCRIPTION_MAX_TOKENS,
    skip_vision: bool = False,
    ocr_workers: int = 1,
    vision_workers: int = 1,
    vision_timeout_seconds: int = 0,
) -> tuple[list[dict[str, Any]], Path]:
    video_path = video_path.resolve()
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    paths = paths_for_video(video_path)
    paths.root.mkdir(parents=True, exist_ok=True)
    log_step(f"Processing {video_path.name}")
    log_step("Reading video metadata")
    metadata = probe_video(video_path)
    log_step(f"Duration {format_time(metadata.get('duration_seconds', 0))}")
    resolved_transcription_provider = resolve_transcription_provider(transcription_provider)
    transcription_disabled = skip_transcription or resolved_transcription_provider == "none"
    if transcription_disabled:
        log_step("Skipping audio transcription")
    else:
        log_step(f"Audio transcription provider: {resolved_transcription_provider}")
        log_step(f"Extracting audio to {paths.audio}")
        if paths.audio.exists() and paths.audio.stat().st_size >= 1024:
            log_step(f"Using existing audio {paths.audio}")
        else:
            if paths.audio.exists():
                log_step(f"Regenerating incomplete audio {paths.audio}")
                paths.audio.unlink(missing_ok=True)
            extract_audio(video_path, paths.audio)
    log_step(f"Extracting frames every {frame_interval_seconds:g}s to {paths.frames_dir}")
    existing_frames = sorted(paths.frames_dir.glob("frame_*.jpg"))
    if existing_frames:
        log_step(f"Using {len(existing_frames)} existing frames")
        frames = [
            {"frame_index": index, "timestamp": round((index - 1) * frame_interval_seconds, 3), "image_path": str(frame_path)}
            for index, frame_path in enumerate(existing_frames, 1)
        ]
    else:
        frames = extract_frames(video_path, paths.frames_dir, interval_seconds=frame_interval_seconds)
    log_step(f"Extracted {len(frames)} frames")
    if transcription_disabled:
        transcript = {
            "text": "",
            "segments": [],
            "skipped": True,
            "reason": "audio transcription disabled for this ingestion run",
        }
    elif paths.transcript.exists():
        log_step(f"Using existing transcript {paths.transcript}")
        transcript = json.loads(paths.transcript.read_text(encoding="utf-8"))
    else:
        log_step("Transcribing audio")
        transcript = transcribe_audio(
            paths.audio,
            provider=resolved_transcription_provider,
            duration_seconds=float(metadata.get("duration_seconds") or 0),
            segment_seconds=audio_segment_seconds,
            max_tokens=transcription_max_tokens,
            checkpoint_path=paths.transcript,
        )
        paths.transcript.write_text(dump_json(transcript), encoding="utf-8")
    log_step(f"Transcript segments: {len(transcript.get('segments', []))}")
    existing_ocr = load_frame_checkpoint(paths.frame_ocr)
    if len(existing_ocr) >= len(frames):
        log_step(f"Using complete frame OCR {paths.frame_ocr}")
        frame_ocr = existing_ocr
    else:
        if existing_ocr:
            log_step(f"Resuming Azure Document Intelligence OCR on frames ({len(existing_ocr)}/{len(frames)} complete)")
        else:
            log_step("Running Azure Document Intelligence OCR on frames")
        frame_ocr = ocr_frames(frames, checkpoint_path=paths.frame_ocr, workers=max(1, ocr_workers))
    log_step(f"OCR frames complete: {len(frame_ocr)}")
    if skip_vision:
        log_step("Skipping frame vision summaries")
        visual_analysis = []
    else:
        existing_vision = load_frame_checkpoint(paths.visual_analysis)
        if len(existing_vision) >= len(frames):
            log_step(f"Using complete frame vision summaries {paths.visual_analysis}")
            visual_analysis = existing_vision
        else:
            if existing_vision:
                log_step(f"Resuming frame vision summaries ({len(existing_vision)}/{len(frames)} complete)")
            else:
                log_step("Running frame vision summaries")
            visual_analysis = analyze_frames_with_vision(
                frames,
                checkpoint_path=paths.visual_analysis,
                workers=max(1, vision_workers),
                timeout_seconds=max(0, vision_timeout_seconds),
            )
    log_step("Merging Azure OCR and frame vision results")
    merged_frames = merge_frame_extractions(frames, frame_ocr, visual_analysis)
    paths.merged_analysis.write_text(dump_json({"frames": merged_frames}), encoding="utf-8")
    log_step("Building timestamped video chunks")
    chunks = build_video_chunks(
        video_path=video_path,
        metadata=metadata,
        transcript=transcript,
        frame_ocr=frame_ocr,
        visual_analysis=visual_analysis,
        merged_frames=merged_frames,
        chunk_window_seconds=chunk_window_seconds,
    )

    paths.transcript.write_text(dump_json(transcript), encoding="utf-8")
    paths.frame_ocr.write_text(dump_json({"frames": frame_ocr}), encoding="utf-8")
    paths.visual_analysis.write_text(dump_json({"frames": visual_analysis}), encoding="utf-8")
    write_jsonl(paths.chunks, chunks)
    summary = {
        "video_id": paths.video_id,
        "source_video": str(video_path),
        "metadata": metadata,
        "frames": len(frames),
        "transcript_segments": len(transcript.get("segments", [])),
        "ocr_frames": len(frame_ocr),
        "vision_frames": len(visual_analysis),
        "merged_frames": len(merged_frames),
        "merged_analysis": str(paths.merged_analysis),
        "chunk_count": len(chunks),
        "chunks": str(paths.chunks),
    }
    paths.summary.write_text(dump_json(summary), encoding="utf-8")
    log_step(f"Done {video_path.name}: {len(chunks)} chunks -> {paths.chunks}")
    return chunks, paths.chunks
