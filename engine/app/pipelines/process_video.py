from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from app.clients.lancedb_store import create_or_replace_index
from app.services.chunk_document import load_chunks_jsonl
from app.services.embed_chunks import embed_chunks_file
from app.services.video_processing import process_video


def main() -> None:
    parser = argparse.ArgumentParser(description="Process a video into timestamped multimodal RAG chunks.")
    parser.add_argument("input", type=Path, help="Video file or directory of videos.")
    parser.add_argument("--frame-interval-seconds", type=float, default=5.0)
    parser.add_argument("--chunk-window-seconds", type=float, default=30.0)
    parser.add_argument("--skip-transcription", action="store_true", help="Skip audio transcription.")
    parser.add_argument(
        "--transcription-provider",
        default="auto",
        choices=["auto", "openai", "databricks", "none"],
        help="Audio transcription provider. In auto mode, Databricks LLM_PROVIDER uses Databricks audio.",
    )
    parser.add_argument("--audio-segment-seconds", type=float, default=600.0, help="Chunk audio before Databricks transcription.")
    parser.add_argument("--transcription-max-tokens", type=int, default=4096, help="Max output tokens per audio transcription segment.")
    parser.add_argument("--skip-vision", action="store_true", help="Skip frame visual summaries.")
    parser.add_argument(
        "--vision-provider",
        default="auto",
        choices=["auto", "openai", "databricks", "none"],
        help="Frame vision provider. In auto mode, Databricks LLM_PROVIDER uses Databricks vision.",
    )
    parser.add_argument("--vision-model", default="", help="Optional frame vision model or Databricks endpoint name.")
    parser.add_argument(
        "--quality-provider",
        default="auto",
        choices=["auto", "databricks", "openai", "heuristic", "none"],
        help="Provider for retrieval-quality text cleanup before embeddings.",
    )
    parser.add_argument("--quality-model", default="", help="Optional quality cleanup model or Databricks endpoint name.")
    parser.add_argument("--quality-required", action="store_true", help="Fail if LLM quality cleanup fails.")
    parser.add_argument("--skip-quality-enrichment", action="store_true", help="Use deterministic quality scoring only.")
    parser.add_argument("--quality-glossary-path", default="", help="Optional JSON glossary file with terms and definitions.")
    parser.add_argument("--quality-min-llm-chars", type=int, default=80)
    parser.add_argument("--quality-max-input-chars", type=int, default=6500)
    parser.add_argument("--quality-max-output-tokens", type=int, default=1800)
    parser.add_argument("--embed", action="store_true", help="Embed the generated chunks.")
    parser.add_argument("--rebuild-index", action="store_true", help="Rebuild LanceDB from all embedded files after embedding.")
    parser.add_argument("--parallel-videos", type=int, default=1, help="Number of videos to process concurrently.")
    parser.add_argument("--ocr-workers", type=int, default=1, help="Azure OCR frame workers per video.")
    parser.add_argument("--vision-workers", type=int, default=1, help="OpenAI vision frame workers per video.")
    parser.add_argument(
        "--vision-timeout-seconds",
        type=int,
        default=0,
        help="Hard timeout for each vision frame. Uses a subprocess per frame when greater than zero.",
    )
    args = parser.parse_args()

    video_paths = discover_videos(args.input)
    processed = []
    try:
        if args.parallel_videos <= 1:
            for index, video_path in enumerate(video_paths, 1):
                processed.append(process_one_video(index, len(video_paths), video_path, args))
        else:
            print(f"[video] Processing {len(video_paths)} videos with {args.parallel_videos} video workers", flush=True)
            with ThreadPoolExecutor(max_workers=args.parallel_videos) as executor:
                future_map = {
                    executor.submit(process_one_video, index, len(video_paths), video_path, args): index
                    for index, video_path in enumerate(video_paths, 1)
                }
                for future in as_completed(future_map):
                    processed.append(future.result())
            processed.sort(key=lambda item: item["order"])
    except Exception as exc:
        print(f"Video processing failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    payload = {"processed": [{k: v for k, v in item.items() if k != "order"} for item in processed], "videos": len(processed)}
    if args.rebuild_index:
        print("[video] Rebuilding combined LanceDB index", flush=True)
        embedded_files = sorted(Path("output/embeddings").glob("*-embedded.jsonl"))
        all_chunks = []
        for path in embedded_files:
            all_chunks.extend(load_chunks_jsonl(path))
        create_or_replace_index(all_chunks)
        payload["indexed_chunks"] = len(all_chunks)
        payload["embedded_files"] = [str(path) for path in embedded_files]
        print(f"[video] Rebuilt index with {len(all_chunks)} chunks", flush=True)
    print(json.dumps(payload, indent=2))


def process_one_video(index: int, total: int, video_path: Path, args: argparse.Namespace) -> dict[str, object]:
    print(f"[video] Starting video {index}/{total}: {video_path}", flush=True)
    chunks, chunk_path = process_video(
        video_path=video_path,
        frame_interval_seconds=args.frame_interval_seconds,
        chunk_window_seconds=args.chunk_window_seconds,
        skip_transcription=args.skip_transcription,
        transcription_provider=args.transcription_provider,
        audio_segment_seconds=args.audio_segment_seconds,
        transcription_max_tokens=args.transcription_max_tokens,
        skip_vision=args.skip_vision,
        vision_provider=args.vision_provider,
        vision_model=args.vision_model or None,
        ocr_workers=max(1, args.ocr_workers),
        vision_workers=max(1, args.vision_workers),
        vision_timeout_seconds=max(0, args.vision_timeout_seconds),
    )
    item: dict[str, object] = {
        "order": index,
        "video": str(video_path),
        "chunk_path": str(chunk_path),
        "chunks": len(chunks),
    }
    if args.embed:
        from app.services.quality_enrichment import QualityEnrichmentConfig, enrich_chunks_file

        print(f"[video] Enriching chunk quality for {video_path.name}", flush=True)
        quality_config = QualityEnrichmentConfig.from_args(args)
        enriched_chunks, quality_path, quality_summary_path, quality_summary = enrich_chunks_file(chunk_path, config=quality_config)
        print(f"[video] Embedding chunks for {video_path.name}", flush=True)
        _, embedded_path = embed_chunks_file(quality_path)
        item["raw_chunk_path"] = str(chunk_path)
        item["chunk_path"] = str(quality_path)
        item["quality_summary_path"] = str(quality_summary_path)
        item["quality"] = quality_summary
        item["chunks"] = len(enriched_chunks)
        item["embedded_path"] = str(embedded_path)
    return item


def discover_videos(input_path: Path) -> list[Path]:
    if input_path.is_file():
        return [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(input_path)
    videos = []
    for pattern in ("*.mp4", "*.mov", "*.m4v", "*.mkv", "*.webm"):
        videos.extend(sorted(input_path.glob(pattern)))
    if not videos:
        raise RuntimeError(f"No supported video files found in {input_path}")
    return videos


if __name__ == "__main__":
    main()
