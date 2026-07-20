from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.databricks_model_serving import chat_completion
from app.clients.document_intelligence import env_value, load_dotenv_file
from app.config import databricks_chat_endpoint, llm_provider
from app.rag.prompt_store import prompt_text
from app.services.chunk_document import estimate_tokens, load_chunks_jsonl, write_jsonl
from app.utils.files import output_dir
from app.utils.logging import dump_json


DEFAULT_QUALITY_MODEL = "gpt-4.1-mini"
DEFAULT_MIN_LLM_CHARS = 80
DEFAULT_MAX_INPUT_CHARS = 6500
DEFAULT_MAX_OUTPUT_TOKENS = 1800
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]*")
NON_TEXT_RE = re.compile(r"[^A-Za-z0-9\s.,;:!?()\[\]{}<>/|+=_@#$%&*'\"\-]")
MULTISPACE_RE = re.compile(r"[ \t]+")
MULTIBLANK_RE = re.compile(r"\n{3,}")
REPEATED_PUNCT_RE = re.compile(r"([!?.,;:])\1{3,}")

HINGLISH_MARKERS = {
    "acha",
    "agar",
    "basically",
    "dekho",
    "hai",
    "hain",
    "hoga",
    "ka",
    "kar",
    "karna",
    "ke",
    "ki",
    "kya",
    "matlab",
    "mein",
    "nahi",
    "samjho",
    "theek",
    "wala",
    "wale",
    "ye",
    "yeh",
}

DEFAULT_GLOSSARY = [
    {"term": "KT", "definition": "knowledge transfer"},
    {"term": "RAG", "definition": "retrieval augmented generation"},
    {"term": "Databricks Volume", "definition": "managed Unity Catalog storage location"},
    {"term": "Auto Loader", "definition": "Databricks incremental file ingestion capability"},
    {"term": "Bronze", "definition": "raw data or raw file audit layer"},
    {"term": "Silver", "definition": "cleaned and extracted content layer"},
    {"term": "Gold", "definition": "retrieval-ready embeddings and metadata layer"},
    {"term": "Unity Catalog", "definition": "Databricks governance catalog"},
    {"term": "embedding", "definition": "vector representation used for semantic retrieval"},
    {"term": "chunk", "definition": "retrieval-sized content segment"},
    {"term": "LanceDB", "definition": "local vector index used by the app"},
    {"term": "OCR", "definition": "optical character recognition"},
    {"term": "LLM vision", "definition": "multimodal model analysis of images or video frames"},
]


class QualityEnrichmentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    normalized_text: str = Field(description="Retrieval-ready normalized text that preserves the source meaning.")
    detected_languages: list[str] = Field(default_factory=list, description="Language codes such as en, hi, hi-en, unknown.")
    glossary_terms: list[str] = Field(default_factory=list, description="Domain glossary terms found or applied.")
    corrections: list[str] = Field(default_factory=list, description="Short descriptions of vocabulary, language, or cleanup corrections.")
    quality_score: float = Field(ge=0.0, le=1.0, description="Estimated extraction quality after cleanup.")
    needs_review: bool = Field(default=False, description="True when the chunk may still be ambiguous or low quality.")
    quality_notes: list[str] = Field(default_factory=list, description="Brief audit notes about remaining uncertainty or extraction quality.")


@dataclass(frozen=True)
class QualityEnrichmentConfig:
    provider: str = "auto"
    model: str = ""
    required: bool = False
    min_llm_chars: int = DEFAULT_MIN_LLM_CHARS
    max_input_chars: int = DEFAULT_MAX_INPUT_CHARS
    max_output_tokens: int = DEFAULT_MAX_OUTPUT_TOKENS
    glossary_path: str = ""
    glossary: list[dict[str, str]] = field(default_factory=lambda: list(DEFAULT_GLOSSARY))

    @classmethod
    def from_args(cls, args: Any) -> "QualityEnrichmentConfig":
        load_dotenv_file()
        skip = bool(getattr(args, "skip_quality_enrichment", False))
        provider = "none" if skip else getattr(args, "quality_provider", "") or env_value("QUALITY_ENRICHMENT_PROVIDER") or "auto"
        glossary_path = getattr(args, "quality_glossary_path", "") or env_value("QUALITY_GLOSSARY_PATH") or ""
        return cls(
            provider=provider,
            model=getattr(args, "quality_model", "") or env_value("QUALITY_ENRICHMENT_MODEL") or "",
            required=bool(getattr(args, "quality_required", False) or truthy_env("QUALITY_ENRICHMENT_REQUIRED")),
            min_llm_chars=safe_int(getattr(args, "quality_min_llm_chars", None), DEFAULT_MIN_LLM_CHARS),
            max_input_chars=safe_int(getattr(args, "quality_max_input_chars", None), DEFAULT_MAX_INPUT_CHARS),
            max_output_tokens=safe_int(getattr(args, "quality_max_output_tokens", None), DEFAULT_MAX_OUTPUT_TOKENS),
            glossary_path=glossary_path,
            glossary=load_glossary(glossary_path),
        )

    @property
    def resolved_provider(self) -> str:
        load_dotenv_file()
        provider = (self.provider or "auto").strip().lower()
        if provider == "auto":
            if llm_provider() == "databricks":
                return "databricks"
            if env_value("OPENAI_API_KEY", "OPANAI_API_KEY"):
                return "openai"
            return "heuristic"
        if provider in {"skip", "disabled"}:
            return "none"
        return provider

    @property
    def resolved_model(self) -> str:
        load_dotenv_file()
        if self.model:
            return self.model
        if self.resolved_provider == "databricks":
            return databricks_chat_endpoint()
        return env_value("OPENAI_QUALITY_MODEL") or env_value("OPENAI_ANSWER_MODEL") or DEFAULT_QUALITY_MODEL

    def fingerprint(self) -> str:
        payload = {
            "provider": self.resolved_provider,
            "model": self.resolved_model,
            "min_llm_chars": self.min_llm_chars,
            "max_input_chars": self.max_input_chars,
            "max_output_tokens": self.max_output_tokens,
            "glossary": self.glossary,
        }
        return short_hash(dump_json(payload))


def truthy_env(name: str) -> bool:
    value = (env_value(name) or "").strip().lower()
    return value in {"1", "true", "yes", "y", "on"}


def safe_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return default
        return int(value)
    except Exception:
        return default


def clamp_score(value: Any, default: float = 0.75) -> float:
    try:
        numeric = float(value)
    except Exception:
        numeric = default
    return round(max(0.0, min(1.0, numeric)), 3)


def short_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()[:16]


def content_hash(value: str) -> str:
    return hashlib.sha256(str(value or "").encode("utf-8")).hexdigest()


def load_glossary(path: str | None = None) -> list[dict[str, str]]:
    if not path:
        return list(DEFAULT_GLOSSARY)
    try:
        payload = json.loads(Path(path).expanduser().read_text(encoding="utf-8"))
    except Exception:
        return list(DEFAULT_GLOSSARY)
    items = payload.get("terms") if isinstance(payload, dict) else payload
    if not isinstance(items, list):
        return list(DEFAULT_GLOSSARY)
    glossary: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            glossary.append({"term": item, "definition": ""})
        elif isinstance(item, dict) and item.get("term"):
            glossary.append({"term": str(item.get("term") or ""), "definition": str(item.get("definition") or "")})
    return glossary or list(DEFAULT_GLOSSARY)


def require_all_properties(schema: dict[str, Any]) -> dict[str, Any]:
    if schema.get("type") == "object" and "properties" in schema:
        schema["required"] = list(schema["properties"].keys())
        schema["additionalProperties"] = False
    for value in schema.get("$defs", {}).values():
        if isinstance(value, dict):
            require_all_properties(value)
    for key in ("items", "anyOf", "oneOf", "allOf"):
        value = schema.get(key)
        if isinstance(value, dict):
            require_all_properties(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    require_all_properties(item)
    return schema


def quality_schema() -> dict[str, Any]:
    return require_all_properties(QualityEnrichmentResult.model_json_schema())


def strip_json_fence(text: str) -> str:
    cleaned = str(text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    return cleaned.strip()


def parse_quality_result(text: str) -> QualityEnrichmentResult:
    cleaned = strip_json_fence(text)
    try:
        return QualityEnrichmentResult.model_validate_json(cleaned)
    except ValidationError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            return QualityEnrichmentResult.model_validate_json(cleaned[start : end + 1])
        raise


def normalize_whitespace(text: str) -> str:
    lines = [MULTISPACE_RE.sub(" ", line).strip() for line in str(text or "").splitlines()]
    cleaned = "\n".join(line for line in lines if line)
    cleaned = MULTIBLANK_RE.sub("\n\n", cleaned)
    return cleaned.strip()


def detected_languages(text: str) -> list[str]:
    normalized_words = {word.lower() for word in WORD_RE.findall(text)}
    has_devanagari = bool(DEVANAGARI_RE.search(text))
    has_latin = bool(LATIN_RE.search(text))
    has_hinglish = bool(normalized_words & HINGLISH_MARKERS)
    if has_devanagari and has_latin:
        return ["hi-en"]
    if has_devanagari:
        return ["hi"]
    if has_latin and has_hinglish:
        return ["hi-en"]
    if has_latin:
        return ["en"]
    return ["unknown"]


def glossary_terms(text: str, glossary: list[dict[str, str]]) -> list[str]:
    lower_text = text.lower()
    found = []
    for item in glossary:
        term = str(item.get("term") or "").strip()
        if term and term.lower() in lower_text:
            found.append(term)
    return sorted(set(found), key=str.lower)


def noise_features(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    if not stripped:
        return {
            "char_count": 0,
            "non_text_ratio": 1.0,
            "short_line_ratio": 1.0,
            "repeated_punctuation": 0,
            "replacement_chars": 0,
            "long_token_ratio": 0.0,
        }
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    words = WORD_RE.findall(stripped)
    long_words = [word for word in words if len(word) > 28]
    return {
        "char_count": len(stripped),
        "non_text_ratio": len(NON_TEXT_RE.findall(stripped)) / max(1, len(stripped)),
        "short_line_ratio": sum(1 for line in lines if len(line) <= 2) / max(1, len(lines)),
        "repeated_punctuation": len(REPEATED_PUNCT_RE.findall(stripped)),
        "replacement_chars": stripped.count("\ufffd"),
        "long_token_ratio": len(long_words) / max(1, len(words)),
    }


def heuristic_quality(raw_text: str, glossary: list[dict[str, str]]) -> QualityEnrichmentResult:
    normalized = normalize_whitespace(raw_text)
    languages = detected_languages(normalized)
    terms = glossary_terms(normalized, glossary)
    features = noise_features(normalized)
    score = 0.94
    notes: list[str] = []
    corrections: list[str] = []

    if normalized != str(raw_text or "").strip():
        corrections.append("Whitespace normalized")
    if "hi" in "-".join(languages):
        score -= 0.12
        notes.append("Hindi or Hinglish detected; LLM normalization is recommended.")
    if features["char_count"] < 80:
        score -= 0.04
        notes.append("Very short extraction segment.")
    if features["non_text_ratio"] > 0.04:
        score -= min(0.25, features["non_text_ratio"] * 2)
        notes.append("Possible OCR noise detected.")
    if features["short_line_ratio"] > 0.35:
        score -= 0.08
        notes.append("Many very short OCR lines detected.")
    if features["repeated_punctuation"]:
        score -= min(0.08, features["repeated_punctuation"] * 0.02)
        notes.append("Repeated punctuation cleaned by downstream model if enabled.")
    if features["replacement_chars"]:
        score -= min(0.25, features["replacement_chars"] * 0.04)
        notes.append("Unicode replacement characters detected.")
    if features["long_token_ratio"] > 0.06:
        score -= 0.08
        notes.append("Long-token OCR artifacts may be present.")
    if terms:
        score += 0.02

    score = clamp_score(score)
    return QualityEnrichmentResult(
        normalized_text=normalized,
        detected_languages=languages,
        glossary_terms=terms,
        corrections=corrections,
        quality_score=score,
        needs_review=score < 0.72,
        quality_notes=notes,
    )


def should_call_llm(raw_text: str, heuristic: QualityEnrichmentResult, config: QualityEnrichmentConfig) -> bool:
    if len(raw_text.strip()) < max(1, config.min_llm_chars):
        return False
    if config.resolved_provider not in {"databricks", "openai"}:
        return False
    return True


def glossary_text(glossary: list[dict[str, str]]) -> str:
    rows = []
    for item in glossary:
        term = item.get("term", "")
        definition = item.get("definition", "")
        rows.append(f"- {term}: {definition}" if definition else f"- {term}")
    return "\n".join(rows)


def source_metadata(chunk: dict[str, Any]) -> dict[str, Any]:
    metadata = chunk.get("metadata") or {}
    return {
        "id": chunk.get("id", ""),
        "doc_id": chunk.get("doc_id", ""),
        "source_type": chunk.get("source_type") or metadata.get("source_type") or "document",
        "content_type": chunk.get("content_type", ""),
        "role": chunk.get("role", ""),
        "section": chunk.get("section") or " > ".join(chunk.get("section_path") or []),
        "page_numbers": chunk.get("page_numbers") or [],
        "start_time_label": metadata.get("start_time_label", ""),
        "end_time_label": metadata.get("end_time_label", ""),
    }


def build_quality_prompt(
    chunk: dict[str, Any],
    raw_text: str,
    heuristic: QualityEnrichmentResult,
    config: QualityEnrichmentConfig,
) -> str:
    return prompt_text(
        "dynamic",
        "quality_enrichment",
        "human",
        source_metadata=dump_json(source_metadata(chunk)),
        glossary=glossary_text(config.glossary),
        heuristic=dump_json(heuristic.model_dump()),
        schema=json.dumps(quality_schema(), indent=2),
        raw_content=raw_text[: config.max_input_chars],
    )


def enrich_with_databricks(
    chunk: dict[str, Any],
    raw_text: str,
    heuristic: QualityEnrichmentResult,
    config: QualityEnrichmentConfig,
) -> QualityEnrichmentResult:
    content = chat_completion(
        endpoint=config.resolved_model,
        messages=[
            {"role": "system", "content": prompt_text("static", "quality_enrichment", "system")},
            {"role": "user", "content": build_quality_prompt(chunk, raw_text, heuristic, config)},
        ],
        temperature=0.0,
        max_tokens=config.max_output_tokens,
    )
    return parse_quality_result(content)


def enrich_with_openai(
    chunk: dict[str, Any],
    raw_text: str,
    heuristic: QualityEnrichmentResult,
    config: QualityEnrichmentConfig,
) -> QualityEnrichmentResult:
    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
    response = client.responses.create(
        model=config.resolved_model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": prompt_text("static", "quality_enrichment", "system")}]},
            {"role": "user", "content": [{"type": "input_text", "text": build_quality_prompt(chunk, raw_text, heuristic, config)}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "quality_enrichment",
                "schema": quality_schema(),
                "strict": True,
            }
        },
    )
    return parse_quality_result(response.output_text)


def sanitize_result(raw_text: str, result: QualityEnrichmentResult, fallback: QualityEnrichmentResult) -> QualityEnrichmentResult:
    normalized = normalize_whitespace(result.normalized_text)
    if not normalized:
        normalized = fallback.normalized_text or normalize_whitespace(raw_text)
    languages = result.detected_languages or fallback.detected_languages or detected_languages(normalized)
    terms = sorted(set([*fallback.glossary_terms, *result.glossary_terms]), key=str.lower)
    return QualityEnrichmentResult(
        normalized_text=normalized,
        detected_languages=languages,
        glossary_terms=terms,
        corrections=list(dict.fromkeys([*fallback.corrections, *result.corrections])),
        quality_score=clamp_score(result.quality_score, fallback.quality_score),
        needs_review=bool(result.needs_review or result.quality_score < 0.72),
        quality_notes=list(dict.fromkeys([*fallback.quality_notes, *result.quality_notes])),
    )


def apply_quality_result(
    chunk: dict[str, Any],
    result: QualityEnrichmentResult,
    raw_text: str,
    *,
    provider: str,
    model: str,
    config_hash: str,
    error: str = "",
) -> dict[str, Any]:
    item = dict(chunk)
    metadata = dict(item.get("metadata") or {})
    metadata["raw_content"] = raw_text
    metadata["quality"] = {
        **result.model_dump(),
        "provider": provider,
        "model": model,
        "raw_content_sha256": content_hash(raw_text),
        "config_hash": config_hash,
        "error": error,
    }
    item["content"] = result.normalized_text
    item["token_count"] = estimate_tokens(result.normalized_text)
    item["metadata"] = metadata
    item["quality_score"] = result.quality_score
    item["detected_languages"] = result.detected_languages
    item["needs_review"] = result.needs_review
    return item


def enrich_chunk(chunk: dict[str, Any], config: QualityEnrichmentConfig) -> dict[str, Any]:
    load_dotenv_file()
    raw_text = str(chunk.get("content") or "")
    fallback = heuristic_quality(raw_text, config.glossary)
    provider = config.resolved_provider
    model = config.resolved_model if provider in {"databricks", "openai"} else ""
    config_hash = config.fingerprint()
    if provider in {"none", "heuristic"} or not should_call_llm(raw_text, fallback, config):
        return apply_quality_result(chunk, fallback, raw_text, provider=provider, model=model, config_hash=config_hash)

    try:
        if provider == "databricks":
            result = enrich_with_databricks(chunk, raw_text, fallback, config)
        elif provider == "openai":
            result = enrich_with_openai(chunk, raw_text, fallback, config)
        else:
            result = fallback
        result = sanitize_result(raw_text, result, fallback)
        return apply_quality_result(chunk, result, raw_text, provider=provider, model=model, config_hash=config_hash)
    except Exception as exc:
        if config.required:
            raise RuntimeError(f"Quality enrichment failed for chunk {chunk.get('id')}: {type(exc).__name__}: {exc}") from exc
        fallback.quality_notes.append(f"LLM quality enrichment failed; heuristic fallback used: {type(exc).__name__}")
        fallback.needs_review = True
        fallback.quality_score = min(fallback.quality_score, 0.7)
        return apply_quality_result(
            chunk,
            fallback,
            raw_text,
            provider="heuristic_fallback",
            model=model,
            config_hash=config_hash,
            error=f"{type(exc).__name__}: {exc}",
        )


def quality_output_path(chunk_path: Path) -> Path:
    name = chunk_path.name
    if name.endswith("-quality-chunks.jsonl"):
        return chunk_path
    if name.endswith("-chunks.jsonl"):
        name = name.replace("-chunks.jsonl", "-quality-chunks.jsonl")
    else:
        name = f"{chunk_path.stem}-quality.jsonl"
    return output_dir("quality") / name


def quality_summary_path(output_path: Path) -> Path:
    if output_path.name.endswith("-quality-chunks.jsonl"):
        return output_path.with_name(output_path.name.replace("-quality-chunks.jsonl", "-quality-summary.json"))
    return output_path.with_suffix(".summary.json")


def cached_quality_chunks(raw_chunks: list[dict[str, Any]], output_path: Path, config_hash: str) -> list[dict[str, Any]] | None:
    if not output_path.exists():
        return None
    existing = load_chunks_jsonl(output_path)
    if len(existing) != len(raw_chunks):
        return None
    existing_by_id = {chunk.get("id"): chunk for chunk in existing}
    for raw_chunk in raw_chunks:
        enriched = existing_by_id.get(raw_chunk.get("id"))
        if not enriched:
            return None
        quality = ((enriched.get("metadata") or {}).get("quality") or {})
        if quality.get("raw_content_sha256") != content_hash(str(raw_chunk.get("content") or "")):
            return None
        if quality.get("config_hash") != config_hash:
            return None
    return existing


def build_summary(chunks: list[dict[str, Any]], output_path: Path, provider: str, model: str, reused_cache: bool) -> dict[str, Any]:
    scores = [float(chunk.get("quality_score") or ((chunk.get("metadata") or {}).get("quality") or {}).get("quality_score") or 0.0) for chunk in chunks]
    languages = sorted({language for chunk in chunks for language in (chunk.get("detected_languages") or [])})
    review_count = sum(1 for chunk in chunks if chunk.get("needs_review"))
    return {
        "chunk_count": len(chunks),
        "quality_provider": provider,
        "quality_model": model,
        "average_quality_score": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "needs_review_count": review_count,
        "detected_languages": languages,
        "reused_cache": reused_cache,
        "output": str(output_path),
    }


def enrich_chunks_file(
    chunk_path: Path,
    config: QualityEnrichmentConfig | None = None,
    output_path: Path | None = None,
) -> tuple[list[dict[str, Any]], Path, Path, dict[str, Any]]:
    config = config or QualityEnrichmentConfig()
    raw_chunks = load_chunks_jsonl(chunk_path)
    output_path = output_path or quality_output_path(chunk_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path = quality_summary_path(output_path)
    config_hash = config.fingerprint()
    cached = cached_quality_chunks(raw_chunks, output_path, config_hash)
    provider = config.resolved_provider
    model = config.resolved_model if provider in {"databricks", "openai"} else ""
    if cached is not None:
        summary = build_summary(cached, output_path, provider, model, reused_cache=True)
        summary_path.write_text(dump_json(summary), encoding="utf-8")
        return cached, output_path, summary_path, summary

    enriched = [enrich_chunk(chunk, config) for chunk in raw_chunks]
    write_jsonl(output_path, enriched)
    summary = build_summary(enriched, output_path, provider, model, reused_cache=False)
    summary_path.write_text(dump_json(summary), encoding="utf-8")
    return enriched, output_path, summary_path, summary
