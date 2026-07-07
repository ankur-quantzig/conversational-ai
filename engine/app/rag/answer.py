from __future__ import annotations

import json
import time
from copy import deepcopy
from hashlib import sha256
from threading import RLock
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.databricks_model_serving import chat_completion
from app.clients.document_intelligence import env_value, load_dotenv_file
from app.config import databricks_chat_endpoint, llm_provider
from app.rag.prompt_store import prompt_text


DEFAULT_ANSWER_MODEL = "gpt-4.1-mini"
CACHE_TTL_SECONDS = 300


class AnswerCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source_index: int = Field(description="1-based index of the retrieved context block.")
    pages: list[int] = Field(default_factory=list, description="PDF page numbers supporting this citation.")
    start_time_label: str = Field(default="", description="Video start timestamp such as 02:14, when the source is video.")
    end_time_label: str = Field(default="", description="Video end timestamp such as 02:48, when the source is video.")
    quote: str = Field(description="Short supporting quote or paraphrase from the retrieved context.")


class RagAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    heading: str = Field(description="Short professional heading for the response, no more than 8 words.")
    answer: str = Field(description="Grounded answer to the user question.")
    citations: list[AnswerCitation] = Field(default_factory=list)
    confidence: str = Field(description="One of: high, medium, low.")
    confidence_score: float = Field(ge=0.0, le=1.0, description="Answer quality confidence from 0 to 1.")
    missing_information: str = Field(default="", description="What is missing if the context is insufficient.")


class FollowUpQuestions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str] = Field(default_factory=list, max_length=4)


class ConversationQuestionPlan(BaseModel):
    model_config = ConfigDict(extra="forbid")

    is_follow_up: bool = Field(description="True when the latest question depends on the prior conversation.")
    standalone_question: str = Field(description="Original question or a rewritten standalone follow-up question.")
    reason: str = Field(description="Brief reason for the classification.")


class ResponseDiagram(BaseModel):
    model_config = ConfigDict(extra="forbid")

    should_show: bool = Field(description="True when a diagram materially improves the answer.")
    title: str = Field(default="", description="Short diagram title.")
    diagram_type: str = Field(default="", description="Currently use mermaid or empty string.")
    code: str = Field(default="", description="Mermaid diagram code when should_show is true.")
    reason: str = Field(default="", description="Brief reason for showing or skipping the diagram.")


class _TtlCache:
    def __init__(self, ttl_seconds: int = CACHE_TTL_SECONDS, maxsize: int = 128) -> None:
        self.ttl_seconds = ttl_seconds
        self.maxsize = maxsize
        self._lock = RLock()
        self._items: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        now = time.monotonic()
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= now:
                self._items.pop(key, None)
                return None
            return deepcopy(value)

    def set(self, key: str, value: Any) -> None:
        with self._lock:
            if len(self._items) >= self.maxsize:
                oldest_key = min(self._items, key=lambda item_key: self._items[item_key][0])
                self._items.pop(oldest_key, None)
            self._items[key] = (time.monotonic() + self.ttl_seconds, deepcopy(value))


_llm_cache = _TtlCache()


def cache_key(task: str, payload: Any) -> str:
    serialized = json.dumps(payload, sort_keys=True, default=str, ensure_ascii=False)
    return f"{task}:{sha256(serialized.encode('utf-8')).hexdigest()}"


def rag_answer_schema() -> dict[str, Any]:
    schema = RagAnswer.model_json_schema()
    return require_all_properties(schema)


def follow_up_schema() -> dict[str, Any]:
    schema = FollowUpQuestions.model_json_schema()
    return require_all_properties(schema)


def conversation_question_schema() -> dict[str, Any]:
    schema = ConversationQuestionPlan.model_json_schema()
    return require_all_properties(schema)


def response_diagram_schema() -> dict[str, Any]:
    schema = ResponseDiagram.model_json_schema()
    return require_all_properties(schema)


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


def build_context(results: list[dict[str, Any]]) -> str:
    blocks = []
    for index, result in enumerate(results, 1):
        pages = ", ".join(str(page) for page in result.get("page_numbers", [])) or "n/a"
        source_type = result.get("source_type") or "document"
        start_label = result.get("start_time_label") or result.get("metadata", {}).get("start_time_label")
        end_label = result.get("end_time_label") or result.get("metadata", {}).get("end_time_label")
        time_range = f"{start_label}-{end_label}" if start_label and end_label else "n/a"
        section = result.get("section") or "Unknown section"
        blocks.append(
            f"[{index}] source_type={source_type}; pages={pages}; time_range={time_range}; "
            f"type={result.get('content_type')}; section={section}\n"
            f"{result.get('content', '')}"
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, results: list[dict[str, Any]]) -> str:
    context = build_context(results)
    schema = json.dumps(rag_answer_schema(), indent=2)
    return prompt_text("dynamic", "rag_answer", "human", question=question, context=context, schema=schema)


def parse_structured_answer(text: str) -> RagAnswer:
    try:
        return RagAnswer.model_validate_json(text)
    except ValidationError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return RagAnswer.model_validate_json(text[start : end + 1])
        raise


def answer_question_structured(question: str, results: list[dict[str, Any]], model: str | None = None) -> RagAnswer:
    load_dotenv_file()
    cache_payload = {"provider": llm_provider(), "model": model, "question": question, "results": results}
    key = cache_key("rag_answer", cache_payload)
    cached = _llm_cache.get(key)
    if cached is not None:
        return RagAnswer.model_validate(cached)

    if llm_provider() == "databricks":
        model = model or databricks_chat_endpoint()
        answer = answer_question_structured_databricks(question, results, endpoint=model)
        _llm_cache.set(key, answer.model_dump())
        return answer

    model = model or env_value("OPENAI_ANSWER_MODEL") or DEFAULT_ANSWER_MODEL
    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
    user_prompt = build_user_prompt(question, results)
    system_prompt = prompt_text("static", "rag_answer", "system")
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "rag_answer",
                "schema": rag_answer_schema(),
                "strict": True,
            }
        },
    )
    answer = parse_structured_answer(response.output_text)
    _llm_cache.set(key, answer.model_dump())
    return answer


def answer_question_structured_databricks(question: str, results: list[dict[str, Any]], endpoint: str) -> RagAnswer:
    user_prompt = build_user_prompt(question, results)
    system_prompt = prompt_text("static", "rag_answer", "system")
    content = chat_completion(
        endpoint=endpoint,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.0,
        max_tokens=1200,
    )
    return parse_structured_answer(content)


def answer_question(question: str, results: list[dict[str, Any]], model: str | None = None) -> str:
    return answer_question_structured(question, results, model=model).answer


def follow_up_prompt(question: str, answer: str, results: list[dict[str, Any]]) -> str:
    context = build_context(results[:4])
    schema = json.dumps(follow_up_schema(), indent=2)
    return prompt_text("dynamic", "follow_up_questions", "human", question=question, answer=answer, context=context, schema=schema)


def parse_follow_up_questions(text: str) -> list[str]:
    try:
        result = FollowUpQuestions.model_validate_json(text)
    except ValidationError:
        start = text.find("{")
        end = text.rfind("}")
        if start < 0 or end <= start:
            raise
        result = FollowUpQuestions.model_validate_json(text[start : end + 1])
    deduped = []
    seen = set()
    for question in result.questions:
        cleaned = " ".join(str(question).strip().split())
        normalized = cleaned.lower()
        if not cleaned or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(cleaned)
    return deduped[:4]


def compact_history(history: list[dict[str, Any]], max_turns: int = 6) -> str:
    lines = []
    for item in history[-max_turns:]:
        role = str(item.get("role") or "").strip().lower()
        content = " ".join(str(item.get("content") or "").split())
        if role not in {"user", "assistant"} or not content:
            continue
        lines.append(f"{role}: {content[:900]}")
    return "\n".join(lines) or "No previous conversation."


def parse_conversation_question_plan(text: str) -> ConversationQuestionPlan:
    try:
        return ConversationQuestionPlan.model_validate_json(text)
    except ValidationError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return ConversationQuestionPlan.model_validate_json(text[start : end + 1])
        raise


def plan_conversation_question(question: str, history: list[dict[str, Any]], model: str | None = None) -> ConversationQuestionPlan:
    cleaned_question = " ".join(question.strip().split())
    if not history:
        return ConversationQuestionPlan(is_follow_up=False, standalone_question=cleaned_question, reason="No prior conversation.")

    load_dotenv_file()
    history_text = compact_history(history)
    key = cache_key("conversation_question", {"provider": llm_provider(), "model": model, "question": cleaned_question, "history": history_text})
    cached = _llm_cache.get(key)
    if cached is not None:
        return ConversationQuestionPlan.model_validate(cached)

    schema = json.dumps(conversation_question_schema(), indent=2)
    user_prompt = prompt_text("dynamic", "conversation_question", "human", history=history_text, question=cleaned_question, schema=schema)
    system_prompt = prompt_text("static", "conversation_question", "system")
    if llm_provider() == "databricks":
        content = chat_completion(
            endpoint=model or databricks_chat_endpoint(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        plan = parse_conversation_question_plan(content)
    else:
        client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
        response = client.responses.create(
            model=model or env_value("OPENAI_ANSWER_MODEL") or DEFAULT_ANSWER_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "conversation_question",
                    "schema": conversation_question_schema(),
                    "strict": True,
                }
            },
        )
        plan = parse_conversation_question_plan(response.output_text)

    if not plan.is_follow_up:
        plan.standalone_question = cleaned_question
    elif not plan.standalone_question.strip():
        plan.standalone_question = cleaned_question
    _llm_cache.set(key, plan.model_dump())
    return plan


def parse_response_diagram(text: str) -> ResponseDiagram:
    try:
        return ResponseDiagram.model_validate_json(text)
    except ValidationError:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return ResponseDiagram.model_validate_json(text[start : end + 1])
        raise


def generate_response_diagram(question: str, answer: str, results: list[dict[str, Any]], model: str | None = None) -> ResponseDiagram:
    if not answer.strip() or "unable to generate" in answer.lower():
        return ResponseDiagram(should_show=False, title="", diagram_type="", code="", reason="No supported answer.")

    load_dotenv_file()
    context = build_context(results[:6])
    key = cache_key("response_diagram", {"provider": llm_provider(), "model": model, "question": question, "answer": answer, "context": context})
    cached = _llm_cache.get(key)
    if cached is not None:
        return ResponseDiagram.model_validate(cached)

    schema = json.dumps(response_diagram_schema(), indent=2)
    user_prompt = prompt_text("dynamic", "response_diagram", "human", question=question, answer=answer, context=context, schema=schema)
    system_prompt = prompt_text("static", "response_diagram", "system")
    if llm_provider() == "databricks":
        content = chat_completion(
            endpoint=model or databricks_chat_endpoint(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.0,
            max_tokens=700,
        )
        diagram = parse_response_diagram(content)
    else:
        client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
        response = client.responses.create(
            model=model or env_value("OPENAI_ANSWER_MODEL") or DEFAULT_ANSWER_MODEL,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
                {"role": "user", "content": [{"type": "input_text", "text": user_prompt}]},
            ],
            text={
                "format": {
                    "type": "json_schema",
                    "name": "response_diagram",
                    "schema": response_diagram_schema(),
                    "strict": True,
                }
            },
        )
        diagram = parse_response_diagram(response.output_text)

    if diagram.diagram_type and diagram.diagram_type != "mermaid":
        diagram.should_show = False
        diagram.title = ""
        diagram.diagram_type = ""
        diagram.code = ""
    if diagram.should_show and not diagram.code.strip():
        diagram.should_show = False
    _llm_cache.set(key, diagram.model_dump())
    return diagram


def generate_follow_up_questions(question: str, answer: str, results: list[dict[str, Any]], model: str | None = None) -> list[str]:
    if not answer.strip() or "unable to generate" in answer.lower():
        return []
    load_dotenv_file()
    key = cache_key("follow_up_questions", {"provider": llm_provider(), "model": model, "question": question, "answer": answer, "results": results[:4]})
    cached = _llm_cache.get(key)
    if cached is not None:
        return list(cached)

    prompt = follow_up_prompt(question=question, answer=answer, results=results)
    system_prompt = prompt_text("static", "follow_up_questions", "system")
    if llm_provider() == "databricks":
        content = chat_completion(
            endpoint=model or databricks_chat_endpoint(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        questions = parse_follow_up_questions(content)
        _llm_cache.set(key, questions)
        return questions

    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
    response = client.responses.create(
        model=model or env_value("OPENAI_ANSWER_MODEL") or DEFAULT_ANSWER_MODEL,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": system_prompt}]},
            {"role": "user", "content": [{"type": "input_text", "text": prompt}]},
        ],
        text={
            "format": {
                "type": "json_schema",
                "name": "follow_up_questions",
                "schema": follow_up_schema(),
                "strict": True,
            }
        },
    )
    questions = parse_follow_up_questions(response.output_text)
    _llm_cache.set(key, questions)
    return questions
