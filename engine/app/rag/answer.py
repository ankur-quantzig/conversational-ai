from __future__ import annotations

import json
from typing import Any

from openai import OpenAI
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from app.clients.databricks_model_serving import chat_completion
from app.clients.document_intelligence import env_value, load_dotenv_file
from app.config import databricks_chat_endpoint, llm_provider


DEFAULT_ANSWER_MODEL = "gpt-4.1-mini"

SYSTEM_PROMPT = """
You answer questions about the user's selected documents.
Use only the provided context as evidence.
If the context does not contain enough evidence, set `heading` to an empty string and set `answer` exactly to: I am unable to generate the response at the moment. Please contact Admin.
Prefer concise, insight-style answers.
Use Markdown bullet points for the answer. Keep bullets precise, accurate, and directly aligned to the question.
Every factual claim must be supported by the provided context.
Never mention chunks, indexed documents, retrieval, RAG, fallback, pipelines, embeddings, context blocks, or implementation details.
Return valid JSON matching the requested Pydantic schema.
""".strip()


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


def rag_answer_schema() -> dict[str, Any]:
    schema = RagAnswer.model_json_schema()
    return require_all_properties(schema)


def follow_up_schema() -> dict[str, Any]:
    schema = FollowUpQuestions.model_json_schema()
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
    return f"""
Question:
{question}

Retrieved context:
{context}

Instructions for this question:
- Use the retrieved context blocks above as the only source of truth.
- Cite the 1-based context block number in `source_index`.
- Copy the relevant PDF page numbers from each cited context block into `pages`.
- For video sources, split the context block `time_range` into `start_time_label` and `end_time_label`.
- For document/PDF sources, set `start_time_label` and `end_time_label` to empty strings.
- If the retrieved context answers the question, do not say information is missing.
- Set `missing_information` to an empty string when the answer is supported.
- Set `heading` to a short professional response title. Use title case when natural. Do not include punctuation at the end.
- Set `confidence_score` from 0 to 1 based on how completely and directly the retrieved evidence answers the question.
- Use `confidence_score` >= 0.8 only when the evidence clearly answers the question.
- Use `confidence_score` < 0.8 when the evidence is weak, partial, off-topic, or mostly inferential.
- If the provided context does not contain enough evidence, set `confidence_score` below 0.8, set `heading` to an empty string, and set `answer` exactly to: I am unable to generate the response at the moment. Please contact Admin.
- Format `answer` as the final user-facing answer only.
- Format supported answers as 3 to 6 concise Markdown bullets.
- Start every answer line with "- ". Do not write a long paragraph before or after the bullets.
- Make the bullets read like business insights: specific, non-repetitive, and directly useful.
- Do not include citation markers, page labels, source numbers, source labels, or internal process notes inside `answer`.
- Do not use words such as chunks, indexed documents, retrieval, RAG, fallback, pipeline, embeddings, or context blocks in `answer`.
- Put source evidence only in the `citations` list.

Return JSON with this schema:
{schema}
""".strip()


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
    if llm_provider() == "databricks":
        model = model or databricks_chat_endpoint()
        return answer_question_structured_databricks(question, results, endpoint=model)

    model = model or env_value("OPENAI_ANSWER_MODEL") or DEFAULT_ANSWER_MODEL
    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
    user_prompt = build_user_prompt(question, results)
    response = client.responses.create(
        model=model,
        input=[
            {"role": "system", "content": [{"type": "input_text", "text": SYSTEM_PROMPT}]},
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
    return parse_structured_answer(response.output_text)


def answer_question_structured_databricks(question: str, results: list[dict[str, Any]], endpoint: str) -> RagAnswer:
    user_prompt = build_user_prompt(question, results)
    content = chat_completion(
        endpoint=endpoint,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
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
    return f"""
Original question:
{question}

Answer:
{answer}

Supporting source context:
{context}

Create up to 4 useful follow-up questions that the user can click next.
Rules:
- Questions must be grounded in the answer and sources.
- Make them specific, natural, and useful for continuing the same investigation.
- Do not suggest generic questions like "What are the main recommendations?" unless the answer is actually about recommendations.
- Do not repeat the original question.
- Return only JSON matching this schema:
{schema}
""".strip()


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


def generate_follow_up_questions(question: str, answer: str, results: list[dict[str, Any]], model: str | None = None) -> list[str]:
    if not answer.strip() or "unable to generate" in answer.lower():
        return []
    load_dotenv_file()
    prompt = follow_up_prompt(question=question, answer=answer, results=results)
    if llm_provider() == "databricks":
        content = chat_completion(
            endpoint=model or databricks_chat_endpoint(),
            messages=[
                {"role": "system", "content": "You create concise, source-grounded follow-up questions. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.2,
            max_tokens=500,
        )
        return parse_follow_up_questions(content)

    client = OpenAI(api_key=env_value("OPENAI_API_KEY", "OPANAI_API_KEY"))
    response = client.responses.create(
        model=model or env_value("OPENAI_ANSWER_MODEL") or DEFAULT_ANSWER_MODEL,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": "follow_up_questions",
                "schema": follow_up_schema(),
                "strict": True,
            }
        },
    )
    return parse_follow_up_questions(response.output_text)
