from __future__ import annotations

import json
import queue
import subprocess
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from psycopg.types.json import Jsonb
from pydantic import BaseModel, ConfigDict, Field
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.api.retrieval import answer_question, list_documents, load_chunks
from app.config import (
    app_env,
    cors_origins,
    databricks_chat_endpoint,
    databricks_embedding_endpoint,
    databricks_host,
    databricks_token,
    is_databricks_env,
    is_local_env,
    llm_provider,
    retrieval_top_k,
)
from app.clients.lancedb_store import DEFAULT_TABLE_NAME, open_table
from app.db.postgres import get_connection, init_db, wait_for_database
from app.rag.answer import (
    DEFAULT_CLARIFICATION_QUESTION,
    QuestionPreparation,
    generate_follow_up_questions,
    plan_conversation_question,
    prepare_retrieval_question,
)
from app.security.audit import log_audit_event
from app.security.auth import UserContext, current_user, ensure_document_access, require_role
from app.security.guardrails import QuerySecurityResult, classify_query
from app.security.quota import enforce_question_quota, question_limit_for, questions_used
from app.security.rate_limit import enforce_rate_limit
from app.utils.files import project_root


UNABLE_TO_GENERATE_MESSAGE = "I am unable to generate the response at the moment. Please contact Admin."


class ChatRequest(BaseModel):
    question: str = Field(min_length=1)
    session_id: str | None = None
    top_k: int | None = Field(default=None, ge=1, le=10)
    doc_id: str | None = None
    source_type: str | None = Field(default=None, pattern="^(document|video)?$")


class ChatResponse(BaseModel):
    session_id: str
    message_id: str | None = None
    request_id: str = ""
    heading: str = ""
    answer: str
    sources: list[dict[str, Any]]
    mode: str
    confidence: str | None = None
    confidence_score: float | None = None
    citations: list[dict[str, Any]] = Field(default_factory=list)
    missing_information: str = ""
    follow_up_questions: list[str] = Field(default_factory=list)
    diagram: dict[str, Any] = Field(default_factory=dict)
    question_analysis: dict[str, Any] = Field(default_factory=dict)
    security: dict[str, Any] = Field(default_factory=dict)
    elapsed_ms: int | None = None


class ChatMessage(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: str
    session_id: str
    role: str
    content: str
    metadata: dict[str, Any]
    created_at: datetime


class FeedbackRequest(BaseModel):
    rating: str = Field(pattern="^(up|down)$")
    comment: str = ""


class RetryRequest(BaseModel):
    top_k: int | None = Field(default=None, ge=1, le=10)


class SessionUpdateRequest(BaseModel):
    title: str = Field(min_length=1, max_length=200)


class ShareResponse(BaseModel):
    share_id: str
    url: str


class ExportResponse(BaseModel):
    session_id: str
    title: str
    messages: list[dict[str, Any]]


app = FastAPI(title="Conversational AI Chat API")

configured_cors_origins = cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=configured_cors_origins,
    allow_credentials=True if configured_cors_origins else False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = request_id
    try:
        response = await call_next(request)
    except Exception as exc:
        response = JSONResponse(status_code=500, content={"detail": "Internal server error", "request_id": request_id})
        response.headers["x-error-type"] = type(exc).__name__
    response.headers["x-request-id"] = request_id
    return response


def elapsed_ms_since(started_at: float) -> int:
    return max(0, int((time.perf_counter() - started_at) * 1000))


def sse_event(event: str, data: dict[str, Any]) -> str:
    payload = json.dumps(data, default=str, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


def user_facing_error_message(exc: Exception) -> str:
    return UNABLE_TO_GENERATE_MESSAGE


def safe_follow_up_questions(question: str, answer: str, sources: list[dict[str, Any]]) -> list[str]:
    try:
        return generate_follow_up_questions(question=question, answer=answer, results=sources)
    except Exception:
        return []


def safe_response_diagram(question: str, answer: str, sources: list[dict[str, Any]]) -> dict[str, Any]:
    return {}


def conversation_history(session_id: str | None, user: UserContext, limit: int = 8) -> list[dict[str, Any]]:
    if not session_id:
        return []
    ensure_session_access(session_id, user)
    with get_connection() as conn:
        rows = conn.execute(
            """
            select role, content
            from chat_messages
            where session_id = %s
            order by created_at desc
            limit %s
            """,
            (session_id, limit),
        ).fetchall()
    return list(reversed(rows))


def analyze_question_for_turn(question: str, history: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    try:
        plan = plan_conversation_question(question=question, history=history)
        payload = plan.model_dump()
        payload["original_question"] = question
        payload["effective_question"] = plan.standalone_question
        return plan.standalone_question, payload
    except Exception as exc:
        return question, {
            "is_follow_up": False,
            "standalone_question": question,
            "original_question": question,
            "effective_question": question,
            "reason": f"Question classifier unavailable: {type(exc).__name__}",
        }


def safe_prepare_retrieval_question(question: str) -> tuple[str, dict[str, Any]]:
    cleaned_question = " ".join(question.strip().split())
    try:
        plan = prepare_retrieval_question(cleaned_question)
    except Exception as exc:
        plan = QuestionPreparation(
            status="ready",
            rephrased_question=cleaned_question,
            clarification_question="",
            issue="none",
            confidence_score=0.0,
            reason=f"Question preparation unavailable: {type(exc).__name__}",
        )
    payload = plan.model_dump()
    return plan.rephrased_question, payload


def prepare_question_for_answering(question: str, history: list[dict[str, Any]]) -> tuple[str, dict[str, Any]]:
    standalone_question, question_analysis = analyze_question_for_turn(question, history)
    retrieval_question, preparation = safe_prepare_retrieval_question(standalone_question)
    question_analysis["question_preparation"] = preparation
    question_analysis["standalone_question"] = standalone_question
    question_analysis["effective_question"] = retrieval_question
    return retrieval_question, question_analysis


def question_needs_clarification(question_analysis: dict[str, Any]) -> bool:
    preparation = question_analysis.get("question_preparation") or {}
    return preparation.get("status") == "needs_clarification"


def clarification_answer(question_analysis: dict[str, Any]) -> str:
    preparation = question_analysis.get("question_preparation") or {}
    answer = str(preparation.get("clarification_question") or "").strip()
    return answer or DEFAULT_CLARIFICATION_QUESTION


def build_clarification_chat_response(
    *,
    session_id: str,
    title: str,
    user: UserContext,
    chat_request: ChatRequest,
    question: str,
    effective_top_k: int,
    question_analysis: dict[str, Any],
    security: QuerySecurityResult,
    quota: dict[str, int | None],
    started_at: float,
    request_id: str,
) -> ChatResponse:
    answer = clarification_answer(question_analysis)
    elapsed_ms = elapsed_ms_since(started_at)
    assistant_message_id = persist_chat_turn(
        session_id=session_id,
        title=title,
        user=user,
        question=question,
        answer=answer,
        user_metadata={"doc_id": chat_request.doc_id, "source_type": chat_request.source_type, "top_k": effective_top_k},
        assistant_metadata={
            "sources": [],
            "mode": "needs_clarification",
            "confidence": "needs_clarification",
            "confidence_score": 0.0,
            "citations": [],
            "follow_up_questions": [],
            "diagram": {},
            "question_analysis": question_analysis,
            "heading": "",
            "missing_information": "",
            "security": {**security.model_dump(), "quota": quota},
            "elapsed_ms": elapsed_ms,
        },
        existing_session=bool(chat_request.session_id),
    )
    log_audit_event(
        "chat_needs_clarification",
        user=user,
        request_id=request_id,
        metadata={"session_id": session_id, "message_id": assistant_message_id, "question_analysis": question_analysis},
    )
    return ChatResponse(
        session_id=session_id,
        message_id=assistant_message_id,
        request_id=request_id,
        heading="",
        answer=answer,
        sources=[],
        mode="needs_clarification",
        confidence="needs_clarification",
        confidence_score=0.0,
        citations=[],
        follow_up_questions=[],
        diagram={},
        question_analysis=question_analysis,
        security={**security.model_dump(), "quota": quota},
        elapsed_ms=elapsed_ms,
    )


def answer_deltas(answer: str, chunk_size: int = 8):
    words = answer.split(" ")
    for start in range(0, len(words), chunk_size):
        piece = " ".join(words[start : start + chunk_size])
        if start + chunk_size < len(words):
            piece += " "
        yield piece


def run_answer_pipeline_with_progress(
    question: str,
    top_k: int,
    doc_id: str | None,
    source_type: str | None,
    request_id: str,
):
    progress_events: queue.Queue[tuple[str, dict[str, Any]]] = queue.Queue()
    result: dict[str, Any] = {}

    def progress(stage: str, message: str, metadata: dict[str, Any] | None = None) -> None:
        progress_events.put(
            (
                "progress",
                {
                    "stage": stage,
                    "message": message,
                    "metadata": metadata or {},
                    "request_id": request_id,
                },
            )
        )

    def target() -> None:
        try:
            result["response"] = answer_question(
                question=question,
                top_k=top_k,
                doc_id=doc_id,
                source_type=source_type,
                progress=progress,
            )
        except Exception as exc:
            result["error"] = exc
        finally:
            progress_events.put(("pipeline_done", {"request_id": request_id}))

    worker = threading.Thread(target=target, name=f"answer-pipeline-{request_id}", daemon=True)
    worker.start()

    while True:
        try:
            event, data = progress_events.get(timeout=0.25)
        except queue.Empty:
            if worker.is_alive():
                yield "heartbeat", {"request_id": request_id}
                continue
            break

        if event == "pipeline_done":
            break
        yield event, data

    worker.join()
    if result.get("error"):
        raise result["error"]
    yield "pipeline_result", result["response"]


@app.on_event("startup")
def startup() -> None:
    if not is_local_env() and not is_databricks_env() and not cors_origins():
        raise RuntimeError("CORS_ORIGINS must be configured outside local development")
    wait_for_database()
    init_db()
    load_chunks()


@app.get("/health")
def health() -> dict[str, Any]:
    with get_connection() as conn:
        conn.execute("select 1")
    return {"ok": True, "env": app_env(), "chunks": len(load_chunks())}


@app.get("/documents")
def documents(user: UserContext = Depends(current_user)) -> list[dict[str, Any]]:
    docs = list_documents()
    if user.is_admin or "*" in user.document_ids:
        return docs
    return [doc for doc in docs if doc.get("id") in user.document_ids]


@app.get("/me")
def me(user: UserContext = Depends(current_user)) -> dict[str, Any]:
    limit = question_limit_for(user)
    used = questions_used(user)
    return {
        "user_id": user.user_id,
        "email": user.email,
        "name": user.name,
        "roles": user.roles,
        "question_quota": {
            "used": used,
            "limit": limit,
            "remaining": None if limit is None else max(0, limit - used),
        },
    }


@app.get("/diagnostics/runtime")
def runtime_diagnostics(user: UserContext = Depends(current_user)) -> dict[str, Any]:
    require_role(user, {"admin", "analyst"})
    vector_rows: int | None = None
    vector_error = ""
    try:
        vector_rows = open_table(DEFAULT_TABLE_NAME).count_rows()
    except Exception as exc:
        vector_error = f"{type(exc).__name__}: {exc}"

    embedding_summaries = []
    for path in sorted((project_root() / "output" / "embeddings").glob("*-embedding-summary.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        embedding_summaries.append(
            {
                "file": path.name,
                "model": payload.get("embedding_model"),
                "dimensions": payload.get("embedding_dimensions"),
                "chunk_count": payload.get("chunk_count"),
            }
        )

    return {
        "env": app_env(),
        "llm_provider": llm_provider(),
        "chunks": len(load_chunks()),
        "vector_db": {"table": DEFAULT_TABLE_NAME, "rows": vector_rows, "error": vector_error},
        "databricks": {
            "host_configured": bool(databricks_host()),
            "token_configured": bool(databricks_token()),
            "chat_endpoint": databricks_chat_endpoint(),
            "embedding_endpoint": databricks_embedding_endpoint(),
        },
        "embedding_summaries": embedding_summaries,
    }


@app.get("/sessions")
def sessions(user: UserContext = Depends(current_user)) -> list[dict[str, Any]]:
    require_role(user, {"admin", "analyst", "viewer"})
    with get_connection() as conn:
        rows = conn.execute(
            """
            select
              s.id::text,
              s.title,
              s.created_at,
              s.updated_at,
              count(m.id)::int as message_count
            from chat_sessions s
            left join chat_messages m on m.session_id = s.id
            where s.user_id = %s and s.tenant_id = %s
            group by s.id
            order by s.updated_at desc
            limit 50
            """,
            (user.user_id, user.tenant_id),
        ).fetchall()
    return rows


@app.get("/sessions/{session_id}/messages")
def session_messages(session_id: str, user: UserContext = Depends(current_user)) -> list[dict[str, Any]]:
    require_role(user, {"admin", "analyst", "viewer"})
    ensure_session_access(session_id, user)
    with get_connection() as conn:
        rows = conn.execute(
            """
            select id::text, session_id::text, role, content, metadata, created_at
            from chat_messages
            where session_id = %s
            order by created_at asc
            """,
            (session_id,),
        ).fetchall()
    if not rows:
        return []
    return rows


@app.post("/sessions")
def create_session(user: UserContext = Depends(current_user)) -> dict[str, Any]:
    require_role(user, {"admin", "analyst"})
    session_id = str(uuid.uuid4())
    with get_connection() as conn:
        row = conn.execute(
            """
            insert into chat_sessions (id, title, user_id, tenant_id)
            values (%s, %s, %s, %s)
            returning id::text, title, created_at, updated_at
            """,
            (session_id, "New chat", user.user_id, user.tenant_id),
        ).fetchone()
    return row


@app.patch("/sessions/{session_id}")
def update_session(session_id: str, payload: SessionUpdateRequest, user: UserContext = Depends(current_user)) -> dict[str, Any]:
    require_role(user, {"admin", "analyst"})
    ensure_session_access(session_id, user)
    title = payload.title.strip()
    if not title:
        raise HTTPException(status_code=400, detail="Title cannot be empty")
    with get_connection() as conn:
        row = conn.execute(
            "update chat_sessions set title = %s where id = %s returning id::text, title",
            (title, session_id),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    return row


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str, user: UserContext = Depends(current_user)) -> dict[str, bool]:
    require_role(user, {"admin", "analyst"})
    ensure_session_access(session_id, user)
    with get_connection() as conn:
        result = conn.execute("delete from chat_sessions where id = %s", (session_id,))
    if result.rowcount == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"ok": True}


@app.post("/chat", response_model=ChatResponse)
def chat(chat_request: ChatRequest, request: Request, user: UserContext = Depends(current_user)) -> ChatResponse:
    started_at = time.perf_counter()
    require_role(user, {"admin", "analyst"})
    enforce_rate_limit(f"{user.tenant_id}:{user.user_id}:chat")
    try:
        quota = enforce_question_quota(user)
    except HTTPException as exc:
        log_audit_event(
            "chat_quota_exceeded",
            user=user,
            request_id=request.state.request_id,
            metadata={"limit": question_limit_for(user), "used": questions_used(user)},
        )
        raise exc
    question = chat_request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    selected_doc_id = None if chat_request.doc_id in (None, "", "all") else chat_request.doc_id
    ensure_document_access(user, selected_doc_id)

    session_id = chat_request.session_id or str(uuid.uuid4())
    title = question[:70]
    effective_top_k = chat_request.top_k or retrieval_top_k()
    request_id = request.state.request_id
    security = classify_query(question)
    log_audit_event(
        "query_security_check",
        user=user,
        request_id=request_id,
        metadata={"question": question, "result": security.model_dump(), "doc_id": selected_doc_id},
    )
    if not security.is_allowed:
        elapsed_ms = elapsed_ms_since(started_at)
        answer = UNABLE_TO_GENERATE_MESSAGE
        assistant_message_id = persist_chat_turn(
            session_id=session_id,
            title=title,
            user=user,
            question=question,
            answer=answer,
            user_metadata={"doc_id": chat_request.doc_id, "source_type": chat_request.source_type, "top_k": effective_top_k},
            assistant_metadata={
                "sources": [],
                "mode": "blocked_guardrail",
                "citations": [],
                "heading": "",
                "security": {**security.model_dump(), "quota": quota},
                "elapsed_ms": elapsed_ms,
            },
            existing_session=bool(chat_request.session_id),
        )
        log_audit_event("chat_blocked", user=user, request_id=request_id, metadata={"session_id": session_id, "reason": security.reason})
        return ChatResponse(
            session_id=session_id,
            message_id=assistant_message_id,
            request_id=request_id,
            heading="",
            answer=answer,
            sources=[],
            mode="blocked_guardrail",
            confidence="blocked",
            citations=[],
            security={**security.model_dump(), "quota": quota},
            elapsed_ms=elapsed_ms,
        )

    history = conversation_history(chat_request.session_id, user)
    effective_question, question_analysis = prepare_question_for_answering(question, history)
    if question_needs_clarification(question_analysis):
        return build_clarification_chat_response(
            session_id=session_id,
            title=title,
            user=user,
            chat_request=chat_request,
            question=question,
            effective_top_k=effective_top_k,
            question_analysis=question_analysis,
            security=security,
            quota=quota,
            started_at=started_at,
            request_id=request_id,
        )

    rag_response = answer_question(
        question=effective_question,
        top_k=effective_top_k,
        doc_id=selected_doc_id,
        source_type=None if chat_request.source_type in (None, "", "all") else chat_request.source_type,
    )
    answer = rag_response["answer"]
    sources = rag_response["sources"]
    heading = rag_response.get("heading", "")
    follow_up_questions = safe_follow_up_questions(question=effective_question, answer=answer, sources=sources)
    diagram = {}
    elapsed_ms = elapsed_ms_since(started_at)

    assistant_message_id = persist_chat_turn(
        session_id=session_id,
        title=title,
        user=user,
        question=question,
        answer=answer,
        user_metadata={"doc_id": chat_request.doc_id, "source_type": chat_request.source_type, "top_k": effective_top_k},
        assistant_metadata={
            "sources": sources,
            "mode": rag_response.get("mode"),
            "confidence": rag_response.get("confidence"),
            "confidence_score": rag_response.get("confidence_score"),
            "citations": rag_response.get("citations", []),
            "follow_up_questions": follow_up_questions,
            "diagram": diagram,
            "question_analysis": question_analysis,
            "heading": heading,
            "missing_information": rag_response.get("missing_information", ""),
            "security": {**security.model_dump(), "quota": quota},
            "elapsed_ms": elapsed_ms,
        },
        existing_session=bool(chat_request.session_id),
    )
    log_audit_event(
        "chat_completed",
        user=user,
        request_id=request_id,
        metadata={"session_id": session_id, "message_id": assistant_message_id, "mode": rag_response.get("mode")},
    )

    return ChatResponse(
        session_id=session_id,
        message_id=assistant_message_id,
        request_id=request_id,
        heading=heading,
        answer=answer,
        sources=sources,
        mode=rag_response.get("mode", "unknown"),
        confidence=rag_response.get("confidence"),
        confidence_score=rag_response.get("confidence_score"),
        citations=rag_response.get("citations", []),
        missing_information=rag_response.get("missing_information", ""),
        follow_up_questions=follow_up_questions,
        diagram=diagram,
        question_analysis=question_analysis,
        security={**security.model_dump(), "quota": quota},
        elapsed_ms=elapsed_ms,
    )


@app.post("/chat/stream")
def chat_stream(chat_request: ChatRequest, request: Request, user: UserContext = Depends(current_user)) -> StreamingResponse:
    def events():
        started_at = time.perf_counter()
        request_id = request.state.request_id
        yield sse_event("status", {"stage": "agent_ready", "message": "Agent is ready", "request_id": request_id})
        try:
            yield sse_event("progress", {"stage": "checking_access", "message": "Checking access and quota", "metadata": {}, "request_id": request_id})
            require_role(user, {"admin", "analyst"})
            enforce_rate_limit(f"{user.tenant_id}:{user.user_id}:chat")
            try:
                quota = enforce_question_quota(user)
            except HTTPException as exc:
                log_audit_event(
                    "chat_quota_exceeded",
                    user=user,
                    request_id=request_id,
                    metadata={"limit": question_limit_for(user), "used": questions_used(user)},
                )
                raise exc

            question = chat_request.question.strip()
            if not question:
                raise HTTPException(status_code=400, detail="Question cannot be empty")

            selected_doc_id = None if chat_request.doc_id in (None, "", "all") else chat_request.doc_id
            ensure_document_access(user, selected_doc_id)

            session_id = chat_request.session_id or str(uuid.uuid4())
            title = question[:70]
            effective_top_k = chat_request.top_k or retrieval_top_k()

            yield sse_event("progress", {"stage": "guardrail_check", "message": "Checking question safety", "metadata": {}, "request_id": request_id})
            security = classify_query(question)
            log_audit_event(
                "query_security_check",
                user=user,
                request_id=request_id,
                metadata={"question": question, "result": security.model_dump(), "doc_id": selected_doc_id},
            )
            if not security.is_allowed:
                elapsed_ms = elapsed_ms_since(started_at)
                answer = UNABLE_TO_GENERATE_MESSAGE
                assistant_message_id = persist_chat_turn(
                    session_id=session_id,
                    title=title,
                    user=user,
                    question=question,
                    answer=answer,
                    user_metadata={"doc_id": chat_request.doc_id, "source_type": chat_request.source_type, "top_k": effective_top_k},
                    assistant_metadata={
                        "sources": [],
                        "mode": "blocked_guardrail",
                        "citations": [],
                        "heading": "",
                        "security": {**security.model_dump(), "quota": quota},
                        "elapsed_ms": elapsed_ms,
                    },
                    existing_session=bool(chat_request.session_id),
                )
                log_audit_event("chat_blocked", user=user, request_id=request_id, metadata={"session_id": session_id, "reason": security.reason})
                response = ChatResponse(
                    session_id=session_id,
                    message_id=assistant_message_id,
                    request_id=request_id,
                    heading="",
                    answer=answer,
                    sources=[],
                    mode="blocked_guardrail",
                    confidence="blocked",
                    citations=[],
                    security={**security.model_dump(), "quota": quota},
                    elapsed_ms=elapsed_ms,
                )
                yield sse_event("status", {"stage": "final_response", "message": "Final response", "request_id": request_id})
                yield sse_event("final", response.model_dump())
                return

            yield sse_event("progress", {"stage": "question_analysis", "message": "Checking conversation context", "metadata": {}, "request_id": request_id})
            history = conversation_history(chat_request.session_id, user)
            standalone_question, question_analysis = analyze_question_for_turn(question, history)
            yield sse_event("progress", {"stage": "question_rephrasing", "message": "Rephrasing question for retrieval", "metadata": {}, "request_id": request_id})
            effective_question, preparation = safe_prepare_retrieval_question(standalone_question)
            question_analysis["question_preparation"] = preparation
            question_analysis["standalone_question"] = standalone_question
            question_analysis["effective_question"] = effective_question
            yield sse_event(
                "progress",
                {
                    "stage": "question_ready",
                    "message": "Prepared retrieval question",
                    "metadata": {
                        "is_follow_up": question_analysis.get("is_follow_up"),
                        "effective_question": effective_question,
                        "question_preparation": question_analysis.get("question_preparation", {}),
                    },
                    "request_id": request_id,
                },
            )
            if question_needs_clarification(question_analysis):
                response = build_clarification_chat_response(
                    session_id=session_id,
                    title=title,
                    user=user,
                    chat_request=chat_request,
                    question=question,
                    effective_top_k=effective_top_k,
                    question_analysis=question_analysis,
                    security=security,
                    quota=quota,
                    started_at=started_at,
                    request_id=request_id,
                )
                yield sse_event("status", {"stage": "writing", "message": "", "request_id": request_id})
                for delta in answer_deltas(response.answer):
                    yield sse_event("answer_delta", {"delta": delta, "request_id": request_id})
                    time.sleep(0.025)
                yield sse_event("progress", {"stage": "complete", "message": "Response ready", "metadata": {}, "request_id": request_id})
                yield sse_event("status", {"stage": "final_response", "message": "Final response", "request_id": request_id})
                yield sse_event("final", response.model_dump())
                return

            source_type = None if chat_request.source_type in (None, "", "all") else chat_request.source_type
            rag_response = None
            for event, data in run_answer_pipeline_with_progress(
                question=effective_question,
                top_k=effective_top_k,
                doc_id=selected_doc_id,
                source_type=source_type,
                request_id=request_id,
            ):
                if event == "pipeline_result":
                    rag_response = data
                    continue
                yield sse_event(event, data)
            if rag_response is None:
                raise RuntimeError("Answer pipeline finished without a response.")

            answer = rag_response["answer"]
            sources = rag_response["sources"]
            heading = rag_response.get("heading", "")
            yield sse_event("status", {"stage": "writing", "message": "", "request_id": request_id})
            for delta in answer_deltas(answer):
                yield sse_event("answer_delta", {"delta": delta, "request_id": request_id})
                time.sleep(0.025)
            yield sse_event("progress", {"stage": "creating_followups", "message": "Creating follow-up questions", "metadata": {}, "request_id": request_id})
            follow_up_questions = safe_follow_up_questions(question=effective_question, answer=answer, sources=sources)
            diagram = {}
            elapsed_ms = elapsed_ms_since(started_at)

            yield sse_event("progress", {"stage": "saving_conversation", "message": "Saving conversation", "metadata": {}, "request_id": request_id})
            assistant_message_id = persist_chat_turn(
                session_id=session_id,
                title=title,
                user=user,
                question=question,
                answer=answer,
                user_metadata={"doc_id": chat_request.doc_id, "source_type": chat_request.source_type, "top_k": effective_top_k},
                assistant_metadata={
                    "sources": sources,
                    "mode": rag_response.get("mode"),
                    "confidence": rag_response.get("confidence"),
                    "confidence_score": rag_response.get("confidence_score"),
                    "citations": rag_response.get("citations", []),
                    "follow_up_questions": follow_up_questions,
                    "diagram": diagram,
                    "question_analysis": question_analysis,
                    "heading": heading,
                    "missing_information": rag_response.get("missing_information", ""),
                    "security": {**security.model_dump(), "quota": quota},
                    "elapsed_ms": elapsed_ms,
                },
                existing_session=bool(chat_request.session_id),
            )
            log_audit_event(
                "chat_completed",
                user=user,
                request_id=request_id,
                metadata={"session_id": session_id, "message_id": assistant_message_id, "mode": rag_response.get("mode")},
            )
            yield sse_event("progress", {"stage": "complete", "message": "Response ready", "metadata": {}, "request_id": request_id})
            response = ChatResponse(
                session_id=session_id,
                message_id=assistant_message_id,
                request_id=request_id,
                heading=heading,
                answer=answer,
                sources=sources,
                mode=rag_response.get("mode", "unknown"),
                confidence=rag_response.get("confidence"),
                confidence_score=rag_response.get("confidence_score"),
                citations=rag_response.get("citations", []),
                missing_information=rag_response.get("missing_information", ""),
                follow_up_questions=follow_up_questions,
                diagram=diagram,
                question_analysis=question_analysis,
                security={**security.model_dump(), "quota": quota},
                elapsed_ms=elapsed_ms,
            )
            yield sse_event("status", {"stage": "final_response", "message": "Final response", "request_id": request_id})
            yield sse_event("final", response.model_dump())
        except Exception as exc:
            try:
                log_audit_event(
                    "chat_stream_error",
                    user=user,
                    request_id=request_id,
                    metadata={"error_type": type(exc).__name__, "message": user_facing_error_message(exc)},
                )
            except Exception:
                pass
            yield sse_event(
                "error",
                {
                    "request_id": request_id,
                    "message": user_facing_error_message(exc),
                    "elapsed_ms": elapsed_ms_since(started_at),
                    "status_code": exc.status_code if isinstance(exc, HTTPException) else 500,
                },
            )

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def persist_chat_turn(
    session_id: str,
    title: str,
    user: UserContext,
    question: str,
    answer: str,
    user_metadata: dict[str, Any],
    assistant_metadata: dict[str, Any],
    existing_session: bool,
) -> str:
    assistant_message_id = str(uuid.uuid4())
    with get_connection() as conn:
        if existing_session:
            existing = conn.execute("select id from chat_sessions where id = %s", (session_id,)).fetchone()
            if not existing:
                raise HTTPException(status_code=404, detail="Session not found")
            ensure_session_access(session_id, user)
        else:
            conn.execute(
                "insert into chat_sessions (id, title, user_id, tenant_id) values (%s, %s, %s, %s)",
                (session_id, title, user.user_id, user.tenant_id),
            )
        conn.execute(
            """
            insert into chat_messages (id, session_id, role, content, metadata)
            values (%s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                session_id,
                "user",
                question,
                Jsonb(user_metadata),
            ),
        )
        conn.execute(
            """
            insert into chat_messages (id, session_id, role, content, metadata)
            values (%s, %s, %s, %s, %s)
            """,
            (
                assistant_message_id,
                session_id,
                "assistant",
                answer,
                Jsonb(assistant_metadata),
            ),
        )
        conn.execute("update chat_sessions set updated_at = now() where id = %s", (session_id,))
    return assistant_message_id


def ensure_session_access(session_id: str, user: UserContext) -> None:
    with get_connection() as conn:
        row = conn.execute("select user_id, tenant_id from chat_sessions where id = %s", (session_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Session not found")
    if user.is_admin:
        return
    if row.get("user_id") != user.user_id or row.get("tenant_id") != user.tenant_id:
        raise HTTPException(status_code=403, detail="Session access denied")


@app.post("/messages/{message_id}/feedback")
def save_feedback(message_id: str, feedback: FeedbackRequest, request: Request, user: UserContext = Depends(current_user)) -> dict[str, Any]:
    require_role(user, {"admin", "analyst", "viewer"})
    with get_connection() as conn:
        message = conn.execute("select session_id::text from chat_messages where id = %s", (message_id,)).fetchone()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        ensure_session_access(message["session_id"], user)
        feedback_id = str(uuid.uuid4())
        conn.execute(
            """
            insert into message_feedback (id, message_id, session_id, user_id, rating, comment)
            values (%s, %s, %s, %s, %s, %s)
            """,
            (feedback_id, message_id, message["session_id"], user.user_id, feedback.rating, feedback.comment),
        )
    log_audit_event("message_feedback", user=user, request_id=request.state.request_id, metadata={"message_id": message_id, "rating": feedback.rating})
    return {"ok": True, "feedback_id": feedback_id}


@app.post("/messages/{message_id}/share", response_model=ShareResponse)
def share_message(message_id: str, request: Request, user: UserContext = Depends(current_user)) -> ShareResponse:
    require_role(user, {"admin", "analyst", "viewer"})
    with get_connection() as conn:
        message = conn.execute(
            "select id::text, session_id::text, role, content, metadata, created_at from chat_messages where id = %s",
            (message_id,),
        ).fetchone()
        if not message:
            raise HTTPException(status_code=404, detail="Message not found")
        ensure_session_access(message["session_id"], user)
        share_id = str(uuid.uuid4())
        title = message["content"][:70]
        conn.execute(
            """
            insert into shared_messages (id, message_id, session_id, user_id, title, payload)
            values (%s, %s, %s, %s, %s, %s)
            """,
            (share_id, message_id, message["session_id"], user.user_id, title, Jsonb(message)),
        )
    log_audit_event("message_shared", user=user, request_id=request.state.request_id, metadata={"message_id": message_id, "share_id": share_id})
    return ShareResponse(share_id=share_id, url=f"/shares/{share_id}")


@app.get("/shares/{share_id}")
def get_share(share_id: str) -> dict[str, Any]:
    with get_connection() as conn:
        row = conn.execute("select id::text, title, payload, created_at from shared_messages where id = %s", (share_id,)).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Share not found")
    return row


@app.get("/sessions/{session_id}/export", response_model=ExportResponse)
def export_session(session_id: str, user: UserContext = Depends(current_user)) -> ExportResponse:
    require_role(user, {"admin", "analyst", "viewer"})
    ensure_session_access(session_id, user)
    with get_connection() as conn:
        session = conn.execute("select id::text, title from chat_sessions where id = %s", (session_id,)).fetchone()
        messages = conn.execute(
            """
            select id::text, session_id::text, role, content, metadata, created_at
            from chat_messages
            where session_id = %s
            order by created_at asc
            """,
            (session_id,),
        ).fetchall()
    return ExportResponse(session_id=session_id, title=session["title"], messages=messages)


@app.get("/sessions/{session_id}/export.txt")
def export_session_text(session_id: str, user: UserContext = Depends(current_user)) -> PlainTextResponse:
    payload = export_session(session_id=session_id, user=user)
    lines = [payload.title, ""]
    for message in payload.messages:
        lines.append(f"{message['role'].upper()}: {message['content']}")
        lines.append("")
    return PlainTextResponse("\n".join(lines), media_type="text/plain")


@app.post("/messages/{message_id}/retry", response_model=ChatResponse)
def retry_message(message_id: str, retry: RetryRequest, request: Request, user: UserContext = Depends(current_user)) -> ChatResponse:
    require_role(user, {"admin", "analyst"})
    with get_connection() as conn:
        message = conn.execute(
            """
            select id::text, session_id::text, role, content, metadata, created_at
            from chat_messages
            where id = %s
            """,
            (message_id,),
        ).fetchone()
    if not message:
        raise HTTPException(status_code=404, detail="Message not found")
    ensure_session_access(message["session_id"], user)
    question = message["content"] if message["role"] == "user" else find_previous_user_question(message["session_id"], message["created_at"])
    metadata = message.get("metadata") or {}
    return chat(
        ChatRequest(
            question=question,
            session_id=message["session_id"],
            top_k=retry.top_k,
            doc_id=metadata.get("doc_id"),
            source_type=metadata.get("source_type"),
        ),
        request=request,
        user=user,
    )


def find_previous_user_question(session_id: str, before_created_at: Any) -> str:
    with get_connection() as conn:
        row = conn.execute(
            """
            select content from chat_messages
            where session_id = %s and role = 'user' and created_at <= %s
            order by created_at desc
            limit 1
            """,
            (session_id, before_created_at),
        ).fetchone()
    if not row:
        raise HTTPException(status_code=404, detail="Original question not found")
    return row["content"]


def indexed_video_path(doc_id: str) -> Path:
    allowed_extensions = {".mp4", ".mov", ".m4v", ".webm"}
    workspace_root = project_root()
    allowed_roots = [
        (workspace_root / "data").resolve(),
        (workspace_root / "output").resolve(),
        (workspace_root / "deploy" / "databricks" / "artifacts" / "video_sources").resolve(),
    ]
    for chunk in load_chunks():
        if chunk.get("doc_id") != doc_id:
            continue
        source_type = chunk.get("source_type") or chunk.get("metadata", {}).get("source_type")
        if source_type != "video":
            continue
        raw_path = chunk.get("source_path") or chunk.get("source_pdf") or chunk.get("metadata", {}).get("source_path")
        if not raw_path:
            continue
        candidates = [Path(raw_path), workspace_root / raw_path]
        candidates.extend(root / Path(raw_path).name for root in allowed_roots)
        for candidate in candidates:
            if not candidate.exists() or candidate.suffix.lower() not in allowed_extensions:
                continue
            resolved = candidate.resolve()
            if any(resolved == root or root in resolved.parents for root in allowed_roots):
                return resolved
    raise HTTPException(status_code=404, detail="Video source not found")


@app.get("/media/videos/{doc_id}")
def video_source(doc_id: str, user: UserContext = Depends(current_user)) -> FileResponse:
    ensure_document_access(user, doc_id)
    path = indexed_video_path(doc_id)
    return FileResponse(path, media_type="video/mp4", filename=path.name)


def clip_cache_path(doc_id: str, start: float, end: float) -> Path:
    workspace_root = project_root()
    cache_dir = workspace_root / "output" / "video_clips"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_doc_id = "".join(char if char.isalnum() or char in "-_" else "-" for char in doc_id)[:140]
    return cache_dir / f"{safe_doc_id}-{int(start * 1000)}-{int(end * 1000)}.mp4"


def packaged_clip_path(doc_id: str, start: float, end: float) -> Path:
    workspace_root = project_root()
    cache_dir = workspace_root / "deploy" / "databricks" / "artifacts" / "video_clips"
    safe_doc_id = "".join(char if char.isalnum() or char in "-_" else "-" for char in doc_id)[:140]
    return cache_dir / f"{safe_doc_id}-{int(start * 1000)}-{int(end * 1000)}.mp4"


def build_video_clip(source_path: Path, clip_path: Path, start: float, end: float) -> None:
    if clip_path.exists() and clip_path.stat().st_size > 0:
        return
    duration = max(0.1, end - start)
    temporary_path = clip_path.with_suffix(".tmp.mp4")
    if temporary_path.exists():
        temporary_path.unlink()
    command = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start:.3f}",
        "-i",
        str(source_path),
        "-t",
        f"{duration:.3f}",
        "-map",
        "0:v:0",
        "-map",
        "0:a?",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "23",
        "-c:a",
        "aac",
        "-movflags",
        "+faststart",
        str(temporary_path),
    ]
    try:
        subprocess.run(command, check=True, capture_output=True, text=True, timeout=180)
        temporary_path.replace(clip_path)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=500, detail="FFmpeg is required to generate video clips") from exc
    except subprocess.SubprocessError as exc:
        if temporary_path.exists():
            temporary_path.unlink()
        raise HTTPException(status_code=500, detail="Unable to generate video clip") from exc


@app.get("/media/video-clips/{doc_id}")
def video_clip_source(
    doc_id: str,
    start: float = Query(ge=0),
    end: float = Query(gt=0),
    user: UserContext = Depends(current_user),
) -> FileResponse:
    ensure_document_access(user, doc_id)
    if end <= start:
        raise HTTPException(status_code=400, detail="Clip end time must be after start time")
    if end - start > 180:
        raise HTTPException(status_code=400, detail="Video clips are limited to 3 minutes")
    packaged_path = packaged_clip_path(doc_id, start, end)
    if packaged_path.exists():
        return FileResponse(packaged_path, media_type="video/mp4", filename=packaged_path.name)
    source_path = indexed_video_path(doc_id)
    clip_path = clip_cache_path(doc_id, start, end)
    build_video_clip(source_path=source_path, clip_path=clip_path, start=start, end=end)
    return FileResponse(clip_path, media_type="video/mp4", filename=clip_path.name)


class SinglePageApp(StaticFiles):
    def is_not_modified(self, response_headers, request_headers) -> bool:
        return False

    async def get_response(self, path: str, scope):
        try:
            response = await super().get_response(path, scope)
        except StarletteHTTPException as exc:
            if exc.status_code == 404:
                response = await super().get_response("index.html", scope)
            else:
                raise
        response.headers["Cache-Control"] = "no-store"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"
        return response


dist_dir = project_root() / "ui" / "dist"
if dist_dir.exists():
    app.mount("/", SinglePageApp(directory=dist_dir, html=True), name="frontend")
