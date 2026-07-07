from __future__ import annotations

from fastapi import HTTPException, status

from app.config import basic_user_question_limit
from app.db.postgres import get_connection
from app.security.auth import UserContext


def enforce_question_quota(user: UserContext) -> dict[str, int | None]:
    limit = question_limit_for(user)
    used = questions_used(user)
    if limit is not None and used >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="I am unable to generate the response at the moment. Please contact Admin.",
        )
    return {"used": used, "limit": limit}


def question_limit_for(user: UserContext) -> int | None:
    if user.is_power_user:
        return None
    if "basic_user" in user.roles:
        return basic_user_question_limit()
    return None


def questions_used(user: UserContext) -> int:
    with get_connection() as conn:
        row = conn.execute(
            """
            select count(m.id)::int as question_count
            from chat_messages m
            join chat_sessions s on s.id = m.session_id
            where s.user_id = %s and s.tenant_id = %s and m.role = 'user'
            """,
            (user.user_id, user.tenant_id),
        ).fetchone()
    return int((row or {}).get("question_count") or 0)
