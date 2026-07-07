from __future__ import annotations

import uuid
from typing import Any

from psycopg.types.json import Jsonb

from app.db.postgres import get_connection
from app.security.auth import UserContext


def log_audit_event(
    event_type: str,
    user: UserContext,
    request_id: str,
    metadata: dict[str, Any] | None = None,
) -> None:
    with get_connection() as conn:
        conn.execute(
            """
            insert into audit_events (id, request_id, user_id, tenant_id, event_type, metadata)
            values (%s, %s, %s, %s, %s, %s)
            """,
            (
                str(uuid.uuid4()),
                request_id,
                user.user_id,
                user.tenant_id,
                event_type,
                Jsonb(metadata or {}),
            ),
        )
