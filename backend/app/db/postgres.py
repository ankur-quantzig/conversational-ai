from __future__ import annotations

import json
import os
import re
import sqlite3
import time
from contextlib import contextmanager
from typing import Any, Iterator

import psycopg
from psycopg.rows import dict_row

from app.utils.files import output_dir


def use_postgres() -> bool:
    if os.getenv("DATABASE_URL", "").startswith(("postgresql://", "postgres://")):
        return True
    if os.getenv("CHAT_DB", "").lower() == "postgres":
        return True
    return os.getenv("POSTGRES_HOST") not in (None, "", "localhost", "127.0.0.1")


def database_url() -> str:
    if os.getenv("DATABASE_URL"):
        return os.environ["DATABASE_URL"]
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "chatbot")
    auth = f"{user}:{password}" if password else user
    return f"postgresql://{auth}@{host}:{port}/{database}"


def sqlite_path() -> str:
    return os.getenv("SQLITE_DB_PATH") or str(output_dir("chat_history.sqlite3"))


def _sqlite_value(value: Any) -> Any:
    if value.__class__.__name__ == "Jsonb" and hasattr(value, "obj"):
        return json.dumps(value.obj)
    return value


def _sqlite_sql(sql: str) -> str:
    replacements = {
        "id::text": "id",
        "s.id::text": "s.id",
        "session_id::text": "session_id",
        "count(m.id)::int": "count(m.id)",
        "now()": "CURRENT_TIMESTAMP",
    }
    for old, new in replacements.items():
        sql = sql.replace(old, new)
    sql = re.sub(r"%s", "?", sql)
    return sql


class SQLiteConnection:
    def __init__(self, path: str):
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row

    def execute(self, sql: str, params: tuple[Any, ...] | None = None):
        cursor = self.conn.execute(_sqlite_sql(sql), tuple(_sqlite_value(item) for item in (params or ())))
        self.conn.commit()
        return SQLiteCursor(cursor)

    def close(self) -> None:
        self.conn.close()


class SQLiteCursor:
    def __init__(self, cursor: sqlite3.Cursor):
        self.cursor = cursor
        self.rowcount = cursor.rowcount

    def fetchone(self) -> dict[str, Any] | None:
        row = self.cursor.fetchone()
        return _sqlite_row(row) if row else None

    def fetchall(self) -> list[dict[str, Any]]:
        return [_sqlite_row(row) for row in self.cursor.fetchall()]


def _sqlite_row(row: sqlite3.Row) -> dict[str, Any]:
    data = dict(row)
    for key in ("metadata", "payload"):
        if isinstance(data.get(key), str):
            data[key] = json.loads(data[key] or "{}")
    return data


@contextmanager
def get_connection() -> Iterator[psycopg.Connection[dict[str, Any]]]:
    if use_postgres():
        with psycopg.connect(database_url(), row_factory=dict_row) as conn:
            yield conn
        return

    path = sqlite_path()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = SQLiteConnection(path)
    try:
        yield conn
    finally:
        conn.close()


def wait_for_database(attempts: int = 30, delay_seconds: float = 1.0) -> None:
    if not use_postgres():
        with get_connection() as conn:
            conn.execute("select 1")
        return

    last_error: Exception | None = None
    for _ in range(attempts):
        try:
            with get_connection() as conn:
                conn.execute("select 1")
            return
        except Exception as exc:  # pragma: no cover - exercised by Docker startup timing
            last_error = exc
            time.sleep(delay_seconds)
    raise RuntimeError("PostgreSQL did not become ready") from last_error


def init_db() -> None:
    if not use_postgres():
        with get_connection() as conn:
            conn.execute(
                """
                create table if not exists chat_sessions (
                  id text primary key,
                  title text not null,
                  created_at text not null default CURRENT_TIMESTAMP,
                  updated_at text not null default CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                create table if not exists chat_messages (
                  id text primary key,
                  session_id text not null references chat_sessions(id) on delete cascade,
                  role text not null check (role in ('user', 'assistant', 'system')),
                  content text not null,
                  metadata text not null default '{}',
                  created_at text not null default CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                create index if not exists idx_chat_messages_session_created
                on chat_messages(session_id, created_at)
                """
            )
            try:
                conn.execute("alter table chat_sessions add column user_id text not null default 'local-dev'")
            except Exception:
                pass
            try:
                conn.execute("alter table chat_sessions add column tenant_id text not null default 'default'")
            except Exception:
                pass
            conn.execute(
                """
                create table if not exists audit_events (
                  id text primary key,
                  request_id text not null,
                  user_id text not null,
                  tenant_id text not null,
                  event_type text not null,
                  metadata text not null default '{}',
                  created_at text not null default CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                create table if not exists message_feedback (
                  id text primary key,
                  message_id text not null,
                  session_id text not null,
                  user_id text not null,
                  rating text not null check (rating in ('up', 'down')),
                  comment text not null default '',
                  created_at text not null default CURRENT_TIMESTAMP
                )
                """
            )
            conn.execute(
                """
                create table if not exists shared_messages (
                  id text primary key,
                  message_id text not null,
                  session_id text not null,
                  user_id text not null,
                  title text not null,
                  payload text not null default '{}',
                  created_at text not null default CURRENT_TIMESTAMP
                )
                """
            )
        return

    with get_connection() as conn:
        conn.execute(
            """
            create table if not exists chat_sessions (
              id uuid primary key,
              title text not null,
              user_id text not null default 'local-dev',
              tenant_id text not null default 'default',
              created_at timestamptz not null default now(),
              updated_at timestamptz not null default now()
            )
            """
        )
        conn.execute("alter table chat_sessions add column if not exists user_id text not null default 'local-dev'")
        conn.execute("alter table chat_sessions add column if not exists tenant_id text not null default 'default'")
        conn.execute(
            """
            create table if not exists chat_messages (
              id uuid primary key,
              session_id uuid not null references chat_sessions(id) on delete cascade,
              role text not null check (role in ('user', 'assistant', 'system')),
              content text not null,
              metadata jsonb not null default '{}'::jsonb,
              created_at timestamptz not null default now()
            )
            """
        )
        conn.execute(
            """
            create index if not exists idx_chat_messages_session_created
            on chat_messages(session_id, created_at)
            """
        )
        conn.execute(
            """
            create table if not exists audit_events (
              id uuid primary key,
              request_id text not null,
              user_id text not null,
              tenant_id text not null,
              event_type text not null,
              metadata jsonb not null default '{}'::jsonb,
              created_at timestamptz not null default now()
            )
            """
        )
        conn.execute(
            """
            create table if not exists message_feedback (
              id uuid primary key,
              message_id uuid not null references chat_messages(id) on delete cascade,
              session_id uuid not null references chat_sessions(id) on delete cascade,
              user_id text not null,
              rating text not null check (rating in ('up', 'down')),
              comment text not null default '',
              created_at timestamptz not null default now()
            )
            """
        )
        conn.execute(
            """
            create table if not exists shared_messages (
              id uuid primary key,
              message_id uuid not null references chat_messages(id) on delete cascade,
              session_id uuid not null references chat_sessions(id) on delete cascade,
              user_id text not null,
              title text not null,
              payload jsonb not null default '{}'::jsonb,
              created_at timestamptz not null default now()
            )
            """
        )
