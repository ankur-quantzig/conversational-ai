from __future__ import annotations

from dataclasses import dataclass, field

from typing import Annotated

from fastapi import Header, HTTPException, status

from app.config import app_api_keys, basic_users, is_databricks_env, is_local_env, power_users


@dataclass(frozen=True)
class UserContext:
    user_id: str
    email: str = ""
    name: str = ""
    tenant_id: str = "default"
    roles: list[str] = field(default_factory=lambda: ["admin"])
    document_ids: list[str] = field(default_factory=lambda: ["*"])

    @property
    def is_admin(self) -> bool:
        return "admin" in self.roles

    @property
    def normalized_email(self) -> str:
        return (self.email or self.user_id).lower()

    @property
    def is_power_user(self) -> bool:
        return self.is_admin or "power_user" in self.roles or self.normalized_email in power_users()


def current_user(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None),
    x_forwarded_email: Annotated[str | None, Header(alias="X-Forwarded-Email")] = None,
    x_forwarded_user: Annotated[str | None, Header(alias="X-Forwarded-User")] = None,
    x_databricks_user_email: Annotated[str | None, Header(alias="X-Databricks-User-Email")] = None,
) -> UserContext:
    configured_keys = app_api_keys()
    if not configured_keys:
        if is_databricks_env():
            email = _first_non_empty(x_databricks_user_email, x_forwarded_email, x_forwarded_user, "databricks-user").lower()
            return UserContext(
                user_id=email,
                email=email,
                roles=_roles_for(email=email, payload_roles=None),
                document_ids=["*"],
            )
        if not is_local_env():
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Authentication is not configured",
            )
        return UserContext(
            user_id="local-dev",
            email="ankurkumarj@quantzig.com",
            name="Ankur Kumar",
            roles=["admin", "power_user"],
            document_ids=["*"],
        )

    token = _extract_token(authorization=authorization, x_api_key=x_api_key)
    if not token or token not in configured_keys:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing or invalid API key")

    payload = configured_keys[token] or {}
    email = str(payload.get("email") or payload.get("user_id") or "api-user").lower()
    roles = _roles_for(email=email, payload_roles=payload.get("roles"))
    return UserContext(
        user_id=str(payload.get("user_id") or email),
        email=email,
        name=str(payload.get("name") or ""),
        tenant_id=str(payload.get("tenant_id") or "default"),
        roles=roles,
        document_ids=list(payload.get("document_ids") or []),
    )


def _extract_token(authorization: str | None, x_api_key: str | None) -> str:
    if x_api_key:
        return x_api_key.strip()
    if authorization and authorization.lower().startswith("bearer "):
        return authorization[7:].strip()
    return ""


def _first_non_empty(*values: str | None) -> str:
    for value in values:
        if value and value.strip():
            return value.strip()
    return ""


def _roles_for(email: str, payload_roles: object) -> list[str]:
    if isinstance(payload_roles, str):
        configured_roles = [payload_roles]
    elif isinstance(payload_roles, list):
        configured_roles = [str(role) for role in payload_roles]
    else:
        configured_roles = []
    roles = configured_roles or ["analyst"]
    normalized_email = email.lower()
    if normalized_email in power_users() and "power_user" not in roles:
        roles.append("power_user")
    if normalized_email in basic_users() and "basic_user" not in roles:
        roles.append("basic_user")
    return roles


def require_role(user: UserContext, allowed_roles: set[str]) -> None:
    if user.is_admin:
        return
    if not allowed_roles.intersection(user.roles):
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient role")


def ensure_document_access(user: UserContext, doc_id: str | None) -> None:
    if user.is_admin or "*" in user.document_ids:
        return
    if not doc_id:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Select a permitted document before asking a question",
        )
    if doc_id not in user.document_ids:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Document access denied")
