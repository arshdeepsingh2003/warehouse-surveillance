"""
app/auth/security.py
─────────────────────
JWT authentication + RBAC for the warehouse surveillance system.

Roles (least → most privileged):
  viewer   → live feed + alerts (read-only)
  operator → viewer + resolve alerts + activity log
  analyst  → operator + analytics + reports
  admin    → full access + camera management + user management

Token flow:
  POST /auth/login  →  {access_token, refresh_token}
  GET  /cameras     →  Authorization: Bearer <access_token>

Tokens are signed with HS256.  In production use RS256 with a
hardware key or AWS KMS to prevent key compromise.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Optional

import jwt
from fastapi import Depends, HTTPException, Security, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from passlib.context import CryptContext
from pydantic import BaseModel

# ── Config ────────────────────────────────────────────────────────────────────
SECRET_KEY      = os.environ.get("JWT_SECRET_KEY", "dev-secret-change-in-production-min-32-chars!")
ALGORITHM       = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES  = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", 60))
REFRESH_TOKEN_EXPIRE_DAYS    = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", 7))

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
_bearer  = HTTPBearer(auto_error=True)


# ── Roles ─────────────────────────────────────────────────────────────────────

class Role(str, Enum):
    VIEWER   = "viewer"
    OPERATOR = "operator"
    ANALYST  = "analyst"
    ADMIN    = "admin"

# Role hierarchy — each role includes all roles below it
_ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER:   0,
    Role.OPERATOR: 1,
    Role.ANALYST:  2,
    Role.ADMIN:    3,
}

def has_role(user_role: Role, required: Role) -> bool:
    return _ROLE_HIERARCHY.get(user_role, -1) >= _ROLE_HIERARCHY.get(required, 99)


# ── Token schemas ─────────────────────────────────────────────────────────────

class TokenPayload(BaseModel):
    sub:    str           # user email
    role:   Role
    exp:    datetime
    iat:    datetime
    type:   str           # "access" | "refresh"


class TokenPair(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int          # seconds until access token expires


class UserInToken(BaseModel):
    """Current user extracted from a valid token — injected by Depends."""
    email:    str
    role:     Role
    is_admin: bool


# ── Password utilities ────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return _pwd_ctx.hash(plain)

def verify_password(plain: str, hashed: str) -> bool:
    return _pwd_ctx.verify(plain, hashed)


# ── Token creation ────────────────────────────────────────────────────────────

def create_token_pair(email: str, role: Role) -> TokenPair:
    now = datetime.now(timezone.utc)

    access_payload = {
        "sub":  email,
        "role": role.value,
        "type": "access",
        "iat":  now,
        "exp":  now + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    }
    refresh_payload = {
        "sub":  email,
        "role": role.value,
        "type": "refresh",
        "iat":  now,
        "exp":  now + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS),
    }

    return TokenPair(
        access_token  = jwt.encode(access_payload,  SECRET_KEY, algorithm=ALGORITHM),
        refresh_token = jwt.encode(refresh_payload, SECRET_KEY, algorithm=ALGORITHM),
        expires_in    = ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )


# ── Token verification ────────────────────────────────────────────────────────

def _decode_token(token: str, expected_type: str = "access") -> TokenPayload:
    try:
        raw = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except jwt.ExpiredSignatureError:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=f"Invalid token: {e}")

    if raw.get("type") != expected_type:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail=f"Expected {expected_type} token")

    return TokenPayload(**raw)


# ── FastAPI dependency: get current user ──────────────────────────────────────

def get_current_user(
    credentials: HTTPAuthorizationCredentials = Security(_bearer),
) -> UserInToken:
    """
    FastAPI dependency. Extracts and validates the JWT from Authorization header.

    Usage in a route:
        @router.get("/cameras")
        async def list_cameras(user: UserInToken = Depends(get_current_user)):
            ...
    """
    payload = _decode_token(credentials.credentials, "access")
    return UserInToken(
        email=    payload.sub,
        role=     Role(payload.role),
        is_admin= payload.role == Role.ADMIN,
    )


def require_role(minimum_role: Role):
    """
    Dependency factory: require a minimum role level.

    Usage:
        @router.delete("/cameras/{id}")
        async def delete_camera(user = Depends(require_role(Role.ADMIN))):
            ...
    """
    def checker(user: UserInToken = Depends(get_current_user)) -> UserInToken:
        if not has_role(user.role, minimum_role):
            raise HTTPException(
                status.HTTP_403_FORBIDDEN,
                detail=f"Requires '{minimum_role.value}' role or higher. Your role: '{user.role.value}'",
            )
        return user
    return checker


# ── Convenience role dependencies ─────────────────────────────────────────────
require_viewer   = require_role(Role.VIEWER)
require_operator = require_role(Role.OPERATOR)
require_analyst  = require_role(Role.ANALYST)
require_admin    = require_role(Role.ADMIN)


# ── Demo user store (replace with DB in production) ───────────────────────────
# Passwords are bcrypt-hashed. Generate: hash_password("your-password")
DEMO_USERS: dict[str, dict] = {
    "admin@warehouse.com":    {"password": hash_password("admin123"),    "role": Role.ADMIN},
    "analyst@warehouse.com":  {"password": hash_password("analyst123"),  "role": Role.ANALYST},
    "operator@warehouse.com": {"password": hash_password("operator123"), "role": Role.OPERATOR},
    "viewer@warehouse.com":   {"password": hash_password("viewer123"),   "role": Role.VIEWER},
}
