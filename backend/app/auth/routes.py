"""
app/auth/routes.py
───────────────────
Authentication endpoints.

POST /auth/login    → issue access + refresh tokens
POST /auth/refresh  → exchange refresh token for new access token
GET  /auth/me       → return current user info
POST /auth/logout   → (stateless — client discards tokens)
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr

from app.auth.security import (
    verify_password, create_token_pair,
    get_current_user, UserInToken,
    DEMO_USERS, Role, _decode_token,
)

router = APIRouter(prefix="/auth", tags=["Authentication"])


# ── Request / Response schemas ────────────────────────────────────────────────

class LoginRequest(BaseModel):
    email:    str
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/login", summary="Login and receive JWT tokens")
async def login(body: LoginRequest):
    """
    Authenticate with email + password.
    Returns an access token (short-lived) and refresh token (long-lived).

    Demo credentials:
      admin@warehouse.com    / admin123
      operator@warehouse.com / operator123
      analyst@warehouse.com  / analyst123
      viewer@warehouse.com   / viewer123
    """
    user = DEMO_USERS.get(body.email)
    if not user or not verify_password(body.password, user["password"]):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    tokens = create_token_pair(email=body.email, role=user["role"])
    return {
        **tokens.model_dump(),
        "user": {
            "email": body.email,
            "role":  user["role"].value,
        }
    }


@router.post("/refresh", summary="Refresh access token")
async def refresh_token(body: RefreshRequest):
    """Exchange a valid refresh token for a new access token."""
    payload = _decode_token(body.refresh_token, expected_type="refresh")
    tokens  = create_token_pair(email=payload.sub, role=Role(payload.role))
    return tokens


@router.get("/me", summary="Get current user info")
async def me(user: UserInToken = Depends(get_current_user)):
    """Return the authenticated user's profile."""
    return {
        "email":    user.email,
        "role":     user.role.value,
        "is_admin": user.is_admin,
        "permissions": _get_permissions(user.role),
    }


@router.post("/logout", summary="Logout (client-side token discard)")
async def logout(user: UserInToken = Depends(get_current_user)):
    """
    Stateless logout — client should discard tokens.
    In production add token to a Redis blocklist here.
    """
    return {"message": f"Logged out: {user.email}"}


def _get_permissions(role: Role) -> list[str]:
    base = ["view:cameras", "view:alerts", "view:livefeed"]
    if role in (Role.OPERATOR, Role.ANALYST, Role.ADMIN):
        base += ["resolve:alerts", "view:activities", "view:timeline"]
    if role in (Role.ANALYST, Role.ADMIN):
        base += ["view:analytics", "export:reports"]
    if role == Role.ADMIN:
        base += ["manage:cameras", "manage:users", "manage:rules", "manage:system"]
    return base
