from datetime import UTC, datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy import select

from app.api.deps import DB, CurrentUser
from app.core.security import create_access_token, verify_password
from app.db.models import AuditLog, User
from app.schemas import LoginRequest, TokenResponse, UserOut
from app.services.rate_limit import limiter

router = APIRouter(prefix="/auth", tags=["auth"])


def _login(db: DB, email: str, password: str, request: Request) -> TokenResponse:
    limiter.check(f"login:{request.client.host if request.client else 'unknown'}")
    user = db.scalar(select(User).where(User.email == email.lower().strip()))
    if user is None or not user.is_active or not verify_password(password, user.password_hash):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")
    token, expires = create_access_token(user.id, user.role)
    user.last_login_at = datetime.now(UTC)
    db.add(
        AuditLog(
            org_id=user.org_id,
            user_id=user.id,
            action="login",
            resource="user",
            resource_id=user.id,
        )
    )
    db.commit()
    return TokenResponse(access_token=token, expires_in=expires, user=UserOut.model_validate(user))


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: DB, request: Request) -> TokenResponse:
    return _login(db, body.email, body.password, request)


@router.post("/token", response_model=TokenResponse, include_in_schema=False)
def token(
    form: Annotated[OAuth2PasswordRequestForm, Depends()], db: DB, request: Request
) -> TokenResponse:
    """OAuth2 form login used by the interactive API docs' Authorize button."""
    return _login(db, form.username, form.password, request)


@router.get("/me", response_model=UserOut)
def me(user: CurrentUser) -> UserOut:
    return UserOut.model_validate(user)
