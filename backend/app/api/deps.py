"""Shared API dependencies: database session, current user, role checks (TRD SEC-04)."""

from collections.abc import Callable
from typing import Annotated

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session

from app.core.security import decode_access_token
from app.db.models import AnalysisSession, User
from app.db.session import get_db

oauth2 = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/token", auto_error=False)
DB = Annotated[Session, Depends(get_db)]
ROLE_RANK = {"agent": 1, "analyst": 2, "admin": 3}


def user_from_token(db: Session, token: str | None) -> User:
    unauthorized = HTTPException(
        status.HTTP_401_UNAUTHORIZED, "Not authenticated", headers={"WWW-Authenticate": "Bearer"}
    )
    if not token:
        raise unauthorized
    try:
        payload = decode_access_token(token)
    except jwt.PyJWTError as exc:
        raise unauthorized from exc
    user = db.get(User, payload.get("sub"))
    if user is None or not user.is_active:
        raise unauthorized
    return user


def current_user(db: DB, token: Annotated[str | None, Depends(oauth2)]) -> User:
    return user_from_token(db, token)


CurrentUser = Annotated[User, Depends(current_user)]


def require_role(minimum: str) -> Callable[[User], User]:
    def check(user: CurrentUser) -> User:
        if ROLE_RANK.get(user.role, 0) < ROLE_RANK[minimum]:
            raise HTTPException(status.HTTP_403_FORBIDDEN, f"Requires {minimum} role")
        return user

    return check


def owned_session(db: Session, session_id: str, user: User) -> AnalysisSession:
    """Org-scoped lookup; agents may only access their own sessions."""
    s = db.get(AnalysisSession, session_id)
    if s is None or s.org_id != user.org_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    if user.role == "agent" and s.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Session not found")
    return s
