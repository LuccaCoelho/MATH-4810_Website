from fastapi import Cookie, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from sqlalchemy.orm import Session
from sqlalchemy import select
from typing import Annotated

from db.database import get_db
from models.models import User
from auth.auth import decode_session_token, SESSION_COOKIE


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
) -> User:
    token = request.cookies.get(SESSION_COOKIE)

    if not token:
        raise _redirect_to_login()

    payload = decode_session_token(token)
    if payload is None:
        raise _redirect_to_login()

    user = db.execute(
        select(User).where(User.id == payload["user_id"])
    ).scalars().first()

    if user is None:
        raise _redirect_to_login()

    return user


def _redirect_to_login():
    # We raise a redirect as an HTTPException so FastAPI unwinds the call stack
    # cleanly. The exception handler in main.py converts it to a real redirect.
    return HTTPException(status_code=307, detail="Not authenticated", headers={"Location": "/"})


# Convenience alias for use in route signatures:
#   current_user: CurrentUser
CurrentUser = Annotated[User, Depends(get_current_user)]