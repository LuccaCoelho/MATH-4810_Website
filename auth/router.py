from fastapi import APIRouter, Depends, Request, Form
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session
from sqlalchemy import select

from db.database import get_db
from models.models import User
from auth.auth import verify_password, create_session_token, SESSION_COOKIE, SESSION_MAX_AGE

router = APIRouter()
templates = Jinja2Templates(directory="templates")


@router.get("/", include_in_schema=False)
async def login_page(request: Request):
    """Serve the login form. If already authenticated, redirect straight to /main."""
    token = request.cookies.get(SESSION_COOKIE)
    if token:
        return RedirectResponse("/main", status_code=302)
    return templates.TemplateResponse(request, "index.html", {})


@router.post("/login", include_in_schema=False)
async def login(
    request: Request,
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db),
):
    user = db.execute(
        select(User).where(User.username == username)
    ).scalars().first()

    # Deliberately vague error — never tell the user which field was wrong.
    if user is None or not verify_password(password, user.password_hash):
        return templates.TemplateResponse(
            request,
            "index.html",
            {"error": "Invalid credentials."},
            status_code=401,
        )

    token = create_session_token(user.id)

    response = RedirectResponse("/main", status_code=302)
    response.set_cookie(
        key=SESSION_COOKIE,
        value=token,
        httponly=True,     
        samesite="lax",   
        secure=False,     
        max_age=SESSION_MAX_AGE,
    )
    return response


@router.get("/logout", include_in_schema=False)
async def logout():
    response = RedirectResponse("/", status_code=302)
    response.delete_cookie(SESSION_COOKIE)
    return response