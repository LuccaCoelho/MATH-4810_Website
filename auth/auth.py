from passlib.context import CryptContext
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
import os

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def _truncate(password: str) -> str:
    return password.encode("utf-8")[:72].decode("utf-8", errors="ignore")

def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(_truncate(plain), hashed)

def hash_password(plain: str) -> str:
    return pwd_context.hash(_truncate(plain))


# ── Session tokens ────────────────────────────────────────────────────────────
SESSION_COOKIE = "session"
SESSION_MAX_AGE = 60 * 60 * 8  # 8 hours


def _get_serializer() -> URLSafeTimedSerializer:
    secret = os.environ.get("SESSION_SECRET")
    if not secret:
        raise RuntimeError("SESSION_SECRET environment variable is not set.")
    return URLSafeTimedSerializer(secret)


def create_session_token(user_id: int) -> str:
    return _get_serializer().dumps({"user_id": user_id})


def decode_session_token(token: str) -> dict | None:
    try:
        return _get_serializer().loads(token, max_age=SESSION_MAX_AGE)
    except (BadSignature, SignatureExpired):
        return None