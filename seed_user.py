"""
Usage:
    python seed_user.py <username> <email> <password>

Example:
    python seed_user.py lucca lucca@example.com mysecretpassword

Run this from the project root (same directory as main.py).
"""
import sys
from db.database import SessionLocal, engine, Base
from models.models import User
from auth.auth import hash_password

Base.metadata.create_all(bind=engine)

def seed_user(username: str, email: str, password: str):
    db = SessionLocal()
    try:
        existing = db.query(User).filter(
            (User.username == username) | (User.email == email)
        ).first()
        if existing:
            print(f"[ERROR] User with username '{username}' or email '{email}' already exists.")
            sys.exit(1)

        user = User(
            username=username,
            email=email,
            password_hash=hash_password(password),
        )
        db.add(user)
        db.commit()
        db.refresh(user)
        print(f"[OK] Created user '{user.username}' (id={user.id})")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python seed_user.py <username> <email> <password>")
        sys.exit(1)
    seed_user(sys.argv[1], sys.argv[2], sys.argv[3])