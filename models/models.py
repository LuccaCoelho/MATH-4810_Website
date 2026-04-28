from sqlalchemy import ForeignKey, Integer, String, DateTime, Float, JSON
from sqlalchemy.orm import relationship, Mapped, mapped_column

from db.database import Base
from datetime import datetime
from typing import List, Optional


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    username: Mapped[str] = mapped_column(String(50), unique=True, nullable=False)
    email: Mapped[str] = mapped_column(String(120), unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    valuations: Mapped[List["Valuation"]] = relationship("Valuation", back_populates="user")

    def __repr__(self):
        return f"<User(username='{self.username}', email='{self.email}')>"


class Valuation(Base):
    __tablename__ = "valuations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"), nullable=False)

    # Model identity
    model_type: Mapped[str] = mapped_column(String(50), nullable=False)        # "linear" | "nonlinear"
    segment: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    n_features: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # linear only

    # Result
    valuation_result: Mapped[float] = mapped_column(Float, nullable=False)
    pi_lo: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    pi_hi: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    baseline_dollars: Mapped[Optional[float]] = mapped_column(Float, nullable=True, default=0.0)

    # Feature contributions — three parallel lists
    # features  : list[str]   — human-readable feature group names
    # coefs     : list[float] — dollar impact per feature
    # feat_vals : list[float] — raw feature value (nonlinear only; empty list for linear)
    features: Mapped[list] = mapped_column(JSON, nullable=False)
    coefs: Mapped[list] = mapped_column(JSON, nullable=False)
    feat_vals: Mapped[Optional[list]] = mapped_column(JSON, nullable=True, default=list)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    user: Mapped["User"] = relationship("User", back_populates="valuations")

    def __repr__(self):
        return f"<Valuation(id={self.id}, model='{self.model_type}', result={self.valuation_result})>"