from __future__ import annotations

from datetime import date, datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import Boolean, Date, DateTime, Float, ForeignKey, Integer, String, Text, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

from config.settings import settings


def _utcnow_naive() -> datetime:
    from datetime import timezone
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Base(DeclarativeBase):
    pass


class User(Base):
    __tablename__ = "users"
    user_id: Mapped[str] = mapped_column(String(50), primary_key=True)
    name: Mapped[str] = mapped_column(String(120))
    monthly_income: Mapped[float] = mapped_column(Float, default=0)
    starting_balance: Mapped[float] = mapped_column(Float, default=0)


class TransactionRow(Base):
    __tablename__ = "transactions"
    transaction_id: Mapped[str] = mapped_column(String(80), primary_key=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    date: Mapped[date] = mapped_column(Date, index=True)
    description: Mapped[str] = mapped_column(String(255))
    merchant: Mapped[str] = mapped_column(String(120))
    amount: Mapped[float] = mapped_column(Float)
    transaction_type: Mapped[str] = mapped_column(String(20))
    account_type: Mapped[str] = mapped_column(String(30))
    payment_mode: Mapped[str] = mapped_column(String(30))
    category: Mapped[str] = mapped_column(String(50), default="Other")
    subcategory: Mapped[str] = mapped_column(String(80), default="")
    is_recurring: Mapped[bool] = mapped_column(Boolean, default=False)
    is_anomaly: Mapped[bool] = mapped_column(Boolean, default=False)
    is_duplicate: Mapped[bool] = mapped_column(Boolean, default=False)
    is_subscription: Mapped[bool] = mapped_column(Boolean, default=False)


class Budget(Base):
    __tablename__ = "budgets"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    month: Mapped[str] = mapped_column(String(7), index=True)
    category: Mapped[str] = mapped_column(String(50))
    allocated: Mapped[float] = mapped_column(Float)


class Goal(Base):
    __tablename__ = "goals"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    name: Mapped[str] = mapped_column(String(120))
    target_amount: Mapped[float] = mapped_column(Float)
    target_date: Mapped[date] = mapped_column(Date)
    current_amount: Mapped[float] = mapped_column(Float, default=0)
    monthly_contribution: Mapped[float] = mapped_column(Float, default=0)


class Subscription(Base):
    __tablename__ = "subscriptions"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    merchant: Mapped[str] = mapped_column(String(120))
    monthly_cost: Mapped[float] = mapped_column(Float)
    billing_frequency: Mapped[str] = mapped_column(String(30))
    observed_months: Mapped[int] = mapped_column(Integer)
    usage_evidence: Mapped[str] = mapped_column(String(50), default="unavailable")


class Alert(Base):
    __tablename__ = "alerts"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(ForeignKey("users.user_id"), index=True)
    severity: Mapped[str] = mapped_column(String(20))
    title: Mapped[str] = mapped_column(String(255))
    explanation: Mapped[str] = mapped_column(Text)
    recommended_action: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)


class AgentRun(Base):
    __tablename__ = "agent_runs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), index=True)
    agent_name: Mapped[str] = mapped_column(String(100))
    status: Mapped[str] = mapped_column(String(30))
    input_summary: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[str] = mapped_column(Text, default="")
    execution_time_ms: Mapped[float] = mapped_column(Float, default=0)
    output_summary: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)


class Recommendation(Base):
    __tablename__ = "recommendations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), index=True)
    text: Mapped[str] = mapped_column(Text)
    evidence: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)


class Simulation(Base):
    __tablename__ = "simulations"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), index=True)
    scenario: Mapped[str] = mapped_column(String(100))
    inputs_json: Mapped[str] = mapped_column(Text)
    result_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)


class FinancialHealthHistory(Base):
    __tablename__ = "financial_health_history"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[str] = mapped_column(String(50), index=True)
    month: Mapped[str] = mapped_column(String(7))
    score: Mapped[float] = mapped_column(Float)
    components_json: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=_utcnow_naive)


# SQLite is the default deployment target. check_same_thread=False keeps the
# connection usable across Streamlit reruns while SQLAlchemy sessions remain scoped.
engine = create_engine(
    settings.sqlite_url,
    future=True,
    connect_args={"check_same_thread": False} if settings.sqlite_url.startswith("sqlite") else {},
)
SessionLocal = sessionmaker(bind=engine, expire_on_commit=False, class_=Session)


def init_db() -> None:
    """Create all database tables if they do not exist."""
    if settings.sqlite_url.startswith("sqlite:///"):
        Path(settings.sqlite_url.removeprefix("sqlite:///" )).parent.mkdir(parents=True, exist_ok=True)
    Base.metadata.create_all(engine)


def get_session() -> Session:
    init_db()
    return SessionLocal()
