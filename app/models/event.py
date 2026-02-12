"""Event and market models."""
import uuid
from datetime import datetime
from decimal import Decimal
from enum import Enum

from sqlalchemy import String, DateTime, Text, Numeric, ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class EventCategory(str, Enum):
    FED = "fed"
    WEATHER = "weather"
    POLITICS = "politics"
    EARNINGS = "earnings"
    CRYPTO = "crypto"
    SPORTS = "sports"
    OTHER = "other"


class EventStatus(str, Enum):
    ACTIVE = "active"
    RESOLVED = "resolved"
    CANCELED = "canceled"


class Event(Base):
    """Prediction market event."""
    
    __tablename__ = "events"
    
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    kalshi_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    polymarket_id: Mapped[str | None] = mapped_column(String(100), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    category: Mapped[str] = mapped_column(String(50), default=EventCategory.OTHER.value)
    resolution_date: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default=EventStatus.ACTIVE.value)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now()
    )
    
    # Relationships
    prices: Mapped[list["MarketPrice"]] = relationship(back_populates="event")
    research_reports: Mapped[list["ResearchReport"]] = relationship(back_populates="event")


class MarketPrice(Base):
    """Historical and current market prices."""
    
    __tablename__ = "market_prices"
    
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    event_id: Mapped[str] = mapped_column(String(36), ForeignKey("events.id"), index=True)
    platform: Mapped[str] = mapped_column(String(20))  # kalshi, polymarket
    yes_price: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    no_price: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    volume: Mapped[Decimal | None] = mapped_column(Numeric(20, 2), nullable=True)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    event: Mapped["Event"] = relationship(back_populates="prices")


class ResearchReport(Base):
    """AI-generated research report for an event."""
    
    __tablename__ = "research_reports"
    
    id: Mapped[str] = mapped_column(
        String(36), 
        primary_key=True, 
        default=lambda: str(uuid.uuid4())
    )
    event_id: Mapped[str] = mapped_column(String(36), ForeignKey("events.id"), index=True)
    probability: Mapped[Decimal] = mapped_column(Numeric(5, 4))
    confidence_low: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    confidence_high: Mapped[Decimal | None] = mapped_column(Numeric(5, 4), nullable=True)
    model_agreement: Mapped[int | None] = mapped_column(nullable=True)  # e.g., 3 of 4 models
    factors: Mapped[str | None] = mapped_column(Text, nullable=True)  # JSON string
    reasoning: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), 
        server_default=func.now()
    )
    
    # Relationships
    event: Mapped["Event"] = relationship(back_populates="research_reports")
