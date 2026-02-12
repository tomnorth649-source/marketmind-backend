"""Research endpoints."""
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.event import Event, ResearchReport
from app.api.v1.auth import get_current_user
from app.models.user import User

router = APIRouter()


class Factor(BaseModel):
    """Individual factor affecting probability."""
    name: str
    impact: Decimal
    source: str
    value: str | None = None


class ResearchResponse(BaseModel):
    """Research report response."""
    id: str
    event_id: str
    probability: Decimal
    confidence_low: Decimal | None
    confidence_high: Decimal | None
    model_agreement: int | None
    factors: list[Factor] | None = None
    reasoning: str | None
    
    # Edge vs market
    kalshi_edge: Decimal | None = None
    polymarket_edge: Decimal | None = None
    recommendation: str | None = None

    class Config:
        from_attributes = True


@router.get("/{event_id}", response_model=ResearchResponse | None)
async def get_research(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get latest research report for an event."""
    # Verify event exists
    event_result = await db.execute(select(Event).where(Event.id == event_id))
    event = event_result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # Get latest research report
    result = await db.execute(
        select(ResearchReport)
        .where(ResearchReport.event_id == event_id)
        .order_by(ResearchReport.created_at.desc())
        .limit(1)
    )
    report = result.scalar_one_or_none()
    
    if not report:
        return None
    
    return ResearchResponse.model_validate(report)


@router.post("/{event_id}/refresh")
async def refresh_research(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Trigger a new research run for an event (async)."""
    # Verify event exists
    event_result = await db.execute(select(Event).where(Event.id == event_id))
    event = event_result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    # TODO: Queue research job
    # For now, return placeholder
    return {
        "status": "queued",
        "message": f"Research refresh queued for event {event_id}",
        "event_title": event.title,
    }
