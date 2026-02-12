"""Event endpoints."""
from datetime import datetime
from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.event import Event, EventCategory, EventStatus, MarketPrice
from app.api.v1.auth import get_current_user
from app.models.user import User

router = APIRouter()


class EventResponse(BaseModel):
    """Event response model."""
    id: str
    kalshi_id: str | None
    polymarket_id: str | None
    title: str
    description: str | None
    category: str
    resolution_date: datetime | None
    status: str
    created_at: datetime
    
    # Latest prices (populated separately)
    kalshi_yes_price: Decimal | None = None
    polymarket_yes_price: Decimal | None = None
    edge: Decimal | None = None

    class Config:
        from_attributes = True


class EventCreate(BaseModel):
    """Event creation request."""
    title: str
    description: str | None = None
    category: EventCategory = EventCategory.OTHER
    resolution_date: datetime | None = None
    kalshi_id: str | None = None
    polymarket_id: str | None = None


class EventList(BaseModel):
    """Paginated event list."""
    items: list[EventResponse]
    total: int
    page: int
    page_size: int


@router.get("", response_model=EventList)
async def list_events(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    category: EventCategory | None = None,
    status: EventStatus | None = None,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List all events with pagination."""
    query = select(Event)
    
    if category:
        query = query.where(Event.category == category.value)
    if status:
        query = query.where(Event.status == status.value)
    
    # Get total count
    count_result = await db.execute(select(Event.id).where(True))
    total = len(count_result.all())
    
    # Get paginated results
    query = query.offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    events = result.scalars().all()
    
    return EventList(
        items=[EventResponse.model_validate(e) for e in events],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{event_id}", response_model=EventResponse)
async def get_event(
    event_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get a single event by ID."""
    result = await db.execute(select(Event).where(Event.id == event_id))
    event = result.scalar_one_or_none()
    
    if not event:
        raise HTTPException(status_code=404, detail="Event not found")
    
    return EventResponse.model_validate(event)


@router.post("", response_model=EventResponse)
async def create_event(
    event_data: EventCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Create a new event (admin only for now)."""
    event = Event(
        title=event_data.title,
        description=event_data.description,
        category=event_data.category.value,
        resolution_date=event_data.resolution_date,
        kalshi_id=event_data.kalshi_id,
        polymarket_id=event_data.polymarket_id,
    )
    db.add(event)
    await db.flush()
    await db.refresh(event)
    
    return EventResponse.model_validate(event)
