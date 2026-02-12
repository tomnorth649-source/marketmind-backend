"""Polymarket API endpoints."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.integrations.polymarket import get_polymarket_client

router = APIRouter(prefix="/polymarket", tags=["polymarket"])


@router.get("/events")
async def get_events(
    active: bool = True,
    closed: bool = False,
    limit: int = Query(default=50, le=100),
    offset: int = 0,
    tag_slug: Optional[str] = None,
):
    """Get Polymarket events.
    
    Args:
        active: Filter active events
        closed: Filter closed events
        limit: Max results (max 100)
        offset: Pagination offset
        tag_slug: Filter by category (e.g., 'fed', 'politics', 'crypto')
    """
    client = get_polymarket_client()
    
    try:
        # If tag_slug provided, need to find tag_id first
        tag_id = None
        if tag_slug:
            tags = await client.get_tags(limit=200)
            for tag in tags:
                if tag.get('slug') == tag_slug:
                    tag_id = tag.get('id')
                    break
        
        events = await client.get_events(
            active=active,
            closed=closed,
            limit=limit,
            offset=offset,
            tag_id=int(tag_id) if tag_id else None,
        )
        
        # Simplify response
        simplified = []
        for event in events:
            markets = event.get('markets', [])
            first_market = markets[0] if markets else {}
            
            simplified.append({
                'id': event.get('id'),
                'slug': event.get('slug'),
                'title': event.get('title'),
                'active': event.get('active'),
                'markets_count': len(markets),
                'volume': first_market.get('volume'),
                'tags': [t.get('slug') for t in event.get('tags', [])],
                'first_market': {
                    'question': first_market.get('question'),
                    'outcomes': first_market.get('outcomes'),
                    'outcome_prices': first_market.get('outcomePrices'),
                } if first_market else None,
            })
        
        return {'events': simplified, 'count': len(simplified)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/events/{event_slug}")
async def get_event(event_slug: str):
    """Get single Polymarket event by slug."""
    client = get_polymarket_client()
    
    try:
        event = await client.get_event(slug=event_slug)
        if not event:
            raise HTTPException(status_code=404, detail="Event not found")
        return event
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/search")
async def search_events(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(default=20, le=50),
):
    """Search Polymarket events by title."""
    client = get_polymarket_client()
    
    try:
        events = await client.search_events(q, limit=limit)
        
        simplified = []
        for event in events:
            markets = event.get('markets', [])
            first_market = markets[0] if markets else {}
            
            simplified.append({
                'id': event.get('id'),
                'slug': event.get('slug'),
                'title': event.get('title'),
                'markets_count': len(markets),
                'outcome_prices': first_market.get('outcomePrices'),
            })
        
        return {'query': q, 'events': simplified, 'count': len(simplified)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/tags")
async def get_tags(limit: int = Query(default=50, le=200)):
    """Get available Polymarket categories/tags."""
    client = get_polymarket_client()
    
    try:
        tags = await client.get_tags(limit=limit)
        return {
            'tags': [{'id': t.get('id'), 'slug': t.get('slug'), 'label': t.get('label')} 
                     for t in tags],
            'count': len(tags),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{token_id}")
async def get_price(token_id: str, side: str = "buy"):
    """Get current price for a Polymarket token.
    
    Args:
        token_id: The clobTokenId from market data
        side: 'buy' or 'sell'
    """
    client = get_polymarket_client()
    
    try:
        price = await client.get_price(token_id, side)
        return price
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/orderbook/{token_id}")
async def get_orderbook(token_id: str):
    """Get orderbook for a Polymarket token."""
    client = get_polymarket_client()
    
    try:
        book = await client.get_orderbook(token_id)
        return book
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
