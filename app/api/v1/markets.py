"""Market data endpoints - Kalshi integration."""
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.v1.auth import get_current_user
from app.models.user import User
from app.integrations.kalshi import get_kalshi_client

router = APIRouter()


class MarketResponse(BaseModel):
    """Market response with price data."""
    ticker: str
    title: str
    subtitle: str | None = None
    status: str
    yes_price: float | None = None  # 0-100 (cents)
    no_price: float | None = None
    volume: int | None = None
    close_time: str | None = None


class MarketsListResponse(BaseModel):
    """List of markets."""
    markets: list[MarketResponse]
    cursor: str | None = None


@router.get("/kalshi", response_model=MarketsListResponse)
async def list_kalshi_markets(
    status: str = Query("open", description="Market status filter"),
    series: str | None = Query(None, description="Series ticker (e.g., FED, WEATHER)"),
    limit: int = Query(20, ge=1, le=100),
    cursor: str | None = None,
    current_user: User = Depends(get_current_user),
):
    """List markets from Kalshi."""
    try:
        client = get_kalshi_client()
        result = await client.get_markets(
            status=status,
            series_ticker=series,
            limit=limit,
            cursor=cursor,
        )
        
        markets = []
        for m in result.get("markets", []):
            markets.append(MarketResponse(
                ticker=m.get("ticker", ""),
                title=m.get("title", ""),
                subtitle=m.get("subtitle"),
                status=m.get("status", ""),
                yes_price=m.get("yes_ask"),  # Current ask price
                no_price=m.get("no_ask"),
                volume=m.get("volume"),
                close_time=m.get("close_time"),
            ))
        
        return MarketsListResponse(
            markets=markets,
            cursor=result.get("cursor"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kalshi API error: {str(e)}")


@router.get("/kalshi/{ticker}", response_model=MarketResponse)
async def get_kalshi_market(
    ticker: str,
    current_user: User = Depends(get_current_user),
):
    """Get single market details from Kalshi."""
    try:
        client = get_kalshi_client()
        result = await client.get_market(ticker)
        m = result.get("market", {})
        
        return MarketResponse(
            ticker=m.get("ticker", ticker),
            title=m.get("title", ""),
            subtitle=m.get("subtitle"),
            status=m.get("status", ""),
            yes_price=m.get("yes_ask"),
            no_price=m.get("no_ask"),
            volume=m.get("volume"),
            close_time=m.get("close_time"),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kalshi API error: {str(e)}")


@router.get("/kalshi/series/list")
async def list_kalshi_series(
    current_user: User = Depends(get_current_user),
):
    """List all event series (categories) from Kalshi."""
    try:
        client = get_kalshi_client()
        result = await client.get_series()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Kalshi API error: {str(e)}")
