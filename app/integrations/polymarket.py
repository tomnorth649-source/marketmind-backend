"""Polymarket API client for market data.

Uses:
- Gamma API: Market discovery, metadata, events
- CLOB API: Real-time prices, orderbooks

No authentication required for read-only access.
"""
import json
from datetime import datetime
from typing import Any

import httpx
from pydantic import BaseModel

from app.config import get_settings


class PolymarketMarket(BaseModel):
    """Polymarket market model."""
    id: str
    question: str
    slug: str | None = None
    outcomes: list[str]
    outcome_prices: list[float]
    clob_token_ids: list[str] | None = None
    volume: float | None = None
    liquidity: float | None = None
    end_date: datetime | None = None
    active: bool = True
    resolved: bool = False
    
    @property
    def yes_price(self) -> float | None:
        """Get YES price (first outcome)."""
        return self.outcome_prices[0] if self.outcome_prices else None
    
    @property
    def no_price(self) -> float | None:
        """Get NO price (second outcome)."""
        return self.outcome_prices[1] if len(self.outcome_prices) > 1 else None


class PolymarketEvent(BaseModel):
    """Polymarket event (contains multiple markets)."""
    id: str
    slug: str
    title: str
    active: bool = True
    closed: bool = False
    tags: list[dict] = []
    markets: list[dict] = []


class PolymarketClient:
    """Client for Polymarket APIs.
    
    Read-only market data access - no authentication required.
    
    APIs:
    - Gamma API: Market discovery & metadata
    - CLOB API: Real-time prices & orderbooks
    """
    
    GAMMA_URL = "https://gamma-api.polymarket.com"
    CLOB_URL = "https://clob.polymarket.com"
    
    def __init__(self):
        settings = get_settings()
        # Credentials only needed for trading, not for read access
        self._api_key = getattr(settings, 'polymarket_api_key', None)
        
    async def _gamma_request(
        self, 
        endpoint: str, 
        params: dict | None = None,
    ) -> Any:
        """Make request to Gamma API (no auth needed)."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.GAMMA_URL}{endpoint}",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def _clob_request(
        self, 
        endpoint: str, 
        params: dict | None = None,
    ) -> Any:
        """Make request to CLOB API (no auth needed for reads)."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.CLOB_URL}{endpoint}",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    # ==================== GAMMA API (Market Discovery) ====================
    
    async def get_events(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
        tag_id: int | None = None,
        series_id: int | None = None,
        order: str = "volume24hr",
        ascending: bool = False,
    ) -> list[dict]:
        """Get list of events (groups of related markets).
        
        Args:
            active: Filter active events
            closed: Filter closed events  
            limit: Max results
            offset: Pagination offset
            tag_id: Filter by category tag
            series_id: Filter by series (e.g., sports leagues)
            order: Sort field (volume24hr, startTime, etc.)
            ascending: Sort direction
        """
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
            "order": order,
            "ascending": str(ascending).lower(),
        }
        if tag_id:
            params["tag_id"] = tag_id
        if series_id:
            params["series_id"] = series_id
            
        return await self._gamma_request("/events", params=params)
    
    async def get_event(self, event_id: str = None, slug: str = None) -> dict | None:
        """Get single event by ID or slug."""
        if event_id:
            events = await self._gamma_request("/events", params={"id": event_id})
        elif slug:
            events = await self._gamma_request("/events", params={"slug": slug})
        else:
            return None
        return events[0] if events else None
    
    async def get_markets(
        self,
        active: bool = True,
        closed: bool = False,
        limit: int = 100,
        offset: int = 0,
    ) -> list[dict]:
        """Get list of markets."""
        params = {
            "active": str(active).lower(),
            "closed": str(closed).lower(),
            "limit": limit,
            "offset": offset,
        }
        return await self._gamma_request("/markets", params=params)
    
    async def get_market(self, market_id: str = None, slug: str = None) -> dict | None:
        """Get single market by ID or slug."""
        if market_id:
            markets = await self._gamma_request("/markets", params={"id": market_id})
        elif slug:
            markets = await self._gamma_request("/markets", params={"slug": slug})
        else:
            return None
        return markets[0] if markets else None
    
    async def get_tags(self, limit: int = 100) -> list[dict]:
        """Get available tags/categories."""
        return await self._gamma_request("/tags", params={"limit": limit})
    
    async def get_sports(self) -> list[dict]:
        """Get available sports leagues."""
        return await self._gamma_request("/sports")
    
    async def search_events(self, query: str, limit: int = 20) -> list[dict]:
        """Search events by keyword."""
        return await self._gamma_request(
            "/events", 
            params={
                "title_contains": query,
                "active": "true",
                "closed": "false",
                "limit": limit,
            }
        )
    
    # ==================== CLOB API (Real-Time Prices) ====================
    
    async def get_price(self, token_id: str, side: str = "buy") -> dict:
        """Get current price for a token.
        
        Args:
            token_id: The clobTokenId from market data
            side: 'buy' or 'sell'
        """
        return await self._clob_request(
            "/price",
            params={"token_id": token_id, "side": side}
        )
    
    async def get_orderbook(self, token_id: str) -> dict:
        """Get orderbook for a token."""
        return await self._clob_request(
            "/book",
            params={"token_id": token_id}
        )
    
    async def get_midpoint(self, token_id: str) -> dict:
        """Get midpoint price for a token."""
        return await self._clob_request(
            "/midpoint",
            params={"token_id": token_id}
        )
    
    async def get_spread(self, token_id: str) -> dict:
        """Get bid-ask spread for a token."""
        return await self._clob_request(
            "/spread",
            params={"token_id": token_id}
        )
    
    # ==================== Helper Methods ====================
    
    def parse_market(self, raw_market: dict) -> PolymarketMarket:
        """Parse raw market data into PolymarketMarket model."""
        # Parse outcomes from JSON string
        outcomes_str = raw_market.get("outcomes", "[]")
        outcomes = json.loads(outcomes_str) if isinstance(outcomes_str, str) else outcomes_str
        
        # Parse prices from JSON string
        prices_str = raw_market.get("outcomePrices", "[]")
        prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
        outcome_prices = [float(p) for p in prices] if prices else []
        
        # Parse token IDs
        token_ids = raw_market.get("clobTokenIds", [])
        if isinstance(token_ids, str):
            token_ids = json.loads(token_ids)
        
        return PolymarketMarket(
            id=raw_market.get("id", ""),
            question=raw_market.get("question", ""),
            slug=raw_market.get("slug"),
            outcomes=outcomes,
            outcome_prices=outcome_prices,
            clob_token_ids=token_ids,
            volume=float(raw_market.get("volume", 0) or 0),
            liquidity=float(raw_market.get("liquidity", 0) or 0),
            active=raw_market.get("active", True),
            resolved=raw_market.get("resolved", False),
        )


# Singleton instance
_client: PolymarketClient | None = None


def get_polymarket_client() -> PolymarketClient:
    """Get or create Polymarket client instance."""
    global _client
    if _client is None:
        _client = PolymarketClient()
    return _client
