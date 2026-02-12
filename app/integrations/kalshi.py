"""Kalshi API client for market data."""
import base64
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from pydantic import BaseModel

from app.config import get_settings


class KalshiMarket(BaseModel):
    """Kalshi market/event model."""
    ticker: str
    title: str
    subtitle: str | None = None
    status: str
    yes_price: float | None = None  # In cents (0-100)
    no_price: float | None = None
    volume: int | None = None
    open_interest: int | None = None
    close_time: datetime | None = None
    category: str | None = None


class KalshiClient:
    """Client for Kalshi Exchange API.
    
    Uses RSA signature authentication for API access.
    Docs: https://trading-api.readme.io/reference/getting-started
    """
    
    # Elections/Prediction API 
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    def __init__(self):
        settings = get_settings()
        self.api_key = settings.kalshi_api_key
        self._private_key = None
        self._private_key_path = getattr(settings, 'kalshi_private_key_path', None)
        
    def _load_private_key(self):
        """Load RSA private key for signing requests."""
        if self._private_key is None and self._private_key_path:
            key_path = Path(self._private_key_path)
            if key_path.exists():
                with open(key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(),
                        password=None,
                    )
        return self._private_key
    
    def _sign_request(self, method: str, path: str, timestamp: int) -> str:
        """Create RSA signature for request authentication."""
        private_key = self._load_private_key()
        if not private_key:
            raise ValueError("Private key not loaded")
        
        # Signature payload: timestamp + method + path
        message = f"{timestamp}{method}{path}".encode()
        
        signature = private_key.sign(
            message,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        
        return base64.b64encode(signature).decode()
    
    def _get_headers(self, method: str, path: str) -> dict:
        """Generate authenticated headers for request."""
        timestamp = int(time.time() * 1000)  # Milliseconds
        signature = self._sign_request(method, path, timestamp)
        
        return {
            "KALSHI-ACCESS-KEY": self.api_key,
            "KALSHI-ACCESS-SIGNATURE": signature,
            "KALSHI-ACCESS-TIMESTAMP": str(timestamp),
            "Content-Type": "application/json",
        }
    
    async def _request(
        self, 
        method: str, 
        endpoint: str, 
        params: dict | None = None,
        json_data: dict | None = None,
    ) -> dict[str, Any]:
        """Make authenticated request to Kalshi API."""
        path = f"/trade-api/v2{endpoint}"
        headers = self._get_headers(method, path)
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=f"{self.BASE_URL}{endpoint}",
                headers=headers,
                params=params,
                json=json_data,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_events(
        self,
        status: str = "open",
        series_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Get list of events (series of markets).
        
        Args:
            status: Filter by status (open, closed, settled)
            series_ticker: Filter by series (e.g., "FED" for Fed events)
            limit: Max results per page
            cursor: Pagination cursor
        """
        params = {"status": status, "limit": limit}
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
            
        return await self._request("GET", "/events", params=params)
    
    async def get_event(self, event_ticker: str) -> dict:
        """Get single event details."""
        return await self._request("GET", f"/events/{event_ticker}")
    
    async def get_markets(
        self,
        status: str = "open",
        event_ticker: str | None = None,
        series_ticker: str | None = None,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Get list of markets (individual contracts).
        
        Args:
            status: Filter by status
            event_ticker: Filter by parent event
            series_ticker: Filter by series
            limit: Max results
            cursor: Pagination cursor
        """
        params = {"status": status, "limit": limit}
        if event_ticker:
            params["event_ticker"] = event_ticker
        if series_ticker:
            params["series_ticker"] = series_ticker
        if cursor:
            params["cursor"] = cursor
            
        return await self._request("GET", "/markets", params=params)
    
    async def get_market(self, ticker: str) -> dict:
        """Get single market details including current prices."""
        return await self._request("GET", f"/markets/{ticker}")
    
    async def get_market_orderbook(self, ticker: str, depth: int = 10) -> dict:
        """Get order book for a market."""
        return await self._request(
            "GET", 
            f"/markets/{ticker}/orderbook",
            params={"depth": depth}
        )
    
    async def get_market_history(
        self,
        ticker: str,
        limit: int = 100,
        cursor: str | None = None,
    ) -> dict:
        """Get trade history for a market."""
        params = {"limit": limit}
        if cursor:
            params["cursor"] = cursor
        return await self._request("GET", f"/markets/{ticker}/trades", params=params)
    
    async def get_series(self) -> dict:
        """Get all event series (categories like FED, WEATHER, etc.)."""
        return await self._request("GET", "/series")


# Singleton instance
_client: KalshiClient | None = None


def get_kalshi_client() -> KalshiClient:
    """Get or create Kalshi client instance."""
    global _client
    if _client is None:
        _client = KalshiClient()
    return _client
