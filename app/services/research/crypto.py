"""Crypto Research Module.

Provides probability estimates for:
- Price brackets (BTC above/below $X)
- ETF approvals
- Protocol events (halvings, upgrades)

Data sources:
- CoinGecko API (free tier)
- On-chain metrics (future: Glassnode)

Caching: In-memory TTL cache (2-5 min depending on endpoint)
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional
import httpx
from cachetools import TTLCache
import threading


@dataclass
class CryptoPrice:
    """Cryptocurrency price data."""
    symbol: str
    name: str
    price_usd: float
    change_24h: float  # Percentage
    change_7d: float
    market_cap: float
    volume_24h: float
    last_updated: datetime


@dataclass
class PriceForecast:
    """Price bracket probability."""
    symbol: str
    threshold: float
    direction: str  # "above" or "below"
    probability: float
    confidence: str
    factors: list[dict]


class CryptoModule:
    """Research module for cryptocurrency analysis."""
    
    COINGECKO_BASE = "https://api.coingecko.com/api/v3"
    
    # Major cryptocurrencies
    COINS = {
        "btc": {"id": "bitcoin", "name": "Bitcoin"},
        "eth": {"id": "ethereum", "name": "Ethereum"},
        "sol": {"id": "solana", "name": "Solana"},
        "xrp": {"id": "ripple", "name": "XRP"},
        "ada": {"id": "cardano", "name": "Cardano"},
        "doge": {"id": "dogecoin", "name": "Dogecoin"},
        "matic": {"id": "matic-network", "name": "Polygon"},
        "link": {"id": "chainlink", "name": "Chainlink"},
        "avax": {"id": "avalanche-2", "name": "Avalanche"},
    }
    
    # Cache TTLs (seconds)
    CACHE_TTL = {
        "price": 120,       # 2 min for individual prices
        "dashboard": 180,   # 3 min for dashboard
        "market": 300,      # 5 min for global market
        "history": 600,     # 10 min for historical data
    }
    
    def __init__(self):
        self._lock = threading.Lock()
        # Separate caches per data type for better TTL control
        self._price_cache = TTLCache(maxsize=100, ttl=self.CACHE_TTL["price"])
        self._dashboard_cache = TTLCache(maxsize=10, ttl=self.CACHE_TTL["dashboard"])
        self._market_cache = TTLCache(maxsize=10, ttl=self.CACHE_TTL["market"])
        self._history_cache = TTLCache(maxsize=50, ttl=self.CACHE_TTL["history"])
    
    async def _coingecko_request(self, endpoint: str, params: dict = None) -> dict:
        """Make request to CoinGecko API."""
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.COINGECKO_BASE}{endpoint}",
                params=params or {},
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_price(self, symbol: str) -> CryptoPrice:
        """Get current price for a cryptocurrency (cached 2 min)."""
        if symbol.lower() not in self.COINS:
            raise ValueError(f"Unknown coin: {symbol}. Available: {list(self.COINS.keys())}")
        
        coin_id = self.COINS[symbol.lower()]["id"]
        coin_name = self.COINS[symbol.lower()]["name"]
        
        # Check cache first
        cache_key = f"price:{coin_id}"
        with self._lock:
            if cache_key in self._price_cache:
                return self._price_cache[cache_key]
        
        data = await self._coingecko_request(
            f"/coins/{coin_id}",
            {"localization": "false", "tickers": "false", "community_data": "false"}
        )
        
        market_data = data.get("market_data", {})
        
        price = CryptoPrice(
            symbol=symbol.upper(),
            name=coin_name,
            price_usd=market_data.get("current_price", {}).get("usd", 0),
            change_24h=market_data.get("price_change_percentage_24h", 0),
            change_7d=market_data.get("price_change_percentage_7d", 0),
            market_cap=market_data.get("market_cap", {}).get("usd", 0),
            volume_24h=market_data.get("total_volume", {}).get("usd", 0),
            last_updated=datetime.now(),
        )
        
        with self._lock:
            self._price_cache[cache_key] = price
        return price
    
    async def get_prices(self, symbols: list[str] = None) -> list[CryptoPrice]:
        """Get prices for multiple cryptocurrencies."""
        if symbols is None:
            symbols = ["btc", "eth", "sol"]
        
        prices = []
        for symbol in symbols:
            try:
                price = await self.get_price(symbol)
                prices.append(price)
            except Exception as e:
                continue
        
        return prices
    
    async def get_market_overview(self) -> dict:
        """Get overall crypto market data (cached 5 min)."""
        cache_key = "market_overview"
        with self._lock:
            if cache_key in self._market_cache:
                return self._market_cache[cache_key]
        
        data = await self._coingecko_request("/global")
        market = data.get("data", {})
        
        result = {
            "total_market_cap": market.get("total_market_cap", {}).get("usd"),
            "total_volume_24h": market.get("total_volume", {}).get("usd"),
            "btc_dominance": market.get("market_cap_percentage", {}).get("btc"),
            "eth_dominance": market.get("market_cap_percentage", {}).get("eth"),
            "market_cap_change_24h": market.get("market_cap_change_percentage_24h_usd"),
            "active_cryptocurrencies": market.get("active_cryptocurrencies"),
        }
        
        with self._lock:
            self._market_cache[cache_key] = result
        return result
    
    async def get_price_history(
        self, 
        symbol: str, 
        days: int = 30,
    ) -> list[dict]:
        """Get historical price data (cached 10 min)."""
        if symbol.lower() not in self.COINS:
            raise ValueError(f"Unknown coin: {symbol}")
        
        coin_id = self.COINS[symbol.lower()]["id"]
        
        cache_key = f"history:{coin_id}:{days}"
        with self._lock:
            if cache_key in self._history_cache:
                return self._history_cache[cache_key]
        
        data = await self._coingecko_request(
            f"/coins/{coin_id}/market_chart",
            {"vs_currency": "usd", "days": days}
        )
        
        prices = data.get("prices", [])
        
        result = [
            {
                "timestamp": datetime.fromtimestamp(p[0] / 1000).isoformat(),
                "price": p[1],
            }
            for p in prices
        ]
        
        with self._lock:
            self._history_cache[cache_key] = result
        return result
    
    async def estimate_price_probability(
        self,
        symbol: str,
        threshold: float,
        direction: str,  # "above" or "below"
        days_ahead: int = 7,
    ) -> PriceForecast:
        """
        Estimate probability of price above/below threshold.
        
        Uses:
        1. Current price vs threshold
        2. Recent volatility
        3. Trend direction
        """
        price = await self.get_price(symbol)
        
        current = price.price_usd
        change_7d = price.change_7d
        
        factors = []
        
        # Distance to threshold
        distance_pct = (threshold - current) / current * 100
        
        # Base probability from distance
        if direction == "above":
            if current >= threshold:
                # Already above
                base_prob = 0.85
                factors.append({"name": "Current price", "value": f"Already above threshold", "impact": "positive"})
            else:
                # Need to rise
                # Use 7d change as trend indicator
                if change_7d > 0:
                    trend_factor = min(change_7d / 10, 0.3)  # Max 30% boost
                else:
                    trend_factor = max(change_7d / 10, -0.3)
                
                base_prob = max(0.1, 0.5 + trend_factor - (distance_pct / 50))
                factors.append({
                    "name": "Distance to threshold",
                    "value": f"{distance_pct:+.1f}%",
                    "impact": "negative" if distance_pct > 10 else "neutral",
                })
        else:  # below
            if current <= threshold:
                base_prob = 0.85
                factors.append({"name": "Current price", "value": f"Already below threshold", "impact": "positive"})
            else:
                if change_7d < 0:
                    trend_factor = min(abs(change_7d) / 10, 0.3)
                else:
                    trend_factor = max(-change_7d / 10, -0.3)
                
                base_prob = max(0.1, 0.5 + trend_factor + (distance_pct / 50))
                factors.append({
                    "name": "Distance to threshold",
                    "value": f"{-distance_pct:+.1f}%",
                    "impact": "negative" if distance_pct < -10 else "neutral",
                })
        
        # Add trend factor
        factors.append({
            "name": "7-day trend",
            "value": f"{change_7d:+.1f}%",
            "impact": "positive" if (direction == "above" and change_7d > 0) or (direction == "below" and change_7d < 0) else "negative",
        })
        
        # Adjust for volatility (crypto is volatile)
        volatility_factor = abs(price.change_24h) / 100
        
        # Confidence based on distance and volatility
        if abs(distance_pct) > 20:
            confidence = "low"
        elif abs(distance_pct) > 10:
            confidence = "medium"
        else:
            confidence = "high"
        
        # Clamp probability
        probability = max(0.05, min(0.95, base_prob))
        
        return PriceForecast(
            symbol=symbol.upper(),
            threshold=threshold,
            direction=direction,
            probability=round(probability, 2),
            confidence=confidence,
            factors=factors,
        )
    
    async def get_crypto_dashboard(self) -> dict:
        """Get comprehensive crypto dashboard (cached 3 min)."""
        cache_key = "dashboard"
        with self._lock:
            if cache_key in self._dashboard_cache:
                return self._dashboard_cache[cache_key]
        
        # Get top coins
        prices = await self.get_prices(["btc", "eth", "sol"])
        market = await self.get_market_overview()
        
        result = {
            "market_overview": {
                "total_market_cap": f"${market['total_market_cap']/1e12:.2f}T" if market.get('total_market_cap') else None,
                "btc_dominance": f"{market.get('btc_dominance', 0):.1f}%",
                "market_cap_change_24h": f"{market.get('market_cap_change_24h', 0):+.2f}%",
            },
            "prices": [
                {
                    "symbol": p.symbol,
                    "name": p.name,
                    "price": f"${p.price_usd:,.2f}",
                    "change_24h": f"{p.change_24h:+.2f}%",
                    "change_7d": f"{p.change_7d:+.2f}%",
                }
                for p in prices
            ],
            "available_coins": list(self.COINS.keys()),
            "cached_at": datetime.now().isoformat(),
        }
        
        with self._lock:
            self._dashboard_cache[cache_key] = result
        return result


# Singleton
_module: CryptoModule | None = None

def get_crypto_module() -> CryptoModule:
    global _module
    if _module is None:
        _module = CryptoModule()
    return _module
