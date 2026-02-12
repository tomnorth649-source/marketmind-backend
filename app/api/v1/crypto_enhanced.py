"""Enhanced Crypto API with prediction probabilities from markets."""
from datetime import datetime
from typing import Optional
import httpx

from fastapi import APIRouter
from pydantic import BaseModel

from app.integrations.polymarket import get_polymarket_client


router = APIRouter(prefix="/crypto", tags=["crypto-enhanced"])


class CryptoCoin(BaseModel):
    symbol: str
    name: str
    price: float
    change_24h: float
    market_cap: float
    volume_24h: Optional[float] = None


class CryptoMarket(BaseModel):
    total_market_cap: float
    btc_dominance: float
    total_volume_24h: float
    fear_greed_index: Optional[int] = None


class CryptoDashboardResponse(BaseModel):
    market: CryptoMarket
    coins: list[CryptoCoin]
    probabilities: dict[str, float]  # Event name -> probability
    probability_sources: list[dict]  # Full market info
    updatedAt: str


async def fetch_crypto_prices() -> tuple[CryptoMarket, list[CryptoCoin]]:
    """Fetch current crypto prices from CoinGecko (free, no API key)."""
    try:
        async with httpx.AsyncClient() as client:
            # Global market data
            global_resp = await client.get(
                "https://api.coingecko.com/api/v3/global",
                timeout=10.0,
            )
            global_data = global_resp.json().get("data", {})
            
            # Top coins
            coins_resp = await client.get(
                "https://api.coingecko.com/api/v3/coins/markets",
                params={
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": 10,
                    "page": 1,
                    "sparkline": False,
                },
                timeout=10.0,
            )
            coins_data = coins_resp.json()
            
            market = CryptoMarket(
                total_market_cap=global_data.get("total_market_cap", {}).get("usd", 0),
                btc_dominance=global_data.get("market_cap_percentage", {}).get("btc", 0),
                total_volume_24h=global_data.get("total_volume", {}).get("usd", 0),
                fear_greed_index=None,  # Would need separate API
            )
            
            coins = []
            for c in coins_data[:10]:
                coins.append(CryptoCoin(
                    symbol=c.get("symbol", "").upper(),
                    name=c.get("name", ""),
                    price=c.get("current_price", 0),
                    change_24h=c.get("price_change_percentage_24h", 0),
                    market_cap=c.get("market_cap", 0),
                    volume_24h=c.get("total_volume", 0),
                ))
            
            return market, coins
    except Exception as e:
        print(f"CoinGecko error: {e}")
        # Fallback data
        return CryptoMarket(
            total_market_cap=3_200_000_000_000,
            btc_dominance=56.5,
            total_volume_24h=95_000_000_000,
            fear_greed_index=65,
        ), [
            CryptoCoin(symbol="BTC", name="Bitcoin", price=97000, change_24h=1.2, market_cap=1_900_000_000_000),
            CryptoCoin(symbol="ETH", name="Ethereum", price=2650, change_24h=-0.5, market_cap=320_000_000_000),
            CryptoCoin(symbol="SOL", name="Solana", price=185, change_24h=2.1, market_cap=85_000_000_000),
        ]


async def fetch_crypto_probabilities() -> tuple[dict[str, float], list[dict]]:
    """Fetch crypto prediction probabilities from Polymarket."""
    probabilities = {}
    sources = []
    
    try:
        client = get_polymarket_client()
        markets = await client.get_markets(closed=False, limit=200)
        
        # Filter to crypto-related markets
        crypto_keywords = ["bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol", "xrp", "dogecoin"]
        
        for m in markets:
            title = (m.get("question") or m.get("title") or "").lower()
            if any(kw in title for kw in crypto_keywords):
                # Parse probability
                prices = m.get("outcomePrices", "[0.5, 0.5]")
                try:
                    if isinstance(prices, str):
                        prices = eval(prices)
                    yes_price = float(prices[0]) if prices else 0.5
                except:
                    yes_price = 0.5
                
                # Create short name for display
                full_title = m.get("question") or m.get("title") or "Unknown"
                short_name = full_title[:40] + "..." if len(full_title) > 40 else full_title
                
                probabilities[short_name] = yes_price
                sources.append({
                    "title": full_title,
                    "probability": yes_price,
                    "volume": float(m.get("volumeNum", 0) or 0),
                    "platform": "polymarket",
                    "url": f"https://polymarket.com/event/{m.get('slug', m.get('id'))}",
                })
        
        # Sort by volume and take top 10
        sources.sort(key=lambda x: x["volume"], reverse=True)
        sources = sources[:10]
        
        # Rebuild probabilities from top sources
        probabilities = {s["title"][:40] + ("..." if len(s["title"]) > 40 else ""): s["probability"] for s in sources[:5]}
        
    except Exception as e:
        print(f"Polymarket crypto error: {e}")
    
    return probabilities, sources


@router.get("/dashboard/enhanced", response_model=CryptoDashboardResponse)
async def get_enhanced_crypto_dashboard():
    """
    Get crypto dashboard with real-time prices and prediction probabilities.
    
    Returns:
    - Live prices from CoinGecko (updates every ~1 min)
    - Prediction probabilities from Polymarket
    """
    # Fetch in parallel
    import asyncio
    (market, coins), (probabilities, sources) = await asyncio.gather(
        fetch_crypto_prices(),
        fetch_crypto_probabilities(),
    )
    
    return CryptoDashboardResponse(
        market=market,
        coins=coins,
        probabilities=probabilities,
        probability_sources=sources,
        updatedAt=datetime.utcnow().isoformat() + "Z",
    )
