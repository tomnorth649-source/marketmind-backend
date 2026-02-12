"""Crypto Research API endpoints."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.research.crypto import get_crypto_module

router = APIRouter(prefix="/crypto", tags=["crypto-research"])


@router.get("/dashboard")
async def get_crypto_dashboard():
    """Get cryptocurrency market dashboard."""
    module = get_crypto_module()
    
    try:
        return await module.get_crypto_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/price/{symbol}")
async def get_crypto_price(symbol: str):
    """
    Get current price for a cryptocurrency.
    
    Symbols: btc, eth, sol, xrp, ada, doge, matic, link, avax
    """
    module = get_crypto_module()
    
    try:
        price = await module.get_price(symbol.lower())
        return {
            "symbol": price.symbol,
            "name": price.name,
            "price_usd": price.price_usd,
            "price_display": f"${price.price_usd:,.2f}",
            "change_24h": f"{price.change_24h:+.2f}%",
            "change_7d": f"{price.change_7d:+.2f}%",
            "market_cap": f"${price.market_cap/1e9:.2f}B",
            "volume_24h": f"${price.volume_24h/1e9:.2f}B",
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/market")
async def get_market_overview():
    """Get overall crypto market data."""
    module = get_crypto_module()
    
    try:
        return await module.get_market_overview()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history/{symbol}")
async def get_price_history(
    symbol: str,
    days: int = Query(default=30, ge=1, le=365),
):
    """Get historical price data for a cryptocurrency."""
    module = get_crypto_module()
    
    try:
        history = await module.get_price_history(symbol.lower(), days)
        return {
            "symbol": symbol.upper(),
            "days": days,
            "data": history,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability/price")
async def get_price_probability(
    symbol: str = Query(..., description="Coin symbol (btc, eth, sol, etc.)"),
    threshold: float = Query(..., description="Price threshold in USD"),
    direction: str = Query(..., description="'above' or 'below' threshold"),
    days: int = Query(default=7, ge=1, le=30, description="Days ahead"),
):
    """
    Estimate probability of price above/below threshold.
    
    Used for Kalshi crypto markets like:
    - "Will BTC exceed $100,000 this week?"
    - "Will ETH drop below $2,000?"
    """
    if direction not in ["above", "below"]:
        raise HTTPException(status_code=400, detail="direction must be 'above' or 'below'")
    
    module = get_crypto_module()
    
    try:
        forecast = await module.estimate_price_probability(
            symbol.lower(), threshold, direction, days
        )
        return {
            "symbol": forecast.symbol,
            "threshold": f"${forecast.threshold:,.0f}",
            "direction": forecast.direction,
            "probability": forecast.probability,
            "probability_display": f"{forecast.probability*100:.0f}%",
            "confidence": forecast.confidence,
            "factors": forecast.factors,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/coins")
async def list_coins():
    """List available cryptocurrencies."""
    module = get_crypto_module()
    return {
        "coins": [
            {"symbol": symbol.upper(), "name": info["name"]}
            for symbol, info in module.COINS.items()
        ]
    }
