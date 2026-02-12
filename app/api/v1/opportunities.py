"""Opportunities API - The heart of MarketMind.

Surfaces top actionable prediction market opportunities across all categories
in a digestible, interactive format.
"""
import asyncio
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel

from app.integrations.polymarket import get_polymarket_client


router = APIRouter(prefix="/opportunities", tags=["opportunities"])


class MarketOdds(BaseModel):
    """Human-readable market odds."""
    yes_pct: float  # e.g., 72.5
    no_pct: float  # e.g., 27.5
    yes_display: str  # "73% YES"
    no_display: str  # "27% NO"


class Opportunity(BaseModel):
    """A single tradeable opportunity."""
    id: str
    platform: str
    title: str
    question: str
    category: str
    odds: MarketOdds
    volume_24h: float
    volume_total: float
    liquidity: float
    liquidity_display: str  # "High", "Medium", "Low"
    closes_at: Optional[str]
    time_remaining: Optional[str]  # "3d 5h" or "2 weeks"
    url: str
    is_hot: bool  # High volume + close deadline


class CategorySummary(BaseModel):
    """Summary for a category tab."""
    category: str
    display_name: str
    icon: str
    market_count: int
    top_opportunities: list[Opportunity]


class DashboardResponse(BaseModel):
    """Main dashboard data."""
    categories: list[CategorySummary]
    featured: list[Opportunity]
    total_markets: int
    updated_at: str


# Category configuration
CATEGORIES = {
    "crypto": {"name": "Crypto", "icon": "₿", "tags": ["crypto", "bitcoin", "ethereum"]},
    "fed": {"name": "Federal Reserve", "icon": "🏛️", "tags": ["fed", "fomc", "interest-rates"]},
    "politics": {"name": "Politics", "icon": "🗳️", "tags": ["politics", "elections", "congress"]},
    "weather": {"name": "Weather", "icon": "🌤️", "tags": ["weather", "climate"]},
    "sports": {"name": "Sports", "icon": "⚽", "tags": ["sports", "nfl", "nba", "soccer"]},
    "ai": {"name": "AI & Tech", "icon": "🤖", "tags": ["ai", "tech", "technology"]},
}


def format_volume(vol: float) -> str:
    """Format volume for display."""
    if vol >= 1_000_000:
        return f"${vol/1_000_000:.1f}M"
    elif vol >= 1_000:
        return f"${vol/1_000:.0f}K"
    else:
        return f"${vol:.0f}"


def format_liquidity(liq: float) -> str:
    """Categorize liquidity."""
    if liq >= 50_000:
        return "High"
    elif liq >= 10_000:
        return "Medium"
    else:
        return "Low"


def format_time_remaining(end_date: str) -> Optional[str]:
    """Format time until market closes."""
    if not end_date:
        return None
    try:
        end = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
        now = datetime.now(end.tzinfo)
        delta = end - now
        
        if delta.total_seconds() < 0:
            return "Closed"
        
        days = delta.days
        hours = delta.seconds // 3600
        
        if days > 30:
            return f"{days // 30}mo"
        elif days > 0:
            return f"{days}d {hours}h"
        elif hours > 0:
            return f"{hours}h"
        else:
            minutes = delta.seconds // 60
            return f"{minutes}m"
    except Exception:
        return None


def parse_market_to_opportunity(market: dict, category: str = "other") -> Opportunity:
    """Convert Polymarket market to Opportunity."""
    # Parse prices
    prices_str = market.get("outcomePrices", "[0.5, 0.5]")
    try:
        if isinstance(prices_str, str):
            prices = eval(prices_str)  # e.g., '["0.72", "0.28"]'
        else:
            prices = prices_str
        yes_price = float(prices[0]) if prices else 0.5
        no_price = float(prices[1]) if len(prices) > 1 else 1 - yes_price
    except Exception:
        yes_price = 0.5
        no_price = 0.5
    
    yes_pct = round(yes_price * 100, 1)
    no_pct = round(no_price * 100, 1)
    
    # Get volumes
    vol_24h = float(market.get("volume24hr", 0) or 0)
    vol_total = float(market.get("volumeNum", 0) or market.get("volume", 0) or 0)
    liquidity = float(market.get("liquidityNum", 0) or market.get("liquidity", 0) or 0)
    
    # Time remaining
    end_date = market.get("endDate")
    time_remaining = format_time_remaining(end_date)
    
    # Is it hot? High activity + soon to close
    is_hot = vol_24h > 50_000 or (time_remaining and "h" in time_remaining and liquidity > 10_000)
    
    # Build URL
    slug = market.get("slug", market.get("id", ""))
    url = f"https://polymarket.com/event/{slug}" if slug else "https://polymarket.com"
    
    return Opportunity(
        id=str(market.get("id", "")),
        platform="polymarket",
        title=market.get("question", market.get("title", "Unknown")),
        question=market.get("question", ""),
        category=category,
        odds=MarketOdds(
            yes_pct=yes_pct,
            no_pct=no_pct,
            yes_display=f"{yes_pct:.0f}% YES",
            no_display=f"{no_pct:.0f}% NO",
        ),
        volume_24h=vol_24h,
        volume_total=vol_total,
        liquidity=liquidity,
        liquidity_display=format_liquidity(liquidity),
        closes_at=end_date,
        time_remaining=time_remaining,
        url=url,
        is_hot=is_hot,
    )


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard(
    limit_per_category: int = Query(default=5, ge=1, le=20),
):
    """
    Get the main dashboard with top opportunities by category.
    
    This is the primary endpoint for the MarketMind UI.
    Returns categorized opportunities with digestible odds.
    """
    client = get_polymarket_client()
    
    # Fetch markets from Polymarket (most liquid/active)
    try:
        markets = await client.get_markets(closed=False, limit=200)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch markets: {e}")
    
    # Sort by volume for featured
    sorted_by_volume = sorted(
        markets,
        key=lambda m: float(m.get("volumeNum", 0) or m.get("volume", 0) or 0),
        reverse=True
    )
    
    # Categorize markets
    categorized: dict[str, list[dict]] = {cat: [] for cat in CATEGORIES}
    categorized["other"] = []
    
    for market in markets:
        question = (market.get("question") or "").lower()
        title = (market.get("title") or "").lower()
        text = question + " " + title
        
        placed = False
        for cat_key, cat_info in CATEGORIES.items():
            for tag in cat_info["tags"]:
                if tag in text:
                    categorized[cat_key].append(market)
                    placed = True
                    break
            if placed:
                break
        
        if not placed:
            categorized["other"].append(market)
    
    # Build category summaries
    categories = []
    for cat_key, cat_info in CATEGORIES.items():
        cat_markets = categorized[cat_key]
        
        # Sort by volume within category
        cat_markets_sorted = sorted(
            cat_markets,
            key=lambda m: float(m.get("volumeNum", 0) or m.get("volume", 0) or 0),
            reverse=True
        )[:limit_per_category]
        
        opportunities = [
            parse_market_to_opportunity(m, cat_key)
            for m in cat_markets_sorted
        ]
        
        categories.append(CategorySummary(
            category=cat_key,
            display_name=cat_info["name"],
            icon=cat_info["icon"],
            market_count=len(cat_markets),
            top_opportunities=opportunities,
        ))
    
    # Featured: Top 10 by volume overall
    featured = [
        parse_market_to_opportunity(m, "featured")
        for m in sorted_by_volume[:10]
    ]
    
    return DashboardResponse(
        categories=categories,
        featured=featured,
        total_markets=len(markets),
        updated_at=datetime.utcnow().isoformat() + "Z",
    )


@router.get("/category/{category}", response_model=CategorySummary)
async def get_category(
    category: str,
    limit: int = Query(default=20, ge=1, le=50),
):
    """Get markets for a specific category."""
    if category not in CATEGORIES:
        raise HTTPException(status_code=404, detail=f"Unknown category: {category}")
    
    cat_info = CATEGORIES[category]
    client = get_polymarket_client()
    
    try:
        markets = await client.get_markets(closed=False, limit=200)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch markets: {e}")
    
    # Filter by category keywords
    category_markets = []
    for market in markets:
        question = (market.get("question") or "").lower()
        title = (market.get("title") or "").lower()
        text = question + " " + title
        
        for tag in cat_info["tags"]:
            if tag in text:
                category_markets.append(market)
                break
    
    # Sort by volume
    category_markets = sorted(
        category_markets,
        key=lambda m: float(m.get("volumeNum", 0) or m.get("volume", 0) or 0),
        reverse=True
    )[:limit]
    
    opportunities = [
        parse_market_to_opportunity(m, category)
        for m in category_markets
    ]
    
    return CategorySummary(
        category=category,
        display_name=cat_info["name"],
        icon=cat_info["icon"],
        market_count=len(category_markets),
        top_opportunities=opportunities,
    )


@router.get("/market/{market_id}")
async def get_market_detail(market_id: str):
    """
    Get detailed info for a single market.
    
    Includes:
    - Current odds with context
    - Historical price data (if available)
    - Research summary
    - Similar markets
    """
    client = get_polymarket_client()
    
    try:
        markets = await client.get_markets(closed=False, limit=500)
        market = next((m for m in markets if str(m.get("id")) == market_id), None)
        
        if not market:
            raise HTTPException(status_code=404, detail="Market not found")
        
        opp = parse_market_to_opportunity(market)
        
        return {
            "market": opp,
            "raw_data": {
                "description": market.get("description", ""),
                "resolution_source": market.get("resolutionSource", ""),
            },
            "analysis": {
                "confidence_indicator": "Based on liquidity and volume",
                "price_stability": "Stable" if opp.liquidity > 50_000 else "Volatile",
                "suggested_action": "Monitor" if opp.odds.yes_pct > 40 and opp.odds.yes_pct < 60 else "Research more",
            },
            "links": {
                "polymarket": opp.url,
                "resolution": market.get("resolutionSource", ""),
            },
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/hot")
async def get_hot_opportunities(limit: int = Query(default=10, ge=1, le=50)):
    """
    Get the hottest opportunities right now.
    
    Hot = High 24h volume + closing soon + good liquidity.
    """
    client = get_polymarket_client()
    
    try:
        markets = await client.get_markets(closed=False, limit=200)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch markets: {e}")
    
    # Convert to opportunities
    opportunities = [parse_market_to_opportunity(m) for m in markets]
    
    # Filter to hot ones
    hot = [o for o in opportunities if o.is_hot]
    
    # If not enough hot, include high volume ones
    if len(hot) < limit:
        by_volume = sorted(opportunities, key=lambda o: o.volume_24h, reverse=True)
        for opp in by_volume:
            if opp not in hot:
                hot.append(opp)
            if len(hot) >= limit:
                break
    
    return {
        "opportunities": hot[:limit],
        "count": len(hot[:limit]),
        "updated_at": datetime.utcnow().isoformat() + "Z",
    }


@router.get("/search")
async def search_opportunities(
    q: str = Query(..., min_length=2, description="Search query"),
    limit: int = Query(default=20, ge=1, le=50),
):
    """Search markets by keyword."""
    client = get_polymarket_client()
    
    try:
        markets = await client.get_markets(closed=False, limit=500)
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Failed to fetch markets: {e}")
    
    q_lower = q.lower()
    
    matching = []
    for market in markets:
        question = (market.get("question") or "").lower()
        title = (market.get("title") or "").lower()
        desc = (market.get("description") or "").lower()
        
        if q_lower in question or q_lower in title or q_lower in desc:
            matching.append(market)
    
    # Sort by relevance (exact match in question > title > description)
    def relevance_score(m):
        q = (m.get("question") or "").lower()
        t = (m.get("title") or "").lower()
        score = 0
        if q_lower in q:
            score += 100
        if q_lower in t:
            score += 50
        score += float(m.get("volumeNum", 0) or 0) / 1000000
        return score
    
    matching = sorted(matching, key=relevance_score, reverse=True)[:limit]
    
    opportunities = [parse_market_to_opportunity(m) for m in matching]
    
    return {
        "query": q,
        "opportunities": opportunities,
        "count": len(opportunities),
    }
