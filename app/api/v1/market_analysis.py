"""
Market Analysis API - Deep analysis for individual markets.

Generates real analysis explaining why our model probability
might differ from market price.

Uses tiered LLM approach:
- Tier 1: Gemini Flash (free)
- Tier 2: OpenAI GPT-4o-mini (cheap)
- Tier 3: Claude (premium fallback)
"""
from datetime import datetime
from typing import Optional
import asyncio

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.integrations.polymarket import get_polymarket_client
from app.services.research_query import research_query, QueryCategory
from app.services.research.fedwatch import get_fedwatch_client
from app.services.llm_analysis import get_event_analysis, get_batch_analysis


router = APIRouter(prefix="/analysis", tags=["market-analysis"])


class MarketAnalysis(BaseModel):
    """Analysis result for a market."""
    market_id: str
    title: str
    market_probability: float  # What the market says
    model_probability: Optional[float]  # What our model says
    edge: Optional[float]  # Difference (model - market)
    edge_direction: str  # "underpriced", "overpriced", "fair"
    confidence: str  # "high", "medium", "low"
    analysis: str  # The actual analysis text
    factors: list[dict]  # Supporting factors
    category: str
    sources_used: int
    updated_at: str


def classify_market_category(title: str) -> QueryCategory:
    """Classify market into category."""
    title_lower = title.lower()
    
    if any(kw in title_lower for kw in ["fed", "rate", "fomc", "powell", "interest"]):
        return QueryCategory.FED
    elif any(kw in title_lower for kw in ["bitcoin", "btc", "eth", "crypto", "solana"]):
        return QueryCategory.CRYPTO
    elif any(kw in title_lower for kw in ["trump", "biden", "election", "president", "democrat", "republican", "congress"]):
        return QueryCategory.POLITICS
    elif any(kw in title_lower for kw in ["nba", "nfl", "mlb", "super bowl", "championship", "finals"]):
        return QueryCategory.SPORTS
    else:
        return QueryCategory.GENERAL


async def analyze_fed_market(title: str, market_prob: float) -> tuple[Optional[float], str, list[dict]]:
    """Analyze a Fed-related market using FedWatch data."""
    try:
        client = get_fedwatch_client()
        timeline = await client.get_timeline()
        
        if not timeline or not timeline.meetings:
            return None, "Unable to fetch FedWatch data for comparison.", []
        
        next_meeting = timeline.meetings[0]
        
        # Calculate cut/hold probabilities
        cut_prob = sum(
            prob for range_str, prob in next_meeting.probabilities.items()
            if float(range_str.split("-")[0]) < next_meeting.current_target_low
        )
        hold_prob = next_meeting.probabilities.get(
            f"{next_meeting.current_target_low}-{next_meeting.current_target_high}", 0
        )
        
        # Determine what the market is asking about
        title_lower = title.lower()
        if "cut" in title_lower:
            model_prob = cut_prob
            action = "cut"
        elif "hike" in title_lower or "raise" in title_lower:
            model_prob = 1 - cut_prob - hold_prob
            action = "hike"
        else:
            model_prob = hold_prob
            action = "hold"
        
        # Generate analysis
        diff = model_prob - market_prob
        if abs(diff) < 0.03:
            analysis = f"Market price ({market_prob*100:.0f}%) aligns with CME FedWatch ({model_prob*100:.0f}% {action} probability). No significant edge detected."
        elif diff > 0:
            analysis = f"CME FedWatch shows {model_prob*100:.0f}% {action} probability, but market prices it at {market_prob*100:.0f}%. This {abs(diff)*100:.1f}% gap suggests the market may be UNDERPRICING this outcome. Consider: Fed officials' recent statements, inflation trends, and employment data."
        else:
            analysis = f"Market prices {action} at {market_prob*100:.0f}%, but CME FedWatch only shows {model_prob*100:.0f}%. The market may be OVERPRICING this {abs(diff)*100:.1f}%. Check if there's news the futures market hasn't fully absorbed."
        
        factors = [
            {"name": "CME FedWatch Cut Prob", "value": f"{cut_prob*100:.1f}%", "signal": "data"},
            {"name": "CME FedWatch Hold Prob", "value": f"{hold_prob*100:.1f}%", "signal": "data"},
            {"name": "Next Meeting", "value": next_meeting.meeting_date, "signal": "info"},
            {"name": "Current Rate", "value": f"{next_meeting.current_target_low}-{next_meeting.current_target_high}%", "signal": "info"},
        ]
        
        return model_prob, analysis, factors
        
    except Exception as e:
        return None, f"FedWatch analysis unavailable: {e}", []


async def analyze_generic_market(title: str, market_prob: float, category: QueryCategory) -> tuple[Optional[float], str, list[dict]]:
    """Analyze a market using research query engine."""
    try:
        result = await research_query(query=title, tier="free")
        
        model_prob = result.probability
        factors = []
        
        if model_prob is None:
            analysis = f"Limited external data available for this market. The {market_prob*100:.0f}% price reflects pure market sentiment. Consider: How liquid is this market? Are there informed traders with edge?"
            return None, analysis, factors
        
        diff = model_prob - market_prob
        sources_count = len(result.sources)
        
        # Build factors from sources
        for s in result.sources[:4]:
            factors.append({
                "name": s.name[:40],
                "value": f"{s.probability*100:.0f}%" if s.probability else "N/A",
                "signal": s.type,
            })
        
        if abs(diff) < 0.03:
            analysis = f"Our model ({model_prob*100:.0f}%) agrees with the market ({market_prob*100:.0f}%). Based on {sources_count} sources, this appears to be fairly priced. Edge opportunity is minimal."
        elif diff > 0:
            analysis = f"Model probability: {model_prob*100:.0f}% vs Market: {market_prob*100:.0f}%. Our analysis of {sources_count} sources suggests this outcome is UNDERPRICED by {abs(diff)*100:.1f}%. {result.reasoning}"
        else:
            analysis = f"Model probability: {model_prob*100:.0f}% vs Market: {market_prob*100:.0f}%. Based on {sources_count} sources, this may be OVERPRICED by {abs(diff)*100:.1f}%. {result.reasoning}"
        
        return model_prob, analysis, factors
        
    except Exception as e:
        return None, f"Analysis unavailable: {e}", []


@router.get("/market/{market_id}", response_model=MarketAnalysis)
async def analyze_market(market_id: str):
    """
    Get deep analysis for a specific market.
    
    Compares market price against our model's probability
    and explains any edge or disagreement.
    """
    # Fetch market data
    poly_client = get_polymarket_client()
    
    try:
        markets = await poly_client.get_markets(closed=False, limit=500)
        market = next((m for m in markets if str(m.get("id")) == market_id), None)
        
        if not market:
            raise HTTPException(status_code=404, detail="Market not found")
        
        title = market.get("question") or market.get("title") or "Unknown"
        
        # Parse market probability
        prices = market.get("outcomePrices", "[0.5, 0.5]")
        try:
            if isinstance(prices, str):
                prices = eval(prices)
            market_prob = float(prices[0]) if prices else 0.5
        except:
            market_prob = 0.5
        
        # Classify and analyze
        category = classify_market_category(title)
        
        if category == QueryCategory.FED:
            model_prob, analysis, factors = await analyze_fed_market(title, market_prob)
        else:
            model_prob, analysis, factors = await analyze_generic_market(title, market_prob, category)
        
        # Calculate edge
        if model_prob is not None:
            edge = model_prob - market_prob
            if edge > 0.03:
                edge_direction = "underpriced"
            elif edge < -0.03:
                edge_direction = "overpriced"
            else:
                edge_direction = "fair"
            confidence = "high" if abs(edge) > 0.1 else "medium" if abs(edge) > 0.03 else "low"
        else:
            edge = None
            edge_direction = "unknown"
            confidence = "low"
        
        return MarketAnalysis(
            market_id=market_id,
            title=title,
            market_probability=market_prob,
            model_probability=model_prob,
            edge=edge,
            edge_direction=edge_direction,
            confidence=confidence,
            analysis=analysis,
            factors=factors,
            category=category.value,
            sources_used=len(factors),
            updated_at=datetime.utcnow().isoformat() + "Z",
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/batch")
async def analyze_batch(market_ids: str):
    """
    Analyze multiple markets at once.
    
    Pass comma-separated market IDs.
    """
    ids = [mid.strip() for mid in market_ids.split(",") if mid.strip()][:10]  # Max 10
    
    results = []
    for market_id in ids:
        try:
            analysis = await analyze_market(market_id)
            results.append(analysis.model_dump())
        except HTTPException:
            results.append({"market_id": market_id, "error": "Not found"})
        except Exception as e:
            results.append({"market_id": market_id, "error": str(e)})
    
    return {"analyses": results, "count": len(results)}


# ─── LLM-Powered Deep Analysis ──────────────────────────────────────────────

class DeepAnalysis(BaseModel):
    """Enhanced LLM-powered analysis."""
    market_id: str
    title: str
    thesis: str
    market_probability: float
    model_probability: Optional[float]
    edge: Optional[float]
    edge_direction: str
    confidence: str
    bull_case: list[str]
    bear_case: list[str]
    key_data: list[str]
    catalysts: list[str]
    risk_factors: list[str]
    provider: str
    from_cache: bool = False
    generated_at: str


@router.get("/deep/{market_id}", response_model=DeepAnalysis)
async def deep_analyze_market(
    market_id: str,
    refresh: bool = Query(default=False, description="Force refresh (bypass cache)")
):
    """
    Get LLM-powered deep analysis for a market.
    
    Uses tiered approach:
    1. Gemini Flash (free - 1500 req/day)
    2. OpenAI GPT-4o-mini (cheap fallback)
    3. Claude (premium fallback)
    
    Results are cached for 24h unless refresh=true.
    """
    # Fetch market data
    poly_client = get_polymarket_client()
    
    try:
        markets = await poly_client.get_markets(closed=False, limit=500)
        market = next((m for m in markets if str(m.get("id")) == market_id), None)
        
        if not market:
            raise HTTPException(status_code=404, detail="Market not found")
        
        # Parse market data
        title = market.get("question") or market.get("title") or "Unknown"
        
        prices = market.get("outcomePrices", "[0.5, 0.5]")
        try:
            if isinstance(prices, str):
                prices = eval(prices)
            market_prob = float(prices[0]) if prices else 0.5
        except:
            market_prob = 0.5
        
        # Build event dict for LLM
        event = {
            "id": market_id,
            "title": title,
            "question": market.get("question", title),
            "category": classify_market_category(title).value,
            "probability": market_prob,
            "volume_24h": float(market.get("volume24hr", 0) or 0),
            "closes_at": market.get("endDate", "Unknown"),
        }
        
        # Get LLM analysis
        result = await get_event_analysis(event, force_refresh=refresh)
        
        return DeepAnalysis(
            market_id=market_id,
            title=title,
            thesis=result.get("thesis", "Analysis unavailable"),
            market_probability=result.get("market_probability", market_prob),
            model_probability=result.get("model_probability"),
            edge=result.get("edge"),
            edge_direction=result.get("edge_direction", "unknown"),
            confidence=result.get("confidence", "low"),
            bull_case=result.get("bull_case", []),
            bear_case=result.get("bear_case", []),
            key_data=result.get("key_data", []),
            catalysts=result.get("catalysts", []),
            risk_factors=result.get("risk_factors", []),
            provider=result.get("provider", "unknown"),
            from_cache=result.get("from_cache", False),
            generated_at=result.get("generated_at", datetime.utcnow().isoformat() + "Z"),
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/deep/batch")
async def deep_analyze_batch(
    market_ids: str,
    refresh: bool = Query(default=False, description="Force refresh all")
):
    """
    Get LLM-powered deep analysis for multiple markets.
    
    Pass comma-separated market IDs. Max 5 at a time to avoid rate limits.
    """
    ids = [mid.strip() for mid in market_ids.split(",") if mid.strip()][:5]  # Max 5
    
    # Fetch all markets
    poly_client = get_polymarket_client()
    markets = await poly_client.get_markets(closed=False, limit=500)
    
    # Build event list
    events = []
    for market_id in ids:
        market = next((m for m in markets if str(m.get("id")) == market_id), None)
        if market:
            title = market.get("question") or market.get("title") or "Unknown"
            prices = market.get("outcomePrices", "[0.5, 0.5]")
            try:
                if isinstance(prices, str):
                    prices = eval(prices)
                market_prob = float(prices[0]) if prices else 0.5
            except:
                market_prob = 0.5
            
            events.append({
                "id": market_id,
                "title": title,
                "question": market.get("question", title),
                "category": classify_market_category(title).value,
                "probability": market_prob,
                "volume_24h": float(market.get("volume24hr", 0) or 0),
                "closes_at": market.get("endDate", "Unknown"),
            })
    
    # Get batch analysis
    results = await get_batch_analysis(events)
    
    return {
        "analyses": results,
        "count": len(results),
        "providers_used": list(set(r.get("provider", "unknown") for r in results)),
    }
