"""Arbitrage Scanner API endpoints."""
from typing import Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.services.arb_scanner import get_arb_scanner, OpportunityTier

router = APIRouter(prefix="/arb", tags=["arbitrage"])


class MarketResponse(BaseModel):
    platform: str
    id: str
    title: str
    question: str
    yes_price: float
    no_price: float
    volume: float
    category: str


class OpportunityResponse(BaseModel):
    kalshi: MarketResponse
    polymarket: MarketResponse
    spread: float
    spread_pct: float
    spread_display: str  # "4.2¢" or "+4.2%"
    direction: str
    tier: str
    match_confidence: float
    explanation: str
    profit_example: str
    risks: list[str]


class ScanResponse(BaseModel):
    opportunities: list[OpportunityResponse]
    total: int
    hot_count: int
    warm_count: int
    cold_count: int
    scan_time_ms: int


class SummaryResponse(BaseModel):
    total_opportunities: int
    hot: int
    warm: int
    cold: int
    best_spread: float
    best_spread_display: str
    best_opportunity: Optional[str]


@router.get("/scan", response_model=ScanResponse)
async def scan_opportunities(
    min_spread: float = Query(default=0.02, ge=0, le=1, description="Minimum spread (0-1 scale, e.g., 0.02 = 2%)"),
    min_confidence: float = Query(default=0.5, ge=0, le=1, description="Minimum match confidence"),
    tier: Optional[str] = Query(default=None, description="Filter by tier: hot, warm, cold"),
    limit: int = Query(default=100, ge=10, le=500, description="Max markets to scan per platform"),
):
    """
    Scan for arbitrage opportunities between Kalshi and Polymarket.
    
    Returns matched markets with price discrepancies, sorted by spread (best first).
    
    **Tiers:**
    - 🔴 HOT: >5% spread, high liquidity
    - 🟡 WARM: 3-5% spread
    - ⚪ COLD: 1-3% spread
    """
    import time
    start = time.time()
    
    scanner = get_arb_scanner()
    opps = await scanner.scan(
        min_spread=min_spread,
        min_confidence=min_confidence,
        limit=limit,
    )
    
    # Filter by tier if specified
    if tier:
        opps = [o for o in opps if o.tier.value == tier.lower()]
    
    # Convert to response format
    results = []
    for opp in opps:
        results.append(OpportunityResponse(
            kalshi=MarketResponse(
                platform=opp.kalshi_market.platform,
                id=opp.kalshi_market.id,
                title=opp.kalshi_market.title,
                question=opp.kalshi_market.question,
                yes_price=opp.kalshi_market.yes_price,
                no_price=opp.kalshi_market.no_price,
                volume=opp.kalshi_market.volume,
                category=opp.kalshi_market.category,
            ),
            polymarket=MarketResponse(
                platform=opp.poly_market.platform,
                id=opp.poly_market.id,
                title=opp.poly_market.title,
                question=opp.poly_market.question,
                yes_price=opp.poly_market.yes_price,
                no_price=opp.poly_market.no_price,
                volume=opp.poly_market.volume,
                category=opp.poly_market.category,
            ),
            spread=opp.spread,
            spread_pct=opp.spread_pct,
            spread_display=f"{opp.spread*100:.1f}¢",
            direction=opp.direction,
            tier=opp.tier.value,
            match_confidence=opp.match_confidence,
            explanation=opp.explanation,
            profit_example=opp.profit_example,
            risks=opp.risks,
        ))
    
    elapsed_ms = int((time.time() - start) * 1000)
    
    return ScanResponse(
        opportunities=results,
        total=len(results),
        hot_count=len([o for o in opps if o.tier == OpportunityTier.HOT]),
        warm_count=len([o for o in opps if o.tier == OpportunityTier.WARM]),
        cold_count=len([o for o in opps if o.tier == OpportunityTier.COLD]),
        scan_time_ms=elapsed_ms,
    )


@router.get("/summary", response_model=SummaryResponse)
async def get_summary():
    """
    Quick summary of current arbitrage opportunities.
    
    Use this for dashboard widgets or quick checks.
    """
    scanner = get_arb_scanner()
    summary = await scanner.get_summary()
    
    return SummaryResponse(
        total_opportunities=summary["total_opportunities"],
        hot=summary["hot"],
        warm=summary["warm"],
        cold=summary["cold"],
        best_spread=summary["best_spread"],
        best_spread_display=f"{summary['best_spread']*100:.1f}¢" if summary["best_spread"] else "0¢",
        best_opportunity=summary["best_opportunity"],
    )


@router.get("/explain/{opportunity_id}")
async def explain_opportunity(opportunity_id: str):
    """
    Get detailed explanation of an arbitrage opportunity.
    
    Explains in plain English:
    - Why the prices differ
    - How to execute the trade
    - What the risks are
    """
    # TODO: Implement opportunity lookup by ID
    # For now, return example
    return {
        "id": opportunity_id,
        "what_is_arb": (
            "Arbitrage is when the same thing is priced differently in two places. "
            "You buy where it's cheap and sell where it's expensive, locking in guaranteed profit."
        ),
        "how_it_works": (
            "1. Buy YES shares on the cheaper platform\n"
            "2. Buy NO shares on the more expensive platform\n"
            "3. One of them MUST pay out $1\n"
            "4. Your total cost is less than $1, so you profit the difference"
        ),
        "example": (
            "Platform A: YES = 60¢, NO = 40¢ (total $1)\n"
            "Platform B: YES = 55¢, NO = 45¢ (total $1)\n\n"
            "You buy: YES on B (55¢) + NO on A (40¢) = 95¢ total\n"
            "Guaranteed payout: $1 (one of them wins)\n"
            "Profit: 5¢ per contract = 5.3% return"
        ),
        "risks": [
            "Resolution timing: Platforms may resolve at different times",
            "Definition differences: 'Rate cut' might be defined slightly differently",
            "Liquidity: You may not be able to execute at displayed prices",
            "Platform risk: Withdrawal issues, disputes, etc.",
        ],
    }
