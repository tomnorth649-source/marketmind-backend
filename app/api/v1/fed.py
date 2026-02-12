"""Federal Reserve Research API endpoints."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from app.config import get_settings
from app.services.research.fed import get_fed_module

router = APIRouter(prefix="/fed", tags=["fed-research"])


class RateProbsResponse(BaseModel):
    meeting_date: str
    cut_50_plus: str
    cut_25: str
    hold: str
    hike_25: str
    hike_50_plus: str
    most_likely: str
    most_likely_prob: str
    confidence: str
    factors: list[dict]


class DashboardResponse(BaseModel):
    current_rate: dict
    next_meeting: Optional[str]
    probability_distribution: dict
    prediction: dict
    factors: list[dict]
    updated_at: str


@router.get("/dashboard", response_model=DashboardResponse)
async def get_fed_dashboard():
    """
    Get comprehensive Fed analysis dashboard.
    
    Includes:
    - Current Fed Funds rate
    - Next FOMC meeting date
    - Probability distribution for rate decision
    - Key economic factors and their signals
    
    **Note:** Requires FRED_API_KEY in environment.
    Get a free key at https://fred.stlouisfed.org/docs/api/api_key.html
    """
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(
            status_code=503,
            detail="FRED API key not configured. Add FRED_API_KEY to .env"
        )
    
    module = get_fed_module(fred_key)
    
    try:
        dashboard = await module.get_fed_dashboard()
        return dashboard
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/rate-probabilities", response_model=RateProbsResponse)
async def get_rate_probabilities(
    meeting_date: Optional[str] = Query(
        default=None,
        description="FOMC meeting date (YYYY-MM-DD). Defaults to next meeting."
    ),
):
    """
    Get probability distribution for Fed rate decision at a specific meeting.
    
    Returns probabilities for:
    - Cut 50+ bps
    - Cut 25 bps
    - Hold (no change)
    - Hike 25 bps
    - Hike 50+ bps
    
    Plus explanatory factors and confidence level.
    """
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(
            status_code=503,
            detail="FRED API key not configured"
        )
    
    module = get_fed_module(fred_key)
    
    try:
        probs = await module.calculate_rate_probabilities(meeting_date)
        most_likely, most_likely_prob = probs.most_likely
        
        return RateProbsResponse(
            meeting_date=probs.meeting_date,
            cut_50_plus=f"{probs.cut_50*100:.1f}%",
            cut_25=f"{probs.cut_25*100:.1f}%",
            hold=f"{probs.hold*100:.1f}%",
            hike_25=f"{probs.hike_25*100:.1f}%",
            hike_50_plus=f"{probs.hike_50*100:.1f}%",
            most_likely=most_likely.value,
            most_likely_prob=f"{most_likely_prob*100:.1f}%",
            confidence=probs.confidence,
            factors=probs.factors,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/indicators")
async def get_economic_indicators():
    """
    Get current values of key economic indicators.
    
    Indicators:
    - Fed Funds Rate
    - Yield Curve (10Y-2Y spread)
    - Unemployment Rate
    - Core PCE Inflation
    - Financial Conditions Index
    """
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(
            status_code=503,
            detail="FRED API key not configured"
        )
    
    module = get_fed_module(fred_key)
    
    try:
        indicators = {}
        
        rate = await module.get_current_rate()
        if rate:
            indicators["fed_funds_rate"] = {
                "value": rate.value,
                "date": rate.date,
                "unit": "%",
            }
        
        curve = await module.get_yield_curve()
        if curve:
            indicators["yield_curve"] = {
                "value": curve.value,
                "date": curve.date,
                "signal": curve.signal,
                "interpretation": "Negative = inverted (recession risk)",
            }
        
        unemp = await module.get_unemployment()
        if unemp:
            indicators["unemployment"] = {
                "value": unemp.value,
                "date": unemp.date,
                "change": unemp.change,
                "signal": unemp.signal,
                "unit": "%",
            }
        
        pce = await module.get_core_pce()
        if pce:
            indicators["core_pce_inflation"] = {
                "value": pce.value,
                "date": pce.date,
                "signal": pce.signal,
                "target": "2.0%",
                "unit": "% YoY",
            }
        
        nfci = await module.get_financial_conditions()
        if nfci:
            indicators["financial_conditions"] = {
                "value": nfci.value,
                "date": nfci.date,
                "signal": nfci.signal,
                "interpretation": "Positive = tight, Negative = loose",
            }
        
        return {
            "indicators": indicators,
            "count": len(indicators),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/fomc-calendar")
async def get_fomc_calendar():
    """Get upcoming FOMC meeting dates."""
    module = get_fed_module()
    
    next_meeting = module.get_next_fomc_meeting()
    
    return {
        "next_meeting": next_meeting,
        "all_2026_meetings": module.FOMC_DATES_2026,
    }
