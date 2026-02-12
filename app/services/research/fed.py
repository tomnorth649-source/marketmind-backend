"""Fed / Monetary Policy Research Module.

Provides probability estimates for:
- Fed rate decisions (cut/hold/hike)
- Fed Chair nominations
- Emergency meetings
- Balance sheet actions

Data sources:
- FRED API (Federal Reserve Economic Data)
- CME FedWatch implied probabilities
- Fed speech sentiment analysis
- Yield curve signals
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import httpx


class FedAction(str, Enum):
    CUT_50 = "cut_50"      # Cut by 50+ bps
    CUT_25 = "cut_25"      # Cut by 25 bps
    HOLD = "hold"          # No change
    HIKE_25 = "hike_25"    # Hike by 25 bps
    HIKE_50 = "hike_50"    # Hike by 50+ bps


@dataclass
class FedRateProbs:
    """Probability distribution for Fed rate decision."""
    meeting_date: str
    cut_50: float
    cut_25: float
    hold: float
    hike_25: float
    hike_50: float
    confidence: str  # "high", "medium", "low"
    factors: list[dict]  # Explanatory factors
    updated_at: datetime
    
    @property
    def most_likely(self) -> tuple[FedAction, float]:
        """Return most likely action and its probability."""
        probs = {
            FedAction.CUT_50: self.cut_50,
            FedAction.CUT_25: self.cut_25,
            FedAction.HOLD: self.hold,
            FedAction.HIKE_25: self.hike_25,
            FedAction.HIKE_50: self.hike_50,
        }
        action = max(probs, key=probs.get)
        return action, probs[action]


@dataclass
class EconomicIndicator:
    """An economic data point from FRED."""
    series_id: str
    name: str
    value: float
    date: str
    previous: Optional[float] = None
    change: Optional[float] = None
    signal: Optional[str] = None  # "hawkish", "dovish", "neutral"


class FedResearchModule:
    """Research module for Federal Reserve monetary policy analysis."""
    
    # FRED API configuration
    FRED_BASE_URL = "https://api.stlouisfed.org/fred"
    
    # Key FRED series for Fed analysis
    FRED_SERIES = {
        "FEDFUNDS": "Federal Funds Rate",
        "DFF": "Daily Fed Funds Rate",
        "T10Y2Y": "10Y-2Y Treasury Spread (Yield Curve)",
        "T10Y3M": "10Y-3M Treasury Spread",
        "UNRATE": "Unemployment Rate",
        "CPIAUCSL": "Consumer Price Index",
        "PCE": "Personal Consumption Expenditures",
        "PCEPILFE": "Core PCE (Fed's preferred inflation)",
        "NFCI": "Chicago Fed Financial Conditions Index",
        "UMCSENT": "Consumer Sentiment",
    }
    
    # FOMC meeting dates (update periodically)
    FOMC_DATES_2026 = [
        "2026-01-28", "2026-01-29",  # January
        "2026-03-17", "2026-03-18",  # March
        "2026-05-05", "2026-05-06",  # May
        "2026-06-16", "2026-06-17",  # June
        "2026-07-28", "2026-07-29",  # July
        "2026-09-15", "2026-09-16",  # September
        "2026-11-03", "2026-11-04",  # November
        "2026-12-15", "2026-12-16",  # December
    ]
    
    def __init__(self, fred_api_key: str = None):
        self.fred_api_key = fred_api_key
        self._cache: dict = {}
        self._cache_ttl = timedelta(hours=1)
    
    async def _fred_request(self, endpoint: str, params: dict) -> dict:
        """Make request to FRED API."""
        if not self.fred_api_key:
            raise ValueError("FRED API key required. Get one free at https://fred.stlouisfed.org/docs/api/api_key.html")
        
        params["api_key"] = self.fred_api_key
        params["file_type"] = "json"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.FRED_BASE_URL}/{endpoint}",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_series(self, series_id: str, limit: int = 10) -> list[dict]:
        """Get recent observations for a FRED series."""
        cache_key = f"fred:{series_id}"
        if cache_key in self._cache:
            cached, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return cached
        
        data = await self._fred_request("series/observations", {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": limit,
        })
        
        observations = data.get("observations", [])
        self._cache[cache_key] = (observations, datetime.now())
        return observations
    
    async def get_current_rate(self) -> EconomicIndicator:
        """Get current Fed Funds rate."""
        obs = await self.get_series("DFF", limit=2)
        if not obs:
            return None
        
        current = float(obs[0]["value"])
        previous = float(obs[1]["value"]) if len(obs) > 1 else None
        
        return EconomicIndicator(
            series_id="DFF",
            name="Fed Funds Rate",
            value=current,
            date=obs[0]["date"],
            previous=previous,
            change=current - previous if previous else None,
        )
    
    async def get_yield_curve(self) -> EconomicIndicator:
        """Get yield curve spread (10Y - 2Y)."""
        obs = await self.get_series("T10Y2Y", limit=5)
        if not obs:
            return None
        
        # Find most recent non-missing value
        for o in obs:
            if o["value"] != ".":
                current = float(o["value"])
                
                # Interpret signal
                if current < 0:
                    signal = "dovish"  # Inverted = recession risk = cuts coming
                elif current < 0.5:
                    signal = "neutral"
                else:
                    signal = "hawkish"  # Steep curve = growth = rates stay/rise
                
                return EconomicIndicator(
                    series_id="T10Y2Y",
                    name="Yield Curve (10Y-2Y)",
                    value=current,
                    date=o["date"],
                    signal=signal,
                )
        return None
    
    async def get_unemployment(self) -> EconomicIndicator:
        """Get unemployment rate."""
        obs = await self.get_series("UNRATE", limit=2)
        if not obs:
            return None
        
        current = float(obs[0]["value"])
        previous = float(obs[1]["value"]) if len(obs) > 1 else None
        change = current - previous if previous else None
        
        # Interpret signal
        if change and change > 0.3:
            signal = "dovish"  # Rising unemployment = cuts
        elif change and change < -0.2:
            signal = "hawkish"  # Falling unemployment = tight labor = rates stay
        else:
            signal = "neutral"
        
        return EconomicIndicator(
            series_id="UNRATE",
            name="Unemployment Rate",
            value=current,
            date=obs[0]["date"],
            previous=previous,
            change=change,
            signal=signal,
        )
    
    async def get_core_pce(self) -> EconomicIndicator:
        """Get Core PCE inflation (Fed's preferred measure)."""
        obs = await self.get_series("PCEPILFE", limit=13)  # 13 months for YoY
        if not obs or len(obs) < 13:
            return None
        
        # Calculate YoY change
        current = float(obs[0]["value"])
        year_ago = float(obs[12]["value"])
        yoy_change = ((current / year_ago) - 1) * 100
        
        # Interpret signal
        if yoy_change > 3.0:
            signal = "hawkish"  # High inflation = rates stay/rise
        elif yoy_change < 2.0:
            signal = "dovish"  # Low inflation = room to cut
        else:
            signal = "neutral"
        
        return EconomicIndicator(
            series_id="PCEPILFE",
            name="Core PCE Inflation (YoY)",
            value=round(yoy_change, 2),
            date=obs[0]["date"],
            signal=signal,
        )
    
    async def get_financial_conditions(self) -> EconomicIndicator:
        """Get Chicago Fed Financial Conditions Index."""
        obs = await self.get_series("NFCI", limit=2)
        if not obs:
            return None
        
        for o in obs:
            if o["value"] != ".":
                current = float(o["value"])
                
                # NFCI: positive = tight conditions, negative = loose
                if current > 0.5:
                    signal = "dovish"  # Tight conditions = may need cuts
                elif current < -0.5:
                    signal = "hawkish"  # Loose conditions = no urgency to cut
                else:
                    signal = "neutral"
                
                return EconomicIndicator(
                    series_id="NFCI",
                    name="Financial Conditions Index",
                    value=current,
                    date=o["date"],
                    signal=signal,
                )
        return None
    
    async def get_initial_claims(self) -> EconomicIndicator:
        """Get initial jobless claims (weekly) - KEY DOVISH SIGNAL."""
        obs = await self.get_series("ICSA", limit=4)  # Last 4 weeks
        if not obs:
            return None
        
        for o in obs:
            if o["value"] != ".":
                current = float(o["value"]) / 1000  # Convert to thousands
                
                # Claims rising = labor market weakening = dovish
                if current > 250:
                    signal = "dovish"  # Elevated claims
                elif current > 220:
                    signal = "neutral"
                else:
                    signal = "hawkish"  # Very low claims = tight labor
                
                return EconomicIndicator(
                    series_id="ICSA",
                    name="Initial Jobless Claims (K)",
                    value=round(current, 1),
                    date=o["date"],
                    signal=signal,
                )
        return None
    
    async def get_credit_spreads(self) -> EconomicIndicator:
        """Get high yield credit spread (OAS) - STRESS SIGNAL."""
        obs = await self.get_series("BAMLH0A0HYM2", limit=5)  # ICE BofA HY OAS
        if not obs:
            return None
        
        for o in obs:
            if o["value"] != ".":
                current = float(o["value"])
                
                # Wide spreads = stress = Fed may cut
                if current > 500:
                    signal = "dovish"  # High stress
                elif current > 400:
                    signal = "neutral"
                else:
                    signal = "hawkish"  # Tight spreads = risk-on
                
                return EconomicIndicator(
                    series_id="BAMLH0A0HYM2",
                    name="HY Credit Spread (bps)",
                    value=round(current, 0),
                    date=o["date"],
                    signal=signal,
                )
        return None
    
    async def get_real_rates(self) -> EconomicIndicator:
        """Get 10Y real interest rate (TIPS) - POLICY STANCE."""
        obs = await self.get_series("DFII10", limit=5)
        if not obs:
            return None
        
        for o in obs:
            if o["value"] != ".":
                current = float(o["value"])
                
                # High real rates = tight policy = room to cut
                if current > 2.0:
                    signal = "dovish"  # Very restrictive
                elif current > 1.0:
                    signal = "neutral"
                else:
                    signal = "hawkish"  # Low real rates
                
                return EconomicIndicator(
                    series_id="DFII10",
                    name="10Y Real Rate (%)",
                    value=round(current, 2),
                    date=o["date"],
                    signal=signal,
                )
        return None
    
    def get_next_fomc_meeting(self) -> Optional[str]:
        """Get date of next FOMC meeting."""
        today = datetime.now().date()
        for date_str in self.FOMC_DATES_2026:
            meeting_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if meeting_date > today:
                return date_str
        return None
    
    async def calculate_rate_probabilities(
        self, 
        meeting_date: str = None,
        include_market_data: bool = True,
    ) -> FedRateProbs:
        """
        Calculate probability distribution for Fed rate decision.
        
        Methodology:
        1. Start with neutral prior (if no market data)
        2. Adjust based on economic indicators
        3. Weight signals by historical predictive power
        """
        if not meeting_date:
            meeting_date = self.get_next_fomc_meeting()
        
        factors = []
        
        # Gather all indicators (including new dovish signals)
        indicators = await asyncio.gather(
            self.get_yield_curve(),
            self.get_unemployment(),
            self.get_core_pce(),
            self.get_financial_conditions(),
            self.get_initial_claims(),      # NEW: Weekly jobless claims
            self.get_credit_spreads(),      # NEW: HY credit spreads
            self.get_real_rates(),          # NEW: Real interest rates
            return_exceptions=True,
        )
        
        # Start with neutral prior
        probs = {
            "cut_50": 0.05,
            "cut_25": 0.20,
            "hold": 0.50,
            "hike_25": 0.20,
            "hike_50": 0.05,
        }
        
        # Count dovish/hawkish signals
        dovish_signals = 0
        hawkish_signals = 0
        total_signals = 0
        
        for indicator in indicators:
            if isinstance(indicator, Exception) or indicator is None:
                continue
            
            total_signals += 1
            
            factor = {
                "name": indicator.name,
                "value": indicator.value,
                "signal": indicator.signal,
                "interpretation": "",
            }
            
            if indicator.signal == "dovish":
                dovish_signals += 1
                factor["interpretation"] = "Supports rate cuts"
            elif indicator.signal == "hawkish":
                hawkish_signals += 1
                factor["interpretation"] = "Supports rates staying/rising"
            else:
                factor["interpretation"] = "Neutral impact"
            
            factors.append(factor)
        
        # Adjust probabilities based on signal balance
        if total_signals > 0:
            dovish_ratio = dovish_signals / total_signals
            hawkish_ratio = hawkish_signals / total_signals
            
            if dovish_ratio > 0.6:
                # Strong dovish signals - shift toward cuts
                probs["cut_50"] += 0.10
                probs["cut_25"] += 0.20
                probs["hold"] -= 0.15
                probs["hike_25"] -= 0.10
                probs["hike_50"] -= 0.05
            elif dovish_ratio > 0.4:
                # Mild dovish
                probs["cut_25"] += 0.10
                probs["hold"] -= 0.05
                probs["hike_25"] -= 0.05
            elif hawkish_ratio > 0.6:
                # Strong hawkish signals - shift toward hikes
                probs["hike_50"] += 0.10
                probs["hike_25"] += 0.20
                probs["hold"] -= 0.15
                probs["cut_25"] -= 0.10
                probs["cut_50"] -= 0.05
            elif hawkish_ratio > 0.4:
                # Mild hawkish
                probs["hike_25"] += 0.10
                probs["hold"] -= 0.05
                probs["cut_25"] -= 0.05
        
        # Normalize to sum to 1
        total = sum(probs.values())
        probs = {k: max(0, v / total) for k, v in probs.items()}
        
        # Determine confidence based on signal agreement
        if total_signals >= 3:
            max_ratio = max(dovish_ratio, hawkish_ratio, 1 - dovish_ratio - hawkish_ratio)
            if max_ratio > 0.7:
                confidence = "high"
            elif max_ratio > 0.5:
                confidence = "medium"
            else:
                confidence = "low"
        else:
            confidence = "low"
        
        return FedRateProbs(
            meeting_date=meeting_date,
            cut_50=round(probs["cut_50"], 3),
            cut_25=round(probs["cut_25"], 3),
            hold=round(probs["hold"], 3),
            hike_25=round(probs["hike_25"], 3),
            hike_50=round(probs["hike_50"], 3),
            confidence=confidence,
            factors=factors,
            updated_at=datetime.now(),
        )
    
    async def get_fed_dashboard(self) -> dict:
        """Get comprehensive Fed analysis dashboard."""
        current_rate = await self.get_current_rate()
        next_meeting = self.get_next_fomc_meeting()
        probs = await self.calculate_rate_probabilities(next_meeting)
        
        most_likely_action, most_likely_prob = probs.most_likely
        
        return {
            "current_rate": {
                "value": current_rate.value if current_rate else None,
                "date": current_rate.date if current_rate else None,
            },
            "next_meeting": next_meeting,
            "probability_distribution": {
                "cut_50_plus": f"{probs.cut_50*100:.1f}%",
                "cut_25": f"{probs.cut_25*100:.1f}%",
                "hold": f"{probs.hold*100:.1f}%",
                "hike_25": f"{probs.hike_25*100:.1f}%",
                "hike_50_plus": f"{probs.hike_50*100:.1f}%",
            },
            "prediction": {
                "action": most_likely_action.value,
                "probability": f"{most_likely_prob*100:.1f}%",
                "confidence": probs.confidence,
            },
            "factors": probs.factors,
            "updated_at": probs.updated_at.isoformat(),
        }


# Singleton
_module: FedResearchModule | None = None

def get_fed_module(fred_api_key: str = None) -> FedResearchModule:
    global _module
    if _module is None:
        _module = FedResearchModule(fred_api_key)
    return _module
