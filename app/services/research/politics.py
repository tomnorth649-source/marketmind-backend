"""Politics Research Module.

Provides probability estimates for:
- Election outcomes
- Policy decisions (tariffs, legislation)
- Nominations/confirmations
- Approval ratings

Data sources:
- FiveThirtyEight (poll aggregates)
- RealClearPolitics
- Ballotpedia
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import httpx


@dataclass
class Poll:
    """Polling data."""
    pollster: str
    date: str
    sample_size: Optional[int]
    margin_of_error: Optional[float]
    results: dict[str, float]  # candidate/option -> percentage


@dataclass
class ElectionForecast:
    """Election probability forecast."""
    race: str
    candidates: dict[str, float]  # candidate -> probability
    confidence: str
    last_updated: datetime
    factors: list[dict]


class PoliticsModule:
    """Research module for political predictions."""
    
    # 538 data endpoints
    FIVETHIRTYEIGHT_BASE = "https://projects.fivethirtyeight.com"
    
    # Key political events
    UPCOMING_EVENTS = {
        "2026_midterms": {
            "name": "2026 Midterm Elections",
            "date": "2026-11-03",
            "type": "election",
        },
        "2028_presidential": {
            "name": "2028 Presidential Election",
            "date": "2028-11-07",
            "type": "election",
        },
    }
    
    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = timedelta(hours=6)  # Polls don't change fast
    
    async def _fetch_url(self, url: str) -> dict:
        """Fetch JSON from URL."""
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=30.0)
            response.raise_for_status()
            return response.json()
    
    def estimate_approval_rating(self, current: float, trend: float = 0) -> dict:
        """
        Estimate probability of approval above/below thresholds.
        
        Args:
            current: Current approval rating
            trend: Recent trend (+/- points per month)
        """
        thresholds = [40, 45, 50, 55]
        probabilities = {}
        
        for threshold in thresholds:
            distance = current - threshold
            
            # Base probability from distance
            if distance > 5:
                prob = 0.9
            elif distance > 2:
                prob = 0.75
            elif distance > 0:
                prob = 0.6
            elif distance > -2:
                prob = 0.4
            elif distance > -5:
                prob = 0.25
            else:
                prob = 0.1
            
            # Adjust for trend
            if trend > 0:
                prob = min(0.95, prob + 0.1)
            elif trend < 0:
                prob = max(0.05, prob - 0.1)
            
            probabilities[f"above_{threshold}"] = round(prob, 2)
            probabilities[f"below_{threshold}"] = round(1 - prob, 2)
        
        return probabilities
    
    def estimate_confirmation_probability(
        self,
        nominee: str,
        position: str,
        senate_margin: int,  # + for Dem majority, - for Rep majority
        controversy_level: str = "low",  # low, medium, high
    ) -> dict:
        """
        Estimate probability of Senate confirmation.
        
        Based on:
        - Senate partisan makeup
        - Historical confirmation rates
        - Controversy level
        """
        # Base: most nominees get confirmed
        base_prob = 0.85
        
        factors = []
        
        # Majority party advantage
        if senate_margin > 5:
            base_prob += 0.10
            factors.append({"name": "Senate margin", "value": f"+{senate_margin}", "impact": "positive"})
        elif senate_margin < -5:
            base_prob -= 0.10
            factors.append({"name": "Senate margin", "value": str(senate_margin), "impact": "negative"})
        else:
            factors.append({"name": "Senate margin", "value": f"{senate_margin:+d}", "impact": "neutral"})
        
        # Controversy adjustment
        if controversy_level == "high":
            base_prob -= 0.25
            factors.append({"name": "Controversy", "value": "High", "impact": "negative"})
        elif controversy_level == "medium":
            base_prob -= 0.10
            factors.append({"name": "Controversy", "value": "Medium", "impact": "negative"})
        else:
            factors.append({"name": "Controversy", "value": "Low", "impact": "positive"})
        
        probability = max(0.1, min(0.95, base_prob))
        
        return {
            "nominee": nominee,
            "position": position,
            "probability": round(probability, 2),
            "probability_display": f"{probability*100:.0f}%",
            "factors": factors,
        }
    
    def estimate_policy_probability(
        self,
        policy: str,
        current_status: str,  # "proposed", "committee", "passed_house", "passed_senate"
        bipartisan_support: str = "low",  # low, medium, high
    ) -> dict:
        """
        Estimate probability of policy/legislation passing.
        """
        # Base probability by status
        status_probs = {
            "proposed": 0.15,
            "committee": 0.25,
            "passed_house": 0.50,
            "passed_senate": 0.60,
            "conference": 0.75,
        }
        
        base_prob = status_probs.get(current_status, 0.20)
        
        factors = [{"name": "Current status", "value": current_status, "impact": "neutral"}]
        
        # Bipartisan support adjustment
        if bipartisan_support == "high":
            base_prob += 0.25
            factors.append({"name": "Bipartisan support", "value": "High", "impact": "positive"})
        elif bipartisan_support == "medium":
            base_prob += 0.10
            factors.append({"name": "Bipartisan support", "value": "Medium", "impact": "positive"})
        else:
            factors.append({"name": "Bipartisan support", "value": "Low", "impact": "negative"})
        
        probability = max(0.05, min(0.95, base_prob))
        
        return {
            "policy": policy,
            "status": current_status,
            "probability": round(probability, 2),
            "probability_display": f"{probability*100:.0f}%",
            "factors": factors,
        }
    
    async def get_politics_dashboard(self) -> dict:
        """Get political research dashboard."""
        # Current political context
        # (In production, would fetch from news APIs)
        
        return {
            "upcoming_events": [
                {
                    "name": event["name"],
                    "date": event["date"],
                    "type": event["type"],
                }
                for event in self.UPCOMING_EVENTS.values()
            ],
            "current_context": {
                "president": "Donald Trump (R)",
                "senate": "Republican majority (51-49)",
                "house": "Republican majority",
                "presidential_approval": "~45%",  # Would fetch from polls
            },
            "note": "Political forecasts require poll aggregation data. Use with caution.",
        }


# Singleton
_module: PoliticsModule | None = None

def get_politics_module() -> PoliticsModule:
    global _module
    if _module is None:
        _module = PoliticsModule()
    return _module
