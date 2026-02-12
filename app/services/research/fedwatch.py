"""CME FedWatch Integration.

Two modes:
1. Official CME API ($25/month EOD, more for intraday)
2. Web scraping fallback (free, less reliable)

The FedWatch tool calculates probabilities from 30-Day Fed Funds futures.
"""
import asyncio
import re
from dataclasses import dataclass, asdict
from datetime import datetime, date
from typing import Optional
import httpx


@dataclass
class FedWatchProbs:
    """CME FedWatch probability data for a single FOMC meeting."""
    meeting_date: str
    current_target_low: float  # Current target range low (e.g., 4.25)
    current_target_high: float  # Current target range high (e.g., 4.50)
    probabilities: dict[str, float]  # target_range -> probability
    most_likely_range: str
    most_likely_prob: float
    implied_rate: float  # Weighted average implied rate
    source: str  # "cme_api" or "scraped"
    updated_at: datetime
    
    def to_dict(self) -> dict:
        d = asdict(self)
        d["updated_at"] = self.updated_at.isoformat()
        return d


@dataclass 
class FedWatchTimeline:
    """FedWatch probabilities for all upcoming meetings."""
    meetings: list[FedWatchProbs]
    current_rate: float
    rate_path: list[dict]  # [{date, implied_rate, change_from_current}]
    cuts_priced_ytd: float  # Total cuts priced in for rest of year
    updated_at: datetime


class CMEFedWatchClient:
    """Client for CME FedWatch data."""
    
    # CME API endpoints (requires subscription)
    CME_API_BASE = "https://api.cmegroup.com/fedwatch/v1"
    
    # Fallback: scrape the public page
    FEDWATCH_PAGE = "https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html"
    
    # FOMC 2026 meeting dates (announcement days)
    FOMC_2026 = [
        "2026-01-29",  # Jan
        "2026-03-18",  # Mar
        "2026-05-06",  # May
        "2026-06-17",  # Jun
        "2026-07-29",  # Jul
        "2026-09-16",  # Sep
        "2026-11-04",  # Nov
        "2026-12-16",  # Dec
    ]
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize FedWatch client.
        
        Args:
            api_key: CME API key (optional - falls back to scraping)
        """
        self.api_key = api_key
        self._cache: dict = {}
        self._cache_ttl_seconds = 300  # 5 min cache
    
    async def get_probabilities(self, meeting_date: str = None) -> FedWatchProbs:
        """
        Get FedWatch probabilities for a specific meeting.
        
        Args:
            meeting_date: FOMC meeting date (YYYY-MM-DD). Defaults to next meeting.
        """
        if not meeting_date:
            meeting_date = self._get_next_meeting()
        
        # Try API first if we have a key
        if self.api_key:
            try:
                return await self._fetch_from_api(meeting_date)
            except Exception as e:
                print(f"CME API failed, falling back to scrape: {e}")
        
        # Fallback to scraping
        return await self._scrape_fedwatch(meeting_date)
    
    async def get_timeline(self) -> FedWatchTimeline:
        """Get FedWatch probabilities for all upcoming meetings."""
        today = date.today()
        upcoming = [d for d in self.FOMC_2026 if date.fromisoformat(d) > today]
        
        meetings = []
        for meeting_date in upcoming[:6]:  # Next 6 meetings
            try:
                probs = await self.get_probabilities(meeting_date)
                meetings.append(probs)
            except Exception as e:
                print(f"Failed to get probs for {meeting_date}: {e}")
        
        if not meetings:
            raise ValueError("Could not fetch any meeting data")
        
        # Calculate rate path
        current_rate = meetings[0].current_target_high
        rate_path = []
        for m in meetings:
            rate_path.append({
                "date": m.meeting_date,
                "implied_rate": m.implied_rate,
                "change_from_current": round(m.implied_rate - current_rate, 3),
            })
        
        # Total cuts priced
        last_implied = meetings[-1].implied_rate if meetings else current_rate
        cuts_priced = round(current_rate - last_implied, 3)
        
        return FedWatchTimeline(
            meetings=meetings,
            current_rate=current_rate,
            rate_path=rate_path,
            cuts_priced_ytd=cuts_priced,
            updated_at=datetime.now(),
        )
    
    async def _fetch_from_api(self, meeting_date: str) -> FedWatchProbs:
        """Fetch from official CME API (requires $25/month subscription)."""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Accept": "application/json",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.CME_API_BASE}/probabilities",
                params={"meeting_date": meeting_date},
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
        
        # Parse CME API response format
        return self._parse_api_response(data, meeting_date)
    
    async def _scrape_fedwatch(self, meeting_date: str) -> FedWatchProbs:
        """
        Scrape FedWatch data from public page.
        
        Note: This is a fallback. The page is JavaScript-heavy,
        so we may need to use browser automation for full data.
        For now, we'll return estimated probabilities based on
        Fed Funds futures prices from other sources.
        """
        # For now, return a placeholder that indicates scraping is needed
        # In production, use Playwright/Selenium to render the JS
        
        # Try to get Fed Funds futures data from alternative sources
        return await self._estimate_from_futures(meeting_date)
    
    async def _estimate_from_futures(self, meeting_date: str) -> FedWatchProbs:
        """
        Estimate FedWatch probabilities from Fed Funds futures.
        
        The FedWatch methodology:
        1. Get the Fed Funds futures price for the meeting month
        2. Implied rate = 100 - futures price
        3. Probability = distance from current rate / 25bps
        
        This is a simplified version of CME's exact calculation.
        """
        # Current Fed Funds target (as of Feb 2026)
        # This should be fetched dynamically in production
        current_low = 4.25
        current_high = 4.50
        current_mid = (current_low + current_high) / 2
        
        # For demo purposes, use reasonable estimates
        # In production, fetch actual futures prices
        meeting_dt = date.fromisoformat(meeting_date)
        months_out = (meeting_dt.year - date.today().year) * 12 + \
                     (meeting_dt.month - date.today().month)
        
        # Market is pricing ~2 cuts by end of 2026 (typical estimate)
        # Each month out adds probability of movement
        cuts_expected = min(months_out * 0.3, 2.0)  # Max ~2 cuts
        implied_rate = current_mid - (cuts_expected * 0.25)
        
        # Build probability distribution
        possible_ranges = [
            "5.25-5.50", "5.00-5.25", "4.75-5.00", "4.50-4.75",
            "4.25-4.50", "4.00-4.25", "3.75-4.00", "3.50-3.75",
            "3.25-3.50", "3.00-3.25"
        ]
        
        # Simple probability model centered on implied rate
        probs = {}
        for range_str in possible_ranges:
            low, high = map(float, range_str.split("-"))
            mid = (low + high) / 2
            distance = abs(mid - implied_rate)
            # Probability decreases with distance
            prob = max(0, 1 - distance * 2)
            probs[range_str] = round(prob, 3)
        
        # Normalize
        total = sum(probs.values())
        if total > 0:
            probs = {k: round(v/total, 3) for k, v in probs.items()}
        
        # Find most likely
        most_likely = max(probs, key=probs.get)
        
        return FedWatchProbs(
            meeting_date=meeting_date,
            current_target_low=current_low,
            current_target_high=current_high,
            probabilities=probs,
            most_likely_range=most_likely,
            most_likely_prob=probs[most_likely],
            implied_rate=round(implied_rate, 3),
            source="estimated",
            updated_at=datetime.now(),
        )
    
    def _parse_api_response(self, data: dict, meeting_date: str) -> FedWatchProbs:
        """Parse CME API response into our data model."""
        # Actual format TBD based on CME API docs
        # This is a placeholder structure
        probs = data.get("probabilities", {})
        
        most_likely = max(probs, key=probs.get) if probs else "4.25-4.50"
        
        return FedWatchProbs(
            meeting_date=meeting_date,
            current_target_low=data.get("current_low", 4.25),
            current_target_high=data.get("current_high", 4.50),
            probabilities=probs,
            most_likely_range=most_likely,
            most_likely_prob=probs.get(most_likely, 0),
            implied_rate=data.get("implied_rate", 4.375),
            source="cme_api",
            updated_at=datetime.now(),
        )
    
    def _get_next_meeting(self) -> str:
        """Get the next FOMC meeting date."""
        today = date.today()
        for meeting in self.FOMC_2026:
            if date.fromisoformat(meeting) > today:
                return meeting
        return self.FOMC_2026[-1]  # Fallback to last meeting


# Singleton
_client: CMEFedWatchClient | None = None

def get_fedwatch_client(api_key: str = None) -> CMEFedWatchClient:
    global _client
    if _client is None:
        _client = CMEFedWatchClient(api_key)
    return _client
