"""Arbitrage Scanner — Find price discrepancies between Kalshi and Polymarket.

Matches similar events across platforms and identifies profitable spreads.
"""
import asyncio
import json
import re
from dataclasses import dataclass
from datetime import datetime
from difflib import SequenceMatcher
from enum import Enum
from typing import Optional

from app.integrations.kalshi import get_kalshi_client
from app.integrations.polymarket import get_polymarket_client


class OpportunityTier(str, Enum):
    HOT = "hot"      # >5% spread, high liquidity, <24h
    WARM = "warm"    # 3-5% spread, decent liquidity
    COLD = "cold"    # 1-3% spread, monitor only


@dataclass
class Market:
    """Normalized market from either platform."""
    platform: str  # "kalshi" or "polymarket"
    id: str
    title: str
    question: str
    yes_price: float  # 0-1 scale
    no_price: float
    volume: float
    category: str
    close_time: Optional[datetime] = None
    raw: dict = None


@dataclass 
class ArbOpportunity:
    """A matched pair with arbitrage potential."""
    kalshi_market: Market
    poly_market: Market
    spread: float  # Price difference (0-1 scale)
    spread_pct: float  # Percentage edge
    direction: str  # "buy_kalshi" or "buy_poly"
    tier: OpportunityTier
    match_confidence: float  # 0-1 how confident we are it's same event
    explanation: str  # Plain English
    profit_example: str  # "Buy X at Y, sell Z at W = $N profit per $100"
    risks: list[str]


class ArbScanner:
    """Scans for arbitrage opportunities between prediction markets."""
    
    # Known matching patterns (Kalshi ticker patterns → Polymarket keywords)
    KNOWN_MATCHES = {
        # Fed Chair nominations - exact name matches
        "KXFEDCHAIRNOM": {
            "poly_keywords": ["fed chair", "nominate"],
            "match_on": "person_name",  # Match on the person's name in both
        },
        # Fed decisions - need to match meeting date
        "KXFEDDECISION": {
            "poly_keywords": ["fed", "interest rate", "fomc"],
            "match_on": "date_and_action",
        },
    }
    
    # Person names that appear in both platforms' Fed Chair markets
    FED_CHAIR_CANDIDATES = [
        "kevin warsh", "warsh",
        "janet yellen", "yellen",
        "larry kudlow", "kudlow",
        "rick rieder", "rieder",
        "jerome powell", "powell",
        "neel kashkari", "kashkari",
    ]
    
    # Category mappings for better matching
    CATEGORY_MAP = {
        "KXFEDDECISION": ["fed", "fed-rates", "fomc"],
        "KXFEDHIKE": ["fed", "fed-rates"],
        "KXRATECUT": ["fed", "fed-rates"],
        "KXFEDCHAIRNOM": ["fed", "fed-chair"],
        "KXCPI": ["inflation", "cpi"],
        "KXPHILSNOWM": ["weather"],
        "KXSNOWNYM": ["weather"],
        "KXHURCAT": ["weather", "hurricane"],
    }
    
    # Words to normalize out for matching
    NOISE_WORDS = {"will", "the", "be", "by", "at", "on", "in", "to", "a", "an", "?"}
    
    def __init__(self):
        self.kalshi = get_kalshi_client()
        self.poly = get_polymarket_client()
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for matching."""
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
        words = [w for w in text.split() if w not in self.NOISE_WORDS]
        return ' '.join(words)
    
    def _similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two strings."""
        norm1 = self._normalize_text(text1)
        norm2 = self._normalize_text(text2)
        return SequenceMatcher(None, norm1, norm2).ratio()
    
    def _extract_key_terms(self, text: str) -> set:
        """Extract key terms for matching."""
        text = text.lower()
        
        # High-value entities that indicate same event
        entities = {
            # Fed/Monetary
            "fed", "federal reserve", "fomc", "rate cut", "rate hike", "interest rate",
            "jerome powell", "powell",
            # Politics
            "trump", "biden", "harris", "election", "president", "senate", "house",
            # Crypto
            "bitcoin", "btc", "ethereum", "eth", "crypto",
            # Economics
            "cpi", "inflation", "unemployment", "gdp", "recession",
            # Weather
            "hurricane", "snow", "weather", "temperature",
            # Specific identifiers
            "2024", "2025", "2026", "2027", "2028",
            "january", "february", "march", "april", "may", "june",
            "july", "august", "september", "october", "november", "december",
        }
        found = set()
        for entity in entities:
            if entity in text:
                found.add(entity)
        return found
    
    def _events_match(self, km: Market, pm: Market) -> tuple[bool, float]:
        """
        Determine if two markets represent the same event.
        
        Returns (is_match, confidence)
        """
        k_title = km.title.lower()
        p_question = pm.question.lower()
        
        # Special case: Fed Chair nominations - match on candidate name
        if "fed chair" in k_title or "fed chair" in p_question:
            for candidate in self.FED_CHAIR_CANDIDATES:
                if candidate in k_title and candidate in p_question:
                    return True, 0.95  # High confidence for name match
        
        # Special case: Fed decisions - match on date and action type
        if "federal reserve" in k_title and "fed" in p_question:
            # Check for same action (cut/hike/hold)
            k_cut = "cut" in k_title
            k_hike = "hike" in k_title
            p_cut = "cut" in p_question or "decrease" in p_question
            p_hike = "hike" in p_question or "increase" in p_question
            
            if (k_cut and p_cut) or (k_hike and p_hike):
                # Check for same meeting (by date keywords)
                months = ["january", "february", "march", "april", "may", "june",
                          "july", "august", "september", "october", "november", "december"]
                for month in months:
                    if month in k_title and month in p_question:
                        return True, 0.85
        
        # Generic matching: Extract key terms
        k_terms = self._extract_key_terms(k_title)
        p_terms = self._extract_key_terms(p_question)
        
        # Must have at least 2 overlapping key terms
        overlap = k_terms & p_terms
        if len(overlap) < 2:
            return False, 0.0
        
        # Check for contradictory terms
        contradictions = [
            ({"cut", "rate cut", "decrease"}, {"hike", "rate hike", "increase"}),
        ]
        for set_a, set_b in contradictions:
            if (k_terms & set_a and p_terms & set_b) or (k_terms & set_b and p_terms & set_a):
                return False, 0.0
        
        # Calculate confidence based on term overlap and text similarity
        term_overlap_score = len(overlap) / max(len(k_terms | p_terms), 1)
        text_sim = self._similarity(k_title, p_question)
        
        confidence = (term_overlap_score * 0.5) + (text_sim * 0.5)
        
        return confidence >= 0.50, confidence
    
    async def _fetch_kalshi_markets(self, limit: int = 200) -> list[Market]:
        """Fetch and normalize Kalshi markets."""
        markets = []
        
        # Get markets from key series (Kalshi uses "open" not "active")
        try:
            result = await self.kalshi.get_markets(status="open", limit=limit)
            raw_markets = result.get("markets", [])
            
            for m in raw_markets:
                # Extract price (yes_bid preferred, fallback to yes_ask midpoint)
                yes_bid = m.get("yes_bid", 0)
                yes_ask = m.get("yes_ask", 100)
                
                # Use midpoint if both bid/ask exist, otherwise use bid
                if yes_bid > 0:
                    yes_price = yes_bid / 100
                elif yes_ask < 100:
                    yes_price = yes_ask / 100
                else:
                    yes_price = 0.5  # Default for inactive markets
                no_price = 1 - yes_price
                
                markets.append(Market(
                    platform="kalshi",
                    id=m.get("ticker", ""),
                    title=m.get("title", ""),
                    question=m.get("title", ""),
                    yes_price=yes_price,
                    no_price=no_price,
                    volume=float(m.get("volume", 0) or 0),
                    category=m.get("series_ticker", ""),
                    raw=m,
                ))
        except Exception as e:
            print(f"Error fetching Kalshi: {e}")
        
        return markets
    
    async def _fetch_poly_markets(self, limit: int = 200) -> list[Market]:
        """Fetch and normalize Polymarket events/markets."""
        markets = []
        
        try:
            events = await self.poly.get_events(active=True, closed=False, limit=limit)
            
            for event in events:
                event_markets = event.get("markets", [])
                tags = [t.get("slug", "") for t in event.get("tags", [])]
                category = tags[0] if tags else ""
                
                for m in event_markets:
                    # Parse prices
                    prices_str = m.get("outcomePrices", "[]")
                    prices = json.loads(prices_str) if isinstance(prices_str, str) else prices_str
                    
                    yes_price = float(prices[0]) if prices else 0.5
                    no_price = float(prices[1]) if len(prices) > 1 else (1 - yes_price)
                    
                    markets.append(Market(
                        platform="polymarket",
                        id=m.get("id", ""),
                        title=event.get("title", ""),
                        question=m.get("question", ""),
                        yes_price=yes_price,
                        no_price=no_price,
                        volume=float(m.get("volume", 0) or 0),
                        category=category,
                        raw=m,
                    ))
        except Exception as e:
            print(f"Error fetching Polymarket: {e}")
        
        return markets
    
    def _find_matches(
        self, 
        kalshi_markets: list[Market], 
        poly_markets: list[Market],
        min_similarity: float = 0.5,
    ) -> list[tuple[Market, Market, float]]:
        """Find matching markets across platforms."""
        matches = []
        
        for km in kalshi_markets:
            # Skip markets with default 50¢ price (no liquidity)
            if km.yes_price == 0.5 and km.volume < 1000:
                continue
                
            for pm in poly_markets:
                # Skip resolved/zero-priced markets
                if pm.yes_price == 0 or pm.yes_price == 1:
                    continue
                
                # Use improved matching
                is_match, confidence = self._events_match(km, pm)
                
                if is_match and confidence >= min_similarity:
                    matches.append((km, pm, confidence))
        
        # Sort by confidence
        matches.sort(key=lambda x: x[2], reverse=True)
        
        # Deduplicate - keep best match for each Kalshi market
        seen_kalshi = set()
        unique_matches = []
        for km, pm, conf in matches:
            if km.id not in seen_kalshi:
                seen_kalshi.add(km.id)
                unique_matches.append((km, pm, conf))
        
        return unique_matches
    
    def _calculate_opportunity(
        self, 
        km: Market, 
        pm: Market, 
        confidence: float,
    ) -> Optional[ArbOpportunity]:
        """Calculate arbitrage opportunity from a matched pair."""
        
        # Calculate spread (absolute difference in YES prices)
        spread = abs(km.yes_price - pm.yes_price)
        
        # Need meaningful spread
        if spread < 0.01:  # Less than 1%
            return None
        
        # Determine direction
        if km.yes_price < pm.yes_price:
            # Buy YES on Kalshi (cheaper), sell YES on Polymarket (expensive)
            # Or: Buy YES Kalshi + Buy NO Polymarket
            direction = "buy_kalshi_yes"
            buy_price = km.yes_price
            sell_price = pm.yes_price
            buy_platform = "Kalshi"
            sell_platform = "Polymarket"
        else:
            direction = "buy_poly_yes"
            buy_price = pm.yes_price
            sell_price = km.yes_price
            buy_platform = "Polymarket"
            sell_platform = "Kalshi"
        
        spread_pct = (spread / buy_price * 100) if buy_price > 0 else 0
        
        # Determine tier
        if spread >= 0.05 and min(km.volume, pm.volume) > 10000:
            tier = OpportunityTier.HOT
        elif spread >= 0.03:
            tier = OpportunityTier.WARM
        else:
            tier = OpportunityTier.COLD
        
        # Generate explanation
        explanation = (
            f"'{km.title[:50]}...' is priced differently across platforms. "
            f"{buy_platform} has YES at {buy_price*100:.0f}¢ while {sell_platform} has it at {sell_price*100:.0f}¢."
        )
        
        # Profit example
        profit_per_100 = spread * 100
        profit_example = (
            f"Buy YES on {buy_platform} at {buy_price*100:.0f}¢ + "
            f"Buy NO on {sell_platform} at {(1-sell_price)*100:.0f}¢ = "
            f"${profit_per_100:.2f} guaranteed profit per $100 risked"
        )
        
        # Risks
        risks = [
            "Resolution timing may differ between platforms",
            "Outcome definitions may have slight differences",
            "Liquidity may not support large positions",
        ]
        if confidence < 0.7:
            risks.insert(0, "⚠️ Match confidence is moderate — verify manually")
        
        return ArbOpportunity(
            kalshi_market=km,
            poly_market=pm,
            spread=spread,
            spread_pct=spread_pct,
            direction=direction,
            tier=tier,
            match_confidence=confidence,
            explanation=explanation,
            profit_example=profit_example,
            risks=risks,
        )
    
    async def scan(
        self, 
        min_spread: float = 0.02,  # Minimum 2% spread
        min_confidence: float = 0.5,
        limit: int = 100,
    ) -> list[ArbOpportunity]:
        """
        Scan for arbitrage opportunities.
        
        Args:
            min_spread: Minimum spread to report (0-1 scale)
            min_confidence: Minimum match confidence
            limit: Max markets to fetch per platform
            
        Returns:
            List of opportunities sorted by spread (best first)
        """
        # Fetch markets concurrently
        kalshi_markets, poly_markets = await asyncio.gather(
            self._fetch_kalshi_markets(limit),
            self._fetch_poly_markets(limit),
        )
        
        # Find matches
        matches = self._find_matches(kalshi_markets, poly_markets, min_confidence)
        
        # Calculate opportunities
        opportunities = []
        for km, pm, confidence in matches:
            opp = self._calculate_opportunity(km, pm, confidence)
            if opp and opp.spread >= min_spread:
                opportunities.append(opp)
        
        # Sort by spread (best opportunities first)
        opportunities.sort(key=lambda x: x.spread, reverse=True)
        
        return opportunities
    
    async def get_summary(self) -> dict:
        """Get quick summary of current opportunities."""
        opps = await self.scan(min_spread=0.01)
        
        return {
            "total_opportunities": len(opps),
            "hot": len([o for o in opps if o.tier == OpportunityTier.HOT]),
            "warm": len([o for o in opps if o.tier == OpportunityTier.WARM]),
            "cold": len([o for o in opps if o.tier == OpportunityTier.COLD]),
            "best_spread": opps[0].spread if opps else 0,
            "best_opportunity": opps[0].explanation if opps else None,
        }


# Singleton
_scanner: ArbScanner | None = None

def get_arb_scanner() -> ArbScanner:
    global _scanner
    if _scanner is None:
        _scanner = ArbScanner()
    return _scanner
