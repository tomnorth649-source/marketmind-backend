"""
Research Query Engine - The brain of MarketMind.

Free tier (A): Rule-based intent classification + templated research pipelines
Premium tier (B): LLM-powered synthesis with any query support

Supports multiple LLM backends: OpenAI, Anthropic, Groq (cheapest)
"""
import asyncio
import re
import json
from datetime import datetime
from typing import Optional, Literal
from dataclasses import dataclass, field
from enum import Enum

import httpx

from app.config import get_settings
from app.integrations.polymarket import get_polymarket_client
from app.integrations.kalshi import get_kalshi_client


# ══════════════════════════════════════════════════════════════════════════════
# TYPES & CONFIG
# ══════════════════════════════════════════════════════════════════════════════

class QueryCategory(str, Enum):
    FED = "fed"
    CRYPTO = "crypto"
    POLITICS = "politics"
    SPORTS = "sports"
    WEATHER = "weather"
    GENERAL = "general"


class Confidence(str, Enum):
    HIGH = "high"      # Multiple agreeing sources
    MEDIUM = "medium"  # Some disagreement or limited sources
    LOW = "low"        # Sparse data or high uncertainty


@dataclass
class Source:
    type: str  # "market", "fedwatch", "price_data", "news", "model"
    name: str
    probability: Optional[float] = None
    data: dict = field(default_factory=dict)
    url: Optional[str] = None


@dataclass
class ResearchResult:
    query: str
    category: QueryCategory
    probability: Optional[float]
    confidence: Confidence
    sources: list[Source]
    reasoning: str
    related_markets: list[dict] = field(default_factory=list)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    tier: Literal["free", "premium"] = "free"
    processing_time_ms: int = 0


# Category keywords for intent classification
CATEGORY_KEYWORDS = {
    QueryCategory.FED: [
        "fed", "fomc", "interest rate", "rate cut", "rate hike", "powell",
        "federal reserve", "monetary policy", "basis points", "bps"
    ],
    QueryCategory.CRYPTO: [
        "bitcoin", "btc", "ethereum", "eth", "crypto", "solana", "sol",
        "dogecoin", "xrp", "cryptocurrency", "blockchain", "halving"
    ],
    QueryCategory.POLITICS: [
        "trump", "biden", "election", "congress", "senate", "president",
        "republican", "democrat", "vote", "polling", "governor", "supreme court"
    ],
    QueryCategory.SPORTS: [
        "nba", "nfl", "mlb", "nhl", "super bowl", "championship", "playoffs",
        "lakers", "celtics", "chiefs", "game", "match", "finals", "mvp"
    ],
    QueryCategory.WEATHER: [
        "weather", "temperature", "hurricane", "tornado", "snow", "rain",
        "climate", "forecast", "storm", "heat wave", "cold"
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
# INTENT CLASSIFICATION (Free Tier)
# ══════════════════════════════════════════════════════════════════════════════

def classify_intent(query: str) -> QueryCategory:
    """Rule-based intent classification."""
    query_lower = query.lower()
    
    scores = {cat: 0 for cat in QueryCategory}
    
    for category, keywords in CATEGORY_KEYWORDS.items():
        for keyword in keywords:
            if keyword in query_lower:
                scores[category] += 1
                # Exact word match gets bonus
                if re.search(rf'\b{re.escape(keyword)}\b', query_lower):
                    scores[category] += 1
    
    best_category = max(scores, key=scores.get)
    if scores[best_category] == 0:
        return QueryCategory.GENERAL
    
    return best_category


def extract_entities(query: str) -> dict:
    """Extract key entities from query."""
    entities = {}
    query_lower = query.lower()
    
    # Dates
    date_patterns = [
        r'(january|february|march|april|may|june|july|august|september|october|november|december)\s*\d{4}',
        r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\s*\d{4}',
        r'\d{4}',
        r'(this|next)\s+(week|month|year)',
        r'(q[1-4])\s*\d{4}',
    ]
    for pattern in date_patterns:
        match = re.search(pattern, query_lower)
        if match:
            entities["timeframe"] = match.group()
            break
    
    # Numbers/prices
    price_match = re.search(r'\$?([\d,]+(?:\.\d+)?)[k|m|b]?', query_lower)
    if price_match:
        entities["target_value"] = price_match.group()
    
    # Crypto symbols
    crypto_match = re.search(r'\b(btc|eth|sol|xrp|doge|ada)\b', query_lower)
    if crypto_match:
        entities["crypto_symbol"] = crypto_match.group().upper()
    
    return entities


# ══════════════════════════════════════════════════════════════════════════════
# DATA FETCHERS
# ══════════════════════════════════════════════════════════════════════════════

async def search_markets(query: str, limit: int = 10) -> list[dict]:
    """Search both Polymarket and Kalshi for relevant markets."""
    poly_client = get_polymarket_client()
    kalshi_client = get_kalshi_client()
    
    results = []
    # Extract meaningful keywords (including short important ones like "fed", "btc")
    important_short_words = {"fed", "btc", "eth", "nba", "nfl", "mlb", "gop", "dem", "ai", "us", "uk"}
    query_lower = query.lower()
    query_words = [w for w in query_lower.split() if len(w) > 2 or w in important_short_words]
    
    # Add category-specific keywords
    if "fed" in query_lower or "rate" in query_lower or "fomc" in query_lower:
        query_words.extend(["fed", "rate", "fomc", "cut", "hike", "interest"])
    if "bitcoin" in query_lower or "btc" in query_lower:
        query_words.extend(["bitcoin", "btc", "crypto"])
    if "trump" in query_lower or "biden" in query_lower or "election" in query_lower:
        query_words.extend(["trump", "biden", "election", "president"])
    
    query_words = list(set(query_words))  # Dedupe
    
    try:
        # Fetch Polymarket
        poly_markets = await poly_client.get_markets(closed=False, limit=200)
        for m in poly_markets:
            title = (m.get("question") or m.get("title") or "").lower()
            # Match if any query word appears in title
            if any(word in title for word in query_words):
                # Parse price
                prices = m.get("outcomePrices", "[0.5, 0.5]")
                try:
                    if isinstance(prices, str):
                        prices = eval(prices)
                    yes_price = float(prices[0]) if prices else 0.5
                except:
                    yes_price = 0.5
                
                results.append({
                    "platform": "polymarket",
                    "id": m.get("id"),
                    "title": m.get("question") or m.get("title"),
                    "probability": yes_price,
                    "volume": float(m.get("volumeNum", 0) or 0),
                    "url": f"https://polymarket.com/event/{m.get('slug', m.get('id'))}",
                })
    except Exception as e:
        print(f"Polymarket search error: {e}")
    
    try:
        # Fetch Kalshi
        kalshi_result = await kalshi_client.get_markets(status="open", limit=100)
        kalshi_markets = kalshi_result.get("markets", [])
        for m in kalshi_markets:
            title = (m.get("title") or m.get("subtitle") or "").lower()
            # Match if any query word appears in title
            if any(word in title for word in query_words):
                yes_price = float(m.get("yes_ask", 50)) / 100
                results.append({
                    "platform": "kalshi",
                    "id": m.get("ticker"),
                    "title": m.get("title") or m.get("subtitle"),
                    "probability": yes_price,
                    "volume": float(m.get("volume", 0) or 0),
                    "url": f"https://kalshi.com/markets/{m.get('ticker')}",
                })
    except Exception as e:
        print(f"Kalshi search error: {e}")
    
    # Sort by volume (most liquid first)
    results.sort(key=lambda x: x["volume"], reverse=True)
    return results[:limit]


async def fetch_fedwatch_data() -> dict:
    """Fetch CME FedWatch probabilities."""
    try:
        from app.services.research.fedwatch import get_fedwatch_client
        client = get_fedwatch_client()
        timeline = await client.get_timeline()
        
        if not timeline or not timeline.meetings:
            return {}
        
        meetings = timeline.meetings
        if not meetings:
            return {}
        
        # Get next meeting (it's a FedWatchProbs dataclass)
        next_meeting = meetings[0]
        current_low = next_meeting.current_target_low
        current_high = next_meeting.current_target_high
        current_range = f"{current_low}-{current_high}"
        
        # Calculate cut probability (sum of all ranges below current)
        probs = next_meeting.probabilities or {}
        cut_prob = 0.0
        hold_prob = 0.0
        hike_prob = 0.0
        
        for range_str, prob in probs.items():
            try:
                low = float(range_str.split("-")[0])
                if low < current_low:
                    cut_prob += prob
                elif low == current_low:
                    hold_prob += prob
                else:
                    hike_prob += prob
            except:
                continue
        
        return {
            "next_meeting": next_meeting.meeting_date,
            "current_rate": f"{current_low}-{current_high}%",
            "cut_prob": cut_prob,
            "hold_prob": hold_prob,
            "hike_prob": hike_prob,
            "implied_rate": next_meeting.implied_rate,
            "probabilities": probs,
        }
    except Exception as e:
        print(f"FedWatch error: {e}")
        return {}


async def fetch_crypto_price(symbol: str) -> dict:
    """Fetch current crypto price data."""
    try:
        from app.services.research.crypto import CryptoResearch
        service = CryptoResearch()
        return await service.get_price(symbol)
    except Exception as e:
        print(f"Crypto price error: {e}")
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# CATEGORY RESEARCH PIPELINES (Free Tier)
# ══════════════════════════════════════════════════════════════════════════════

async def research_fed(query: str, entities: dict) -> ResearchResult:
    """Research pipeline for Fed/rates questions."""
    sources = []
    probabilities = []
    query_lower = query.lower()
    
    # Determine what the user is asking about
    asking_about_cut = any(w in query_lower for w in ["cut", "lower", "reduce", "decrease"])
    asking_about_hike = any(w in query_lower for w in ["hike", "raise", "increase"])
    
    # 1. FedWatch data (primary source for Fed questions)
    fedwatch = await fetch_fedwatch_data()
    if fedwatch and fedwatch.get("next_meeting"):
        # Determine which probability to report based on query
        if asking_about_cut:
            prob = fedwatch.get("cut_prob", 0)
            prob_type = "cut"
        elif asking_about_hike:
            prob = fedwatch.get("hike_prob", 0)
            prob_type = "hike"
        else:
            # Default to most likely action
            cut_prob = fedwatch.get("cut_prob", 0)
            hold_prob = fedwatch.get("hold_prob", 0)
            if cut_prob > hold_prob:
                prob = cut_prob
                prob_type = "cut"
            else:
                prob = hold_prob
                prob_type = "hold"
        
        sources.append(Source(
            type="fedwatch",
            name=f"CME FedWatch - {fedwatch.get('next_meeting', 'Next Meeting')}",
            probability=prob,
            data={
                "next_meeting": fedwatch.get("next_meeting"),
                "current_rate": fedwatch.get("current_rate"),
                "cut_prob": f"{fedwatch.get('cut_prob', 0)*100:.1f}%",
                "hold_prob": f"{fedwatch.get('hold_prob', 0)*100:.1f}%",
                "implied_rate": fedwatch.get("implied_rate"),
            },
            url="https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html",
        ))
        probabilities.append(prob)
    
    # 2. Search prediction markets for Fed-related markets
    markets = await search_markets(query)
    fed_markets = [m for m in markets if any(
        kw in m["title"].lower() for kw in ["fed", "rate", "fomc", "interest", "powell", "monetary"]
    )]
    
    for m in fed_markets[:3]:
        sources.append(Source(
            type="market",
            name=f"{m['platform']}: {m['title'][:60]}",
            probability=m["probability"],
            data={"volume": m["volume"]},
            url=m["url"],
        ))
        probabilities.append(m["probability"])
    
    # Calculate probability (FedWatch weighted more heavily)
    if probabilities:
        # FedWatch is first and gets 2x weight
        if len(probabilities) > 1:
            weighted = probabilities[0] * 2 + sum(probabilities[1:])
            avg_prob = weighted / (len(probabilities) + 1)
        else:
            avg_prob = probabilities[0]
        confidence = Confidence.HIGH if fedwatch else Confidence.MEDIUM
    else:
        avg_prob = None
        confidence = Confidence.LOW
    
    # Generate detailed reasoning
    reasoning = ""
    if fedwatch:
        reasoning = f"CME FedWatch shows {fedwatch.get('cut_prob', 0)*100:.0f}% cut probability and {fedwatch.get('hold_prob', 0)*100:.0f}% hold probability for the {fedwatch.get('next_meeting', 'next')} meeting. "
        reasoning += f"Current target rate is {fedwatch.get('current_rate', 'N/A')}, implied rate is {fedwatch.get('implied_rate', 'N/A')}%. "
    if fed_markets:
        reasoning += f"Found {len(fed_markets)} related prediction markets. "
    if not reasoning:
        reasoning = "Limited Fed-specific data available for this query."
    
    return ResearchResult(
        query=query,
        category=QueryCategory.FED,
        probability=avg_prob,
        confidence=confidence,
        sources=sources,
        reasoning=reasoning.strip(),
        related_markets=[m for m in markets[:5]],
        tier="free",
    )


async def research_crypto(query: str, entities: dict) -> ResearchResult:
    """Research pipeline for crypto questions."""
    sources = []
    probabilities = []
    
    symbol = entities.get("crypto_symbol", "BTC")
    target = entities.get("target_value")
    
    # 1. Search markets
    markets = await search_markets(query)
    crypto_markets = [m for m in markets if any(
        kw in m["title"].lower() for kw in ["bitcoin", "btc", "eth", "crypto", "solana"]
    )]
    
    for m in crypto_markets[:3]:
        sources.append(Source(
            type="market",
            name=f"{m['platform']}: {m['title'][:50]}",
            probability=m["probability"],
            data={"volume": m["volume"]},
            url=m["url"],
        ))
        probabilities.append(m["probability"])
    
    # 2. Current price data
    price_data = await fetch_crypto_price(symbol)
    if price_data:
        sources.append(Source(
            type="price_data",
            name=f"Current {symbol} Price",
            data=price_data,
        ))
    
    # Calculate probability
    if probabilities:
        avg_prob = sum(probabilities) / len(probabilities)
        confidence = Confidence.HIGH if len(probabilities) >= 2 else Confidence.MEDIUM
    else:
        avg_prob = None
        confidence = Confidence.LOW
    
    # Reasoning
    reasoning = f"Analyzed {len(sources)} sources for {symbol}. "
    if crypto_markets:
        reasoning += f"Market consensus: {avg_prob*100:.0f}% probability. "
    if price_data.get("price"):
        reasoning += f"Current price: ${price_data['price']:,.0f}. "
    
    return ResearchResult(
        query=query,
        category=QueryCategory.CRYPTO,
        probability=avg_prob,
        confidence=confidence,
        sources=sources,
        reasoning=reasoning.strip(),
        related_markets=[m for m in markets[:5]],
        tier="free",
    )


async def research_politics(query: str, entities: dict) -> ResearchResult:
    """Research pipeline for politics questions."""
    sources = []
    probabilities = []
    
    # Search markets
    markets = await search_markets(query)
    
    for m in markets[:5]:
        sources.append(Source(
            type="market",
            name=f"{m['platform']}: {m['title'][:50]}",
            probability=m["probability"],
            data={"volume": m["volume"]},
            url=m["url"],
        ))
        probabilities.append(m["probability"])
    
    if probabilities:
        avg_prob = sum(probabilities) / len(probabilities)
        confidence = Confidence.HIGH if len(probabilities) >= 3 else Confidence.MEDIUM
    else:
        avg_prob = None
        confidence = Confidence.LOW
    
    reasoning = f"Political markets show {avg_prob*100:.0f}% average probability across {len(markets)} relevant markets." if avg_prob else "Limited market data available for this query."
    
    return ResearchResult(
        query=query,
        category=QueryCategory.POLITICS,
        probability=avg_prob,
        confidence=confidence,
        sources=sources,
        reasoning=reasoning,
        related_markets=[m for m in markets[:5]],
        tier="free",
    )


async def research_general(query: str, entities: dict) -> ResearchResult:
    """Fallback research for uncategorized queries."""
    markets = await search_markets(query)
    
    sources = []
    probabilities = []
    
    for m in markets[:5]:
        sources.append(Source(
            type="market",
            name=f"{m['platform']}: {m['title'][:50]}",
            probability=m["probability"],
            data={"volume": m["volume"]},
            url=m["url"],
        ))
        probabilities.append(m["probability"])
    
    if probabilities:
        avg_prob = sum(probabilities) / len(probabilities)
        confidence = Confidence.MEDIUM
    else:
        avg_prob = None
        confidence = Confidence.LOW
    
    reasoning = f"Found {len(markets)} related markets. " if markets else "No direct market matches found. "
    if avg_prob:
        reasoning += f"Average market probability: {avg_prob*100:.0f}%."
    
    return ResearchResult(
        query=query,
        category=QueryCategory.GENERAL,
        probability=avg_prob,
        confidence=confidence,
        sources=sources,
        reasoning=reasoning,
        related_markets=[m for m in markets[:5]],
        tier="free",
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM SYNTHESIS (Premium Tier)
# ══════════════════════════════════════════════════════════════════════════════

class LLMProvider:
    """Unified interface for multiple LLM providers."""
    
    def __init__(self, provider: str = "openai"):
        self.settings = get_settings()
        self.provider = provider
        self.client = httpx.AsyncClient(timeout=30.0)
    
    async def synthesize(self, query: str, sources: list[Source], category: QueryCategory) -> str:
        """Generate synthesis using LLM."""
        
        # Format sources for prompt
        sources_text = "\n".join([
            f"- {s.name}: {s.probability*100:.0f}% probability" if s.probability else f"- {s.name}: {s.data}"
            for s in sources
        ])
        
        prompt = f"""You are a prediction market analyst. Given the following question and data sources, provide a concise probability estimate and reasoning.

QUESTION: {query}

DATA SOURCES:
{sources_text}

Respond in this exact JSON format:
{{
  "probability": 0.XX,
  "confidence": "high|medium|low",
  "reasoning": "2-3 sentences explaining the probability estimate",
  "key_factors": ["factor1", "factor2"]
}}"""

        if self.provider == "openai":
            return await self._call_openai(prompt)
        elif self.provider == "anthropic":
            return await self._call_anthropic(prompt)
        elif self.provider == "groq":
            return await self._call_groq(prompt)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
    
    async def _call_openai(self, prompt: str) -> dict:
        """Call OpenAI API (GPT-4o-mini for cost efficiency)."""
        response = await self.client.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {self.settings.openai_api_key}"},
            json={
                "model": "gpt-4o-mini",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        # Parse JSON from response
        try:
            # Find JSON in response
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"probability": None, "confidence": "low", "reasoning": content}
    
    async def _call_anthropic(self, prompt: str) -> dict:
        """Call Anthropic API (Claude Haiku for cost efficiency)."""
        response = await self.client.post(
            "https://api.anthropic.com/v1/messages",
            headers={
                "x-api-key": self.settings.anthropic_api_key,
                "anthropic-version": "2023-06-01",
            },
            json={
                "model": "claude-3-5-haiku-latest",
                "max_tokens": 500,
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        response.raise_for_status()
        content = response.json()["content"][0]["text"]
        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"probability": None, "confidence": "low", "reasoning": content}
    
    async def _call_groq(self, prompt: str) -> dict:
        """Call Groq API (Llama 3.1 70B - fast and cheap)."""
        groq_key = self.settings.groq_api_key or self.settings.openai_api_key
        response = await self.client.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {groq_key}"},
            json={
                "model": "llama-3.1-70b-versatile",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.3,
                "max_tokens": 500,
            },
        )
        response.raise_for_status()
        content = response.json()["choices"][0]["message"]["content"]
        try:
            json_match = re.search(r'\{[\s\S]*\}', content)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        return {"probability": None, "confidence": "low", "reasoning": content}
    
    async def close(self):
        await self.client.aclose()


async def research_premium(query: str, provider: str = "openai") -> ResearchResult:
    """
    Premium research: LLM-powered synthesis.
    
    1. Run free tier research to gather sources
    2. Use LLM to synthesize and generate better reasoning
    """
    import time
    start = time.time()
    
    # First, run the free tier to gather data
    category = classify_intent(query)
    entities = extract_entities(query)
    
    # Get base research
    if category == QueryCategory.FED:
        base_result = await research_fed(query, entities)
    elif category == QueryCategory.CRYPTO:
        base_result = await research_crypto(query, entities)
    elif category == QueryCategory.POLITICS:
        base_result = await research_politics(query, entities)
    else:
        base_result = await research_general(query, entities)
    
    # If we have sources, enhance with LLM
    if base_result.sources:
        llm = LLMProvider(provider=provider)
        try:
            synthesis = await llm.synthesize(query, base_result.sources, category)
            
            # Update result with LLM insights
            if synthesis.get("probability"):
                base_result.probability = synthesis["probability"]
            if synthesis.get("confidence"):
                base_result.confidence = Confidence(synthesis["confidence"])
            if synthesis.get("reasoning"):
                base_result.reasoning = synthesis["reasoning"]
            
            base_result.tier = "premium"
        except Exception as e:
            print(f"LLM synthesis error: {e}")
            base_result.reasoning += f" [LLM enhancement failed: {e}]"
        finally:
            await llm.close()
    
    base_result.processing_time_ms = int((time.time() - start) * 1000)
    return base_result


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

async def research_query(
    query: str,
    tier: Literal["free", "premium"] = "free",
    llm_provider: str = "openai",
) -> ResearchResult:
    """
    Main entry point for research queries.
    
    Args:
        query: Natural language question
        tier: "free" (rule-based) or "premium" (LLM-enhanced)
        llm_provider: "openai", "anthropic", or "groq" (premium only)
    """
    import time
    start = time.time()
    
    if tier == "premium":
        return await research_premium(query, provider=llm_provider)
    
    # Free tier: rule-based routing
    category = classify_intent(query)
    entities = extract_entities(query)
    
    if category == QueryCategory.FED:
        result = await research_fed(query, entities)
    elif category == QueryCategory.CRYPTO:
        result = await research_crypto(query, entities)
    elif category == QueryCategory.POLITICS:
        result = await research_politics(query, entities)
    elif category == QueryCategory.SPORTS:
        # Sports uses market search
        result = await research_general(query, entities)
        result.category = QueryCategory.SPORTS
    elif category == QueryCategory.WEATHER:
        result = await research_general(query, entities)
        result.category = QueryCategory.WEATHER
    else:
        result = await research_general(query, entities)
    
    result.processing_time_ms = int((time.time() - start) * 1000)
    return result


# Singleton for easy access
_research_engine = None

def get_research_engine():
    global _research_engine
    if _research_engine is None:
        _research_engine = research_query
    return _research_engine
