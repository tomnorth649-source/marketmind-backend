"""
Tiered LLM Analysis Service for MarketMind.

Cost-optimized architecture:
- Tier 1: Gemini Flash (FREE - 1500 req/day)
- Tier 2: OpenAI GPT-4o-mini (cheap fallback)
- Tier 3: Anthropic Claude (premium fallback)

All analyses are cached for 24h to minimize API calls.
"""
import os
import json
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any
import httpx

# In-memory cache (replace with Redis in production)
_analysis_cache: Dict[str, Dict[str, Any]] = {}
CACHE_TTL_HOURS = 24


def _cache_key(event_id: str) -> str:
    return f"analysis:{event_id}"


def _get_cached(event_id: str) -> Optional[Dict[str, Any]]:
    """Get cached analysis if not stale."""
    key = _cache_key(event_id)
    if key in _analysis_cache:
        cached = _analysis_cache[key]
        if datetime.utcnow() - cached["timestamp"] < timedelta(hours=CACHE_TTL_HOURS):
            return cached["data"]
    return None


def _set_cache(event_id: str, data: Dict[str, Any]):
    """Cache analysis result."""
    key = _cache_key(event_id)
    _analysis_cache[key] = {
        "data": data,
        "timestamp": datetime.utcnow()
    }


# ─── Gemini Flash (FREE TIER) ───────────────────────────────────────────────

async def analyze_with_gemini(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate analysis using Gemini 2.0 Flash.
    Free tier: 1500 requests/day, 1M tokens/day.
    """
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return None
    
    prompt = _build_analysis_prompt(event)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}",
                json={
                    "contents": [{"parts": [{"text": prompt}]}],
                    "generationConfig": {
                        "temperature": 0.7,
                        "maxOutputTokens": 1024,
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return _parse_analysis_response(text, event, "gemini")
            elif response.status_code == 429:
                print("Gemini rate limited, falling back...")
                return None
            else:
                print(f"Gemini error: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"Gemini exception: {e}")
        return None


# ─── OpenAI GPT-4o-mini (CHEAP FALLBACK) ────────────────────────────────────

async def analyze_with_openai(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate analysis using GPT-4o-mini.
    ~$0.15 per 1M input tokens, $0.60 per 1M output tokens.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return None
    
    prompt = _build_analysis_prompt(event)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": "gpt-4o-mini",
                    "messages": [
                        {"role": "system", "content": "You are a prediction market research analyst. Provide concise, data-driven analysis."},
                        {"role": "user", "content": prompt}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 1024,
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["choices"][0]["message"]["content"]
                return _parse_analysis_response(text, event, "openai")
            else:
                print(f"OpenAI error: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"OpenAI exception: {e}")
        return None


# ─── Anthropic Claude (PREMIUM FALLBACK) ────────────────────────────────────

async def analyze_with_claude(event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Generate analysis using Claude 3.5 Sonnet.
    Premium fallback for complex analysis.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    
    prompt = _build_analysis_prompt(event)
    
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json={
                    "model": "claude-3-sonnet-20240229",
                    "max_tokens": 1024,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                text = result["content"][0]["text"]
                return _parse_analysis_response(text, event, "claude")
            else:
                print(f"Claude error: {response.status_code} - {response.text}")
                return None
    except Exception as e:
        print(f"Claude exception: {e}")
        return None


# ─── Analysis Prompt Builder ────────────────────────────────────────────────

def _build_analysis_prompt(event: Dict[str, Any]) -> str:
    """Build the analysis prompt for any LLM."""
    title = event.get("title", "Unknown Event")
    question = event.get("question", title)
    category = event.get("category", "general")
    market_prob = event.get("probability", 0.5)
    if isinstance(market_prob, (int, float)) and market_prob <= 1:
        market_prob = market_prob * 100
    
    volume = event.get("volume_24h", 0)
    closes_at = event.get("closes_at", "Unknown")
    
    return f"""Analyze this prediction market event and provide a research report.

**Event:** {question}
**Category:** {category}
**Current Market Probability:** {market_prob:.1f}% YES
**24h Volume:** ${volume:,.0f}
**Closes:** {closes_at}

Provide your analysis in this EXACT JSON format:
{{
    "thesis": "2-3 sentence summary of the key dynamics and your assessment",
    "model_probability": 0.XX,
    "confidence": "high|medium|low",
    "bull_case": ["Point 1 for YES", "Point 2 for YES", "Point 3 for YES"],
    "bear_case": ["Point 1 for NO", "Point 2 for NO", "Point 3 for NO"],
    "key_data": ["Specific data point 1 to watch", "Data point 2", "Data point 3"],
    "catalysts": ["Upcoming event 1 with date if known", "Event 2"],
    "risk_factors": ["Risk 1", "Risk 2"]
}}

Be specific and data-driven. Reference concrete numbers, dates, and sources where possible.
Output ONLY valid JSON, no markdown formatting."""


def _parse_analysis_response(text: str, event: Dict[str, Any], provider: str) -> Dict[str, Any]:
    """Parse LLM response into structured analysis."""
    # Try to extract JSON from response
    try:
        # Handle markdown code blocks
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        
        data = json.loads(text.strip())
        
        # Calculate edge
        market_prob = event.get("probability", 0.5)
        if isinstance(market_prob, (int, float)) and market_prob > 1:
            market_prob = market_prob / 100
        
        model_prob = data.get("model_probability", market_prob)
        edge = model_prob - market_prob
        
        if edge > 0.05:
            edge_direction = "underpriced"
        elif edge < -0.05:
            edge_direction = "overpriced"
        else:
            edge_direction = "fair"
        
        return {
            "market_id": event.get("id", ""),
            "title": event.get("title", ""),
            "thesis": data.get("thesis", "Analysis unavailable"),
            "market_probability": market_prob,
            "model_probability": model_prob,
            "edge": edge,
            "edge_direction": edge_direction,
            "confidence": data.get("confidence", "medium"),
            "bull_case": data.get("bull_case", []),
            "bear_case": data.get("bear_case", []),
            "key_data": data.get("key_data", []),
            "catalysts": data.get("catalysts", []),
            "risk_factors": data.get("risk_factors", []),
            "provider": provider,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        # Return basic analysis from raw text
        return {
            "market_id": event.get("id", ""),
            "title": event.get("title", ""),
            "thesis": text[:500] if text else "Analysis unavailable",
            "market_probability": event.get("probability", 0.5),
            "model_probability": None,
            "edge": None,
            "edge_direction": "unknown",
            "confidence": "low",
            "bull_case": [],
            "bear_case": [],
            "key_data": [],
            "catalysts": [],
            "risk_factors": [],
            "provider": provider,
            "generated_at": datetime.utcnow().isoformat() + "Z"
        }


# ─── Main Analysis Function (Tiered) ────────────────────────────────────────

async def get_event_analysis(event: Dict[str, Any], force_refresh: bool = False) -> Dict[str, Any]:
    """
    Get analysis for an event using tiered LLM approach.
    
    Order: Cache → Gemini (free) → OpenAI (cheap) → Claude (premium)
    """
    event_id = event.get("id", "")
    
    # Check cache first
    if not force_refresh:
        cached = _get_cached(event_id)
        if cached:
            cached["from_cache"] = True
            return cached
    
    # Tier 1: Gemini Flash (FREE)
    analysis = await analyze_with_gemini(event)
    if analysis:
        _set_cache(event_id, analysis)
        return analysis
    
    # Tier 2: OpenAI GPT-4o-mini (cheap)
    analysis = await analyze_with_openai(event)
    if analysis:
        _set_cache(event_id, analysis)
        return analysis
    
    # Tier 3: Claude (premium fallback)
    analysis = await analyze_with_claude(event)
    if analysis:
        _set_cache(event_id, analysis)
        return analysis
    
    # All failed - return basic template
    return {
        "market_id": event_id,
        "title": event.get("title", ""),
        "thesis": "Analysis temporarily unavailable. Please try again later.",
        "market_probability": event.get("probability", 0.5),
        "model_probability": None,
        "edge": None,
        "edge_direction": "unknown",
        "confidence": "low",
        "bull_case": [],
        "bear_case": [],
        "key_data": [],
        "catalysts": [],
        "risk_factors": [],
        "provider": "none",
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }


async def get_batch_analysis(events: list[Dict[str, Any]]) -> list[Dict[str, Any]]:
    """Get analysis for multiple events."""
    import asyncio
    tasks = [get_event_analysis(e) for e in events]
    return await asyncio.gather(*tasks)
