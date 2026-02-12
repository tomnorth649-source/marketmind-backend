"""
Research Query API - Natural language research endpoint.

POST /research/query - Ask any question, get probability + sources
"""
from typing import Literal, Optional
from datetime import datetime

from fastapi import APIRouter, Query, HTTPException
from pydantic import BaseModel, Field

from app.services.research_query import (
    research_query,
    ResearchResult,
    QueryCategory,
    Confidence,
)


router = APIRouter(prefix="/research", tags=["research-query"])


class QueryRequest(BaseModel):
    """Research query request."""
    query: str = Field(..., min_length=5, max_length=500, description="Natural language question")
    tier: Literal["free", "premium"] = Field(default="free", description="free = rule-based, premium = LLM-enhanced")
    llm_provider: Literal["openai", "anthropic", "groq"] = Field(default="openai", description="LLM provider for premium tier")


class SourceResponse(BaseModel):
    """A single data source."""
    type: str
    name: str
    probability: Optional[float] = None
    data: dict = {}
    url: Optional[str] = None


class MarketResponse(BaseModel):
    """A related market."""
    platform: str
    id: str
    title: str
    probability: float
    volume: float
    url: str


class QueryResponse(BaseModel):
    """Research query response."""
    query: str
    category: str
    probability: Optional[float] = Field(None, description="Estimated probability (0-1)")
    probability_display: Optional[str] = Field(None, description="Human-readable probability")
    confidence: str = Field(description="high, medium, or low")
    sources: list[SourceResponse]
    reasoning: str
    related_markets: list[dict]
    timestamp: str
    tier: str
    processing_time_ms: int


@router.post("/query", response_model=QueryResponse)
async def query_research(request: QueryRequest):
    """
    Ask any prediction market question and get a probability estimate.
    
    **Free Tier:**
    - Rule-based intent classification
    - Searches Polymarket + Kalshi for relevant markets
    - Category-specific data (FedWatch for Fed, price data for crypto)
    - Weighted probability from multiple sources
    
    **Premium Tier:**
    - Everything in free tier
    - LLM-powered synthesis for better reasoning
    - Handles edge cases and novel queries
    - Choose provider: openai (GPT-4o-mini), anthropic (Haiku), groq (Llama 3.1)
    
    **Examples:**
    - "Will the Fed cut rates in March 2026?"
    - "Will Bitcoin hit $150k by end of 2026?"
    - "Who will win the 2026 Super Bowl?"
    - "Will Trump be re-elected in 2028?"
    """
    try:
        result = await research_query(
            query=request.query,
            tier=request.tier,
            llm_provider=request.llm_provider,
        )
        
        # Convert to response
        return QueryResponse(
            query=result.query,
            category=result.category.value,
            probability=result.probability,
            probability_display=f"{result.probability*100:.0f}%" if result.probability else None,
            confidence=result.confidence.value,
            sources=[
                SourceResponse(
                    type=s.type,
                    name=s.name,
                    probability=s.probability,
                    data=s.data,
                    url=s.url,
                )
                for s in result.sources
            ],
            reasoning=result.reasoning,
            related_markets=result.related_markets,
            timestamp=result.timestamp,
            tier=result.tier,
            processing_time_ms=result.processing_time_ms,
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Research query failed: {str(e)}")


@router.get("/query")
async def query_research_get(
    q: str = Query(..., min_length=5, max_length=500, description="Natural language question"),
    tier: Literal["free", "premium"] = Query(default="free"),
    provider: Literal["openai", "anthropic", "groq"] = Query(default="openai"),
):
    """GET version of query endpoint for easier testing."""
    request = QueryRequest(query=q, tier=tier, llm_provider=provider)
    return await query_research(request)


@router.get("/categories")
async def list_categories():
    """List supported query categories with examples."""
    return {
        "categories": [
            {
                "id": "fed",
                "name": "Federal Reserve & Rates",
                "icon": "🏛️",
                "examples": [
                    "Will the Fed cut rates in March?",
                    "What's the probability of a 50bp cut?",
                    "When will the Fed start cutting?",
                ],
            },
            {
                "id": "crypto",
                "name": "Cryptocurrency",
                "icon": "₿",
                "examples": [
                    "Will Bitcoin hit $150k by end of 2026?",
                    "Will ETH flip BTC in market cap?",
                    "Probability of Bitcoin ETF approval?",
                ],
            },
            {
                "id": "politics",
                "name": "Politics & Elections",
                "icon": "🗳️",
                "examples": [
                    "Who will win the 2028 presidential election?",
                    "Will Trump run in 2028?",
                    "Will Democrats win the Senate?",
                ],
            },
            {
                "id": "sports",
                "name": "Sports",
                "icon": "⚽",
                "examples": [
                    "Who will win the Super Bowl?",
                    "Lakers vs Celtics tonight - who wins?",
                    "Will the Chiefs repeat?",
                ],
            },
            {
                "id": "weather",
                "name": "Weather & Climate",
                "icon": "🌤️",
                "examples": [
                    "Will 2026 be the hottest year on record?",
                    "Hurricane season predictions?",
                ],
            },
            {
                "id": "general",
                "name": "General / Other",
                "icon": "🔮",
                "examples": [
                    "Will GTA 6 release in 2025?",
                    "Will AI replace programmers by 2030?",
                ],
            },
        ],
        "tiers": {
            "free": {
                "description": "Rule-based research with market data aggregation",
                "speed": "~500ms",
                "best_for": "Common questions in supported categories",
            },
            "premium": {
                "description": "LLM-enhanced synthesis with better reasoning",
                "speed": "~2-5s",
                "best_for": "Complex or novel questions, edge cases",
                "providers": ["openai", "anthropic", "groq"],
            },
        },
    }
