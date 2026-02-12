"""Rate Limiting & Caching Strategy.

Solves API rate limits for all data sources at scale.
"""
from datetime import timedelta

# =============================================================================
# RATE LIMIT SOLUTIONS - LOCKED IN
# =============================================================================

API_LIMITS = {
    "fred": {
        "limit": "120 requests/minute",
        "our_solution": "1-hour cache (data changes daily at most)",
        "shared_cache": True,  # All users get same Fed data
        "risk": "NONE - well under limit with caching",
    },
    "nws": {
        "limit": "No hard limit, 'reasonable use'",
        "our_solution": "1-hour cache for forecasts",
        "shared_cache": True,  # Weather is same for all users
        "risk": "NONE",
    },
    "coingecko": {
        "limit": "10-50 requests/minute (free tier)",
        "our_solution": "2-minute cache + rate limiter middleware",
        "shared_cache": True,  # Prices same for all
        "risk": "LOW - may need Pro tier ($129/mo) at scale",
        "fallback": "CoinMarketCap API as backup",
    },
    "espn": {
        "limit": "Unofficial API, no documented limit",
        "our_solution": "5-minute cache",
        "shared_cache": True,
        "risk": "LOW - could break if ESPN changes API",
    },
    "kalshi": {
        "limit": "Unknown, likely generous",
        "our_solution": "1-hour cache for market list, real-time for prices",
        "shared_cache": True,
        "risk": "NONE",
    },
    "polymarket": {
        "limit": "Unknown, generous for reads",
        "our_solution": "5-minute cache",
        "shared_cache": True,
        "risk": "NONE",
    },
}

# Cache TTLs
CACHE_TTL = {
    "fred_series": timedelta(hours=1),      # Economic data
    "weather_forecast": timedelta(hours=1), # NWS forecasts
    "crypto_price": timedelta(minutes=2),   # Fast-moving
    "sports_games": timedelta(minutes=5),   # Game updates
    "market_list": timedelta(hours=1),      # Kalshi/Poly markets
    "market_price": timedelta(seconds=30),  # Real-time prices
    "arb_scan": timedelta(minutes=1),       # Arb opportunities
}

# User rate limits (requests per minute)
USER_RATE_LIMITS = {
    "free": 10,    # 10 req/min
    "paid": 60,    # 60 req/min (1/sec)
}

# =============================================================================
# IMPLEMENTATION
# =============================================================================

class RateLimitConfig:
    """Rate limiting configuration."""
    
    @staticmethod
    def get_cache_key(endpoint: str, params: dict = None) -> str:
        """Generate cache key for endpoint."""
        base = endpoint.replace("/", ":")
        if params:
            param_str = ":".join(f"{k}={v}" for k, v in sorted(params.items()))
            return f"{base}:{param_str}"
        return base
    
    @staticmethod
    def get_ttl(endpoint: str) -> timedelta:
        """Get cache TTL for endpoint."""
        if "fred" in endpoint or "fed" in endpoint:
            return CACHE_TTL["fred_series"]
        elif "weather" in endpoint:
            return CACHE_TTL["weather_forecast"]
        elif "crypto" in endpoint:
            return CACHE_TTL["crypto_price"]
        elif "sports" in endpoint:
            return CACHE_TTL["sports_games"]
        elif "arb" in endpoint:
            return CACHE_TTL["arb_scan"]
        else:
            return CACHE_TTL["market_list"]


# For Redis implementation (production)
REDIS_CONFIG = {
    "enabled": False,  # Enable when deploying
    "url": "redis://localhost:6379/0",
    "prefix": "marketmind:",
}
