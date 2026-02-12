"""Main API v1 router."""
from fastapi import APIRouter

from app.api.v1 import (
    auth, events, research, health, markets, polymarket, arb, 
    fed, inflation, weather, sports, crypto, politics,
    fedwatch, calibration, opportunities, query,
)

router = APIRouter()

router.include_router(health.router, tags=["health"])
router.include_router(query.router)  # Research Query Engine
router.include_router(auth.router, prefix="/auth", tags=["auth"])
router.include_router(events.router, prefix="/events", tags=["events"])
router.include_router(research.router, prefix="/research", tags=["research"])
router.include_router(markets.router, prefix="/markets", tags=["markets"])
router.include_router(polymarket.router, tags=["polymarket"])
router.include_router(arb.router, tags=["arbitrage"])
router.include_router(opportunities.router, tags=["opportunities"])  # Main dashboard
router.include_router(fed.router, tags=["fed-research"])
router.include_router(fedwatch.router, tags=["fedwatch"])  # CME FedWatch integration
router.include_router(calibration.router, tags=["calibration"])  # Prediction tracking
router.include_router(inflation.router, tags=["inflation-research"])
router.include_router(weather.router, tags=["weather-research"])
router.include_router(sports.router, tags=["sports-research"])
router.include_router(crypto.router, tags=["crypto-research"])
router.include_router(politics.router, tags=["politics-research"])
