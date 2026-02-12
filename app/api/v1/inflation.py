"""Inflation / CPI Research API endpoints."""
from fastapi import APIRouter, HTTPException

from app.config import get_settings
from app.services.research.inflation import get_inflation_module

router = APIRouter(prefix="/inflation", tags=["inflation-research"])


@router.get("/dashboard")
async def get_inflation_dashboard():
    """
    Get comprehensive inflation analysis dashboard.
    
    Includes:
    - Latest CPI release (headline & core, MoM & YoY)
    - Next release date
    - Forecast with confidence interval
    - Key components (shelter, energy, food)
    - Market expectations (breakevens)
    """
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")
    
    module = get_inflation_module(fred_key)
    
    try:
        return await module.get_inflation_dashboard()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/latest")
async def get_latest_cpi():
    """Get most recent CPI release data."""
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")
    
    module = get_inflation_module(fred_key)
    
    try:
        cpi = await module.get_latest_cpi()
        if not cpi:
            raise HTTPException(status_code=404, detail="No CPI data available")
        
        return {
            "date": cpi.date,
            "headline": {
                "yoy": f"{cpi.headline_yoy:.1f}%",
                "mom": f"{cpi.headline_mom:.2f}%",
            },
            "core": {
                "yoy": f"{cpi.core_yoy:.1f}%",
                "mom": f"{cpi.core_mom:.2f}%",
            },
            "fed_target": "2.0%",
            "above_target": cpi.core_yoy > 2.0,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/history")
async def get_cpi_history(months: int = 12):
    """Get CPI history for the past N months."""
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")
    
    module = get_inflation_module(fred_key)
    
    try:
        history = await module.get_cpi_history(months)
        return {
            "releases": [
                {
                    "date": r.date,
                    "headline_yoy": f"{r.headline_yoy:.1f}%",
                    "core_yoy": f"{r.core_yoy:.1f}%",
                }
                for r in history
            ],
            "count": len(history),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast")
async def get_cpi_forecast():
    """
    Get forecast for next CPI release.
    
    Returns headline and core YoY estimates with confidence range.
    """
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")
    
    module = get_inflation_module(fred_key)
    
    try:
        forecast = await module.forecast_cpi()
        if not forecast:
            raise HTTPException(status_code=404, detail="Unable to generate forecast")
        
        return {
            "release_date": forecast.release_date,
            "headline_yoy": {
                "forecast": f"{forecast.headline_yoy_forecast:.1f}%",
                "range": f"{forecast.headline_yoy_range[0]:.1f}% - {forecast.headline_yoy_range[1]:.1f}%",
            },
            "core_yoy": {
                "forecast": f"{forecast.core_yoy_forecast:.1f}%",
                "range": f"{forecast.core_yoy_range[0]:.1f}% - {forecast.core_yoy_range[1]:.1f}%",
            },
            "confidence": forecast.confidence,
            "factors": forecast.factors,
            "methodology": forecast.methodology,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/expectations")
async def get_inflation_expectations():
    """Get market and consumer inflation expectations."""
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")
    
    module = get_inflation_module(fred_key)
    
    try:
        return await module.get_inflation_expectations()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/components")
async def get_cpi_components():
    """Get breakdown of CPI components (shelter, energy, food)."""
    settings = get_settings()
    fred_key = getattr(settings, 'fred_api_key', None)
    
    if not fred_key:
        raise HTTPException(status_code=503, detail="FRED API key not configured")
    
    module = get_inflation_module(fred_key)
    
    try:
        return await module.get_cpi_components()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
