"""Weather Research API endpoints."""
from typing import Optional

from fastapi import APIRouter, HTTPException, Query

from app.services.research.weather import get_weather_module

router = APIRouter(prefix="/weather", tags=["weather-research"])


@router.get("/dashboard/{city}")
async def get_weather_dashboard(city: str):
    """
    Get weather dashboard for a city.
    
    Available cities: nyc, chicago, miami, denver, philadelphia, 
    boston, houston, phoenix, los_angeles, dallas
    """
    module = get_weather_module()
    
    try:
        return await module.get_weather_dashboard(city.lower())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/current/{city}")
async def get_current_conditions(city: str):
    """Get current weather conditions for a city."""
    module = get_weather_module()
    
    try:
        conditions = await module.get_current_conditions(city.lower())
        if not conditions:
            raise HTTPException(status_code=404, detail="No observation data available")
        return conditions
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/forecast/{city}")
async def get_forecast(city: str):
    """Get 7-day forecast for a city."""
    module = get_weather_module()
    
    try:
        forecasts = await module.get_forecast(city.lower())
        return {
            "location": city,
            "periods": [
                {
                    "date": f.date,
                    "high": f.high_temp,
                    "low": f.low_temp,
                    "conditions": f.conditions,
                    "precip_chance": f.precipitation_chance,
                }
                for f in forecasts
            ],
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/alerts")
async def get_weather_alerts(state: Optional[str] = Query(default=None, description="2-letter state code")):
    """Get active weather alerts, optionally filtered by state."""
    module = get_weather_module()
    
    try:
        alerts = await module.get_alerts(state)
        return {
            "state": state,
            "count": len(alerts),
            "alerts": alerts,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability/snow")
async def get_snow_probability(
    city: str = Query(..., description="City code (e.g., nyc, chicago)"),
    threshold: float = Query(..., description="Snow threshold in inches"),
    days: int = Query(default=7, ge=1, le=14, description="Days ahead to forecast"),
):
    """
    Estimate probability of snowfall exceeding threshold.
    
    Used for Kalshi snow markets like:
    - "Will NYC get 3+ inches of snow this week?"
    """
    module = get_weather_module()
    
    try:
        return await module.estimate_snow_probability(city.lower(), threshold, days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/probability/temperature")
async def get_temp_probability(
    city: str = Query(..., description="City code (e.g., nyc, miami)"),
    threshold: float = Query(..., description="Temperature threshold in Fahrenheit"),
    direction: str = Query(..., description="'above' or 'below' threshold"),
    days: int = Query(default=7, ge=1, le=14, description="Days ahead to forecast"),
):
    """
    Estimate probability of temperature above/below threshold.
    
    Used for Kalshi temperature markets like:
    - "Will Miami exceed 95°F this week?"
    - "Will Denver drop below 10°F?"
    """
    if direction not in ["above", "below"]:
        raise HTTPException(status_code=400, detail="direction must be 'above' or 'below'")
    
    module = get_weather_module()
    
    try:
        return await module.estimate_temp_probability(city.lower(), threshold, direction, days)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cities")
async def list_available_cities():
    """List all available cities for weather data."""
    module = get_weather_module()
    return {
        "cities": [
            {"code": code, "name": info["name"]}
            for code, info in module.CITIES.items()
        ]
    }
