"""Weather Research Module.

Provides probability estimates for:
- Temperature brackets (high/low for cities)
- Snowfall totals
- Hurricane landfalls and categories
- Extreme weather events

Data sources:
- NOAA/NWS API (free, no key required)
- OpenWeather API (backup)
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import httpx


@dataclass
class WeatherForecast:
    """Weather forecast for a location."""
    location: str
    date: str
    high_temp: Optional[float]  # Fahrenheit
    low_temp: Optional[float]
    precipitation_chance: Optional[float]  # 0-100
    conditions: str
    wind_speed: Optional[float]  # mph
    

@dataclass
class SnowForecast:
    """Snowfall forecast."""
    location: str
    period: str  # "next 24h", "next 7 days", etc.
    expected_inches: float
    range_low: float
    range_high: float
    confidence: str


@dataclass
class HurricaneInfo:
    """Hurricane tracking information."""
    name: str
    category: int  # 1-5
    max_winds: int  # mph
    location: str
    movement: str
    forecast_track: list[dict]


class WeatherModule:
    """Research module for weather predictions."""
    
    # NWS API (free, no key required)
    NWS_BASE_URL = "https://api.weather.gov"
    
    # Major cities with NWS grid points
    CITIES = {
        "nyc": {"name": "New York City", "lat": 40.7128, "lon": -74.0060},
        "chicago": {"name": "Chicago", "lat": 41.8781, "lon": -87.6298},
        "miami": {"name": "Miami", "lat": 25.7617, "lon": -80.1918},
        "denver": {"name": "Denver", "lat": 39.7392, "lon": -104.9903},
        "philadelphia": {"name": "Philadelphia", "lat": 39.9526, "lon": -75.1652},
        "boston": {"name": "Boston", "lat": 42.3601, "lon": -71.0589},
        "houston": {"name": "Houston", "lat": 29.7604, "lon": -95.3698},
        "phoenix": {"name": "Phoenix", "lat": 33.4484, "lon": -112.0740},
        "los_angeles": {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437},
        "dallas": {"name": "Dallas", "lat": 32.7767, "lon": -96.7970},
    }
    
    def __init__(self):
        self._cache: dict = {}
        self._cache_ttl = timedelta(hours=1)
        self._grid_cache: dict = {}
    
    async def _nws_request(self, endpoint: str) -> dict:
        """Make request to NWS API."""
        headers = {
            "User-Agent": "MarketMind/1.0 (weather@marketmind.ai)",
            "Accept": "application/geo+json",
        }
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.NWS_BASE_URL}{endpoint}",
                headers=headers,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def _get_grid_point(self, lat: float, lon: float) -> tuple[str, int, int]:
        """Get NWS grid point for coordinates."""
        cache_key = f"grid:{lat},{lon}"
        if cache_key in self._grid_cache:
            return self._grid_cache[cache_key]
        
        data = await self._nws_request(f"/points/{lat},{lon}")
        props = data.get("properties", {})
        
        grid_id = props.get("gridId")
        grid_x = props.get("gridX")
        grid_y = props.get("gridY")
        
        self._grid_cache[cache_key] = (grid_id, grid_x, grid_y)
        return grid_id, grid_x, grid_y
    
    async def get_forecast(self, city_code: str) -> list[WeatherForecast]:
        """Get 7-day forecast for a city."""
        if city_code not in self.CITIES:
            raise ValueError(f"Unknown city: {city_code}. Available: {list(self.CITIES.keys())}")
        
        city = self.CITIES[city_code]
        
        # Get grid point
        grid_id, grid_x, grid_y = await self._get_grid_point(city["lat"], city["lon"])
        
        # Get forecast
        data = await self._nws_request(f"/gridpoints/{grid_id}/{grid_x},{grid_y}/forecast")
        periods = data.get("properties", {}).get("periods", [])
        
        forecasts = []
        for period in periods[:14]:  # 7 days (day/night pairs)
            forecasts.append(WeatherForecast(
                location=city["name"],
                date=period.get("startTime", "")[:10],
                high_temp=period.get("temperature") if "Day" in period.get("name", "") or not period.get("isDaytime") == False else None,
                low_temp=period.get("temperature") if period.get("isDaytime") == False else None,
                precipitation_chance=period.get("probabilityOfPrecipitation", {}).get("value"),
                conditions=period.get("shortForecast", ""),
                wind_speed=None,  # Would need parsing
            ))
        
        return forecasts
    
    async def get_current_conditions(self, city_code: str) -> dict:
        """Get current weather conditions for a city."""
        if city_code not in self.CITIES:
            raise ValueError(f"Unknown city: {city_code}")
        
        city = self.CITIES[city_code]
        
        # Get observation stations
        grid_id, grid_x, grid_y = await self._get_grid_point(city["lat"], city["lon"])
        
        # Get stations
        stations_data = await self._nws_request(f"/gridpoints/{grid_id}/{grid_x},{grid_y}/stations")
        stations = stations_data.get("features", [])
        
        if not stations:
            return None
        
        # Get latest observation from first station
        station_id = stations[0]["properties"]["stationIdentifier"]
        obs_data = await self._nws_request(f"/stations/{station_id}/observations/latest")
        props = obs_data.get("properties", {})
        
        temp_c = props.get("temperature", {}).get("value")
        temp_f = (temp_c * 9/5 + 32) if temp_c is not None else None
        
        return {
            "location": city["name"],
            "temperature_f": round(temp_f, 1) if temp_f else None,
            "conditions": props.get("textDescription"),
            "humidity": props.get("relativeHumidity", {}).get("value"),
            "wind_speed_mph": props.get("windSpeed", {}).get("value"),
            "observed_at": props.get("timestamp"),
        }
    
    async def get_alerts(self, state: str = None) -> list[dict]:
        """Get active weather alerts."""
        endpoint = "/alerts/active"
        if state:
            endpoint += f"?area={state.upper()}"
        
        data = await self._nws_request(endpoint)
        features = data.get("features", [])
        
        alerts = []
        for feature in features[:20]:  # Limit to 20
            props = feature.get("properties", {})
            alerts.append({
                "event": props.get("event"),
                "headline": props.get("headline"),
                "severity": props.get("severity"),
                "urgency": props.get("urgency"),
                "areas": props.get("areaDesc"),
                "effective": props.get("effective"),
                "expires": props.get("expires"),
            })
        
        return alerts
    
    async def estimate_snow_probability(
        self, 
        city_code: str,
        threshold_inches: float,
        days_ahead: int = 7,
    ) -> dict:
        """
        Estimate probability of snowfall exceeding threshold.
        
        Used for Kalshi snow markets.
        """
        forecasts = await self.get_forecast(city_code)
        city = self.CITIES[city_code]
        
        # Check for snow in forecast
        snow_days = []
        for f in forecasts[:days_ahead * 2]:  # Day/night pairs
            conditions = f.conditions.lower()
            if "snow" in conditions:
                snow_days.append({
                    "date": f.date,
                    "conditions": f.conditions,
                    "precip_chance": f.precipitation_chance,
                })
        
        # Estimate probability based on forecast
        if not snow_days:
            probability = 0.05  # Base rate for surprise snow
            confidence = "high"
        else:
            # More snow days = higher probability
            avg_precip_chance = sum(d.get("precip_chance") or 50 for d in snow_days) / len(snow_days)
            
            # Rough conversion: chance of ANY snow → chance of exceeding threshold
            # This is simplified - real model would use ensemble data
            if threshold_inches <= 1:
                probability = avg_precip_chance / 100 * 0.9
            elif threshold_inches <= 3:
                probability = avg_precip_chance / 100 * 0.6
            elif threshold_inches <= 6:
                probability = avg_precip_chance / 100 * 0.35
            else:
                probability = avg_precip_chance / 100 * 0.15
            
            confidence = "medium" if len(snow_days) > 0 else "low"
        
        return {
            "location": city["name"],
            "threshold_inches": threshold_inches,
            "days_ahead": days_ahead,
            "probability": round(probability, 2),
            "probability_display": f"{probability*100:.0f}%",
            "confidence": confidence,
            "snow_in_forecast": len(snow_days) > 0,
            "forecast_details": snow_days,
        }
    
    async def estimate_temp_probability(
        self,
        city_code: str,
        threshold_temp: float,
        above_or_below: str,  # "above" or "below"
        days_ahead: int = 7,
    ) -> dict:
        """
        Estimate probability of temperature exceeding/below threshold.
        
        Used for Kalshi temperature markets.
        """
        forecasts = await self.get_forecast(city_code)
        city = self.CITIES[city_code]
        
        # Extract high temps
        high_temps = []
        low_temps = []
        
        for f in forecasts[:days_ahead * 2]:
            if f.high_temp is not None:
                high_temps.append(f.high_temp)
            if f.low_temp is not None:
                low_temps.append(f.low_temp)
        
        if above_or_below == "above":
            # Check if any high exceeds threshold
            exceeds = [t for t in high_temps if t >= threshold_temp]
            if high_temps:
                probability = len(exceeds) / len(high_temps)
                max_temp = max(high_temps)
                margin = max_temp - threshold_temp
            else:
                probability = 0.5
                margin = 0
        else:  # below
            # Check if any low is below threshold
            below = [t for t in low_temps if t <= threshold_temp]
            if low_temps:
                probability = len(below) / len(low_temps)
                min_temp = min(low_temps)
                margin = threshold_temp - min_temp
            else:
                probability = 0.5
                margin = 0
        
        # Adjust confidence based on margin
        if abs(margin) > 10:
            confidence = "high"
        elif abs(margin) > 5:
            confidence = "medium"
        else:
            confidence = "low"
        
        return {
            "location": city["name"],
            "threshold_f": threshold_temp,
            "direction": above_or_below,
            "days_ahead": days_ahead,
            "probability": round(probability, 2),
            "probability_display": f"{probability*100:.0f}%",
            "confidence": confidence,
            "forecast_high": max(high_temps) if high_temps else None,
            "forecast_low": min(low_temps) if low_temps else None,
        }
    
    async def get_weather_dashboard(self, city_code: str = "nyc") -> dict:
        """Get comprehensive weather dashboard for a city."""
        forecasts = await self.get_forecast(city_code)
        current = await self.get_current_conditions(city_code)
        city = self.CITIES[city_code]
        
        # Extract next 3 days
        daily_forecasts = []
        for i in range(0, min(6, len(forecasts)), 2):
            day = forecasts[i] if i < len(forecasts) else None
            night = forecasts[i+1] if i+1 < len(forecasts) else None
            
            daily_forecasts.append({
                "date": day.date if day else None,
                "high": day.high_temp if day else None,
                "low": night.low_temp if night else None,
                "conditions": day.conditions if day else None,
            })
        
        return {
            "location": city["name"],
            "current": current,
            "forecast": daily_forecasts,
            "available_cities": list(self.CITIES.keys()),
        }


# Singleton
_module: WeatherModule | None = None

def get_weather_module() -> WeatherModule:
    global _module
    if _module is None:
        _module = WeatherModule()
    return _module
