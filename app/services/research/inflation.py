"""Inflation / CPI Research Module.

Provides probability estimates for:
- CPI releases (headline, core, MoM, YoY)
- PCE inflation
- Inflation bracket predictions (above/below X%)

Data sources:
- FRED API (BLS CPI data)
- Cleveland Fed Inflation Nowcast
- Truflation (real-time daily estimate)
"""
import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional
import httpx


@dataclass
class CPIRelease:
    """CPI release data point."""
    date: str
    headline_yoy: float  # All items YoY %
    headline_mom: float  # All items MoM %
    core_yoy: float      # Ex food/energy YoY %
    core_mom: float      # Ex food/energy MoM %
    

@dataclass
class InflationForecast:
    """Forecast for upcoming CPI release."""
    release_date: str
    headline_yoy_forecast: float
    headline_yoy_range: tuple[float, float]  # Low, high
    core_yoy_forecast: float
    core_yoy_range: tuple[float, float]
    confidence: str
    factors: list[dict]
    methodology: str


@dataclass 
class InflationNowcast:
    """Real-time inflation estimate."""
    date: str
    headline_estimate: float
    core_estimate: float
    source: str
    updated_at: datetime


class InflationModule:
    """Research module for inflation and CPI analysis."""
    
    FRED_BASE_URL = "https://api.stlouisfed.org/fred"
    CLEVELAND_FED_URL = "https://www.clevelandfed.org/indicators-and-data/inflation-nowcasting"
    
    # Key FRED series for inflation
    FRED_SERIES = {
        # CPI
        "CPIAUCSL": "CPI All Urban Consumers (headline)",
        "CPILFESL": "CPI Less Food and Energy (core)",
        "CPIUFDSL": "CPI Food",
        "CPIENGSL": "CPI Energy",
        
        # Components
        "CUSR0000SAH1": "CPI Shelter",
        "CUSR0000SETB01": "CPI Gasoline",
        "CUSR0000SAF11": "CPI Food at Home",
        
        # PCE (Fed's preferred)
        "PCEPI": "PCE Price Index",
        "PCEPILFE": "PCE Less Food and Energy (Core PCE)",
        
        # Expectations
        "MICH": "Michigan Inflation Expectations",
        "T5YIE": "5-Year Breakeven Inflation",
        "T10YIE": "10-Year Breakeven Inflation",
    }
    
    # CPI release schedule (BLS releases ~10th-15th of each month)
    CPI_RELEASE_DATES_2026 = [
        "2026-01-14", "2026-02-12", "2026-03-12", "2026-04-10",
        "2026-05-13", "2026-06-11", "2026-07-15", "2026-08-12",
        "2026-09-11", "2026-10-13", "2026-11-12", "2026-12-10",
    ]
    
    def __init__(self, fred_api_key: str = None):
        self.fred_api_key = fred_api_key
        self._cache: dict = {}
        self._cache_ttl = timedelta(hours=1)
    
    async def _fred_request(self, endpoint: str, params: dict) -> dict:
        """Make request to FRED API."""
        if not self.fred_api_key:
            raise ValueError("FRED API key required")
        
        params["api_key"] = self.fred_api_key
        params["file_type"] = "json"
        
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{self.FRED_BASE_URL}/{endpoint}",
                params=params,
                timeout=30.0,
            )
            response.raise_for_status()
            return response.json()
    
    async def get_series(self, series_id: str, limit: int = 24) -> list[dict]:
        """Get recent observations for a FRED series."""
        cache_key = f"fred_infl:{series_id}"
        if cache_key in self._cache:
            cached, cached_at = self._cache[cache_key]
            if datetime.now() - cached_at < self._cache_ttl:
                return cached
        
        data = await self._fred_request("series/observations", {
            "series_id": series_id,
            "sort_order": "desc",
            "limit": limit,
        })
        
        observations = data.get("observations", [])
        self._cache[cache_key] = (observations, datetime.now())
        return observations
    
    def _calc_yoy_change(self, current: float, year_ago: float) -> float:
        """Calculate year-over-year percentage change."""
        if year_ago == 0:
            return 0
        return ((current / year_ago) - 1) * 100
    
    def _calc_mom_change(self, current: float, previous: float) -> float:
        """Calculate month-over-month percentage change."""
        if previous == 0:
            return 0
        return ((current / previous) - 1) * 100
    
    async def get_latest_cpi(self) -> CPIRelease:
        """Get most recent CPI release data."""
        # Get headline CPI (need 13 months for YoY)
        headline_obs = await self.get_series("CPIAUCSL", limit=13)
        core_obs = await self.get_series("CPILFESL", limit=13)
        
        if not headline_obs or not core_obs:
            return None
        
        # Calculate changes
        headline_current = float(headline_obs[0]["value"])
        headline_prev = float(headline_obs[1]["value"])
        headline_year_ago = float(headline_obs[12]["value"])
        
        core_current = float(core_obs[0]["value"])
        core_prev = float(core_obs[1]["value"])
        core_year_ago = float(core_obs[12]["value"])
        
        return CPIRelease(
            date=headline_obs[0]["date"],
            headline_yoy=round(self._calc_yoy_change(headline_current, headline_year_ago), 2),
            headline_mom=round(self._calc_mom_change(headline_current, headline_prev), 2),
            core_yoy=round(self._calc_yoy_change(core_current, core_year_ago), 2),
            core_mom=round(self._calc_mom_change(core_current, core_prev), 2),
        )
    
    async def get_cpi_history(self, months: int = 12) -> list[CPIRelease]:
        """Get CPI history for the past N months."""
        headline_obs = await self.get_series("CPIAUCSL", limit=months + 12)
        core_obs = await self.get_series("CPILFESL", limit=months + 12)
        
        releases = []
        for i in range(months):
            if i + 12 >= len(headline_obs) or i + 12 >= len(core_obs):
                break
            
            headline_current = float(headline_obs[i]["value"])
            headline_prev = float(headline_obs[i + 1]["value"])
            headline_year_ago = float(headline_obs[i + 12]["value"])
            
            core_current = float(core_obs[i]["value"])
            core_prev = float(core_obs[i + 1]["value"])
            core_year_ago = float(core_obs[i + 12]["value"])
            
            releases.append(CPIRelease(
                date=headline_obs[i]["date"],
                headline_yoy=round(self._calc_yoy_change(headline_current, headline_year_ago), 2),
                headline_mom=round(self._calc_mom_change(headline_current, headline_prev), 2),
                core_yoy=round(self._calc_yoy_change(core_current, core_year_ago), 2),
                core_mom=round(self._calc_mom_change(core_current, core_prev), 2),
            ))
        
        return releases
    
    async def get_inflation_expectations(self) -> dict:
        """Get market and consumer inflation expectations."""
        results = {}
        
        # Michigan Consumer Survey expectations
        mich = await self.get_series("MICH", limit=2)
        if mich and mich[0]["value"] != ".":
            results["consumer_1yr"] = {
                "value": float(mich[0]["value"]),
                "date": mich[0]["date"],
                "source": "Michigan Consumer Survey",
            }
        
        # 5-Year breakeven
        t5yie = await self.get_series("T5YIE", limit=5)
        for obs in t5yie:
            if obs["value"] != ".":
                results["market_5yr"] = {
                    "value": float(obs["value"]),
                    "date": obs["date"],
                    "source": "5-Year Treasury Breakeven",
                }
                break
        
        # 10-Year breakeven
        t10yie = await self.get_series("T10YIE", limit=5)
        for obs in t10yie:
            if obs["value"] != ".":
                results["market_10yr"] = {
                    "value": float(obs["value"]),
                    "date": obs["date"],
                    "source": "10-Year Treasury Breakeven",
                }
                break
        
        return results
    
    async def get_cpi_components(self) -> dict:
        """Get breakdown of CPI components."""
        components = {}
        
        series_map = {
            "shelter": "CUSR0000SAH1",
            "food_at_home": "CUSR0000SAF11", 
            "energy": "CPIENGSL",
        }
        
        for name, series_id in series_map.items():
            try:
                obs = await self.get_series(series_id, limit=13)
                if obs and len(obs) >= 13:
                    current = float(obs[0]["value"])
                    year_ago = float(obs[12]["value"])
                    yoy = self._calc_yoy_change(current, year_ago)
                    
                    components[name] = {
                        "yoy_change": round(yoy, 2),
                        "date": obs[0]["date"],
                    }
            except Exception:
                continue
        
        return components
    
    def get_next_cpi_release(self) -> Optional[str]:
        """Get date of next CPI release."""
        today = datetime.now().date()
        for date_str in self.CPI_RELEASE_DATES_2026:
            release_date = datetime.strptime(date_str, "%Y-%m-%d").date()
            if release_date > today:
                return date_str
        return None
    
    async def forecast_cpi(self) -> InflationForecast:
        """
        Generate CPI forecast for next release.
        
        Methodology:
        1. Look at recent trend (last 3-6 months)
        2. Factor in known components (energy prices, shelter)
        3. Consider market expectations
        4. Generate range estimate
        """
        # Get historical data
        history = await self.get_cpi_history(6)
        expectations = await self.get_inflation_expectations()
        components = await self.get_cpi_components()
        
        if not history:
            return None
        
        # Calculate trend
        recent_headline = [h.headline_yoy for h in history[:3]]
        recent_core = [h.core_yoy for h in history[:3]]
        
        avg_headline = sum(recent_headline) / len(recent_headline)
        avg_core = sum(recent_core) / len(recent_core)
        
        # Trend direction
        headline_trend = recent_headline[0] - recent_headline[-1]
        core_trend = recent_core[0] - recent_core[-1]
        
        factors = []
        
        # Energy impact
        if "energy" in components:
            energy_yoy = components["energy"]["yoy_change"]
            if energy_yoy > 5:
                factors.append({
                    "name": "Energy prices",
                    "value": f"+{energy_yoy:.1f}% YoY",
                    "impact": "upward pressure on headline",
                })
            elif energy_yoy < -5:
                factors.append({
                    "name": "Energy prices", 
                    "value": f"{energy_yoy:.1f}% YoY",
                    "impact": "downward pressure on headline",
                })
        
        # Shelter impact (sticky component)
        if "shelter" in components:
            shelter_yoy = components["shelter"]["yoy_change"]
            factors.append({
                "name": "Shelter costs",
                "value": f"+{shelter_yoy:.1f}% YoY",
                "impact": "sticky upward pressure" if shelter_yoy > 3 else "moderating",
            })
        
        # Market expectations
        if "market_5yr" in expectations:
            mkt_exp = expectations["market_5yr"]["value"]
            factors.append({
                "name": "Market expectations (5Y breakeven)",
                "value": f"{mkt_exp:.2f}%",
                "impact": "anchored" if 2.0 <= mkt_exp <= 2.5 else "elevated",
            })
        
        # Generate forecast
        # Base: recent average with trend adjustment
        headline_forecast = avg_headline + (headline_trend * 0.3)
        core_forecast = avg_core + (core_trend * 0.3)
        
        # Range: ±0.3% based on historical volatility
        headline_range = (headline_forecast - 0.3, headline_forecast + 0.3)
        core_range = (core_forecast - 0.2, core_forecast + 0.2)
        
        # Confidence based on component stability
        if abs(headline_trend) < 0.2 and abs(core_trend) < 0.2:
            confidence = "high"
        elif abs(headline_trend) < 0.5:
            confidence = "medium"
        else:
            confidence = "low"
        
        return InflationForecast(
            release_date=self.get_next_cpi_release(),
            headline_yoy_forecast=round(headline_forecast, 2),
            headline_yoy_range=(round(headline_range[0], 2), round(headline_range[1], 2)),
            core_yoy_forecast=round(core_forecast, 2),
            core_yoy_range=(round(core_range[0], 2), round(core_range[1], 2)),
            confidence=confidence,
            factors=factors,
            methodology="Trend extrapolation with component adjustment",
        )
    
    async def get_inflation_dashboard(self) -> dict:
        """Get comprehensive inflation dashboard."""
        latest = await self.get_latest_cpi()
        forecast = await self.forecast_cpi()
        expectations = await self.get_inflation_expectations()
        components = await self.get_cpi_components()
        
        return {
            "latest_cpi": {
                "date": latest.date if latest else None,
                "headline_yoy": f"{latest.headline_yoy:.1f}%" if latest else None,
                "headline_mom": f"{latest.headline_mom:.2f}%" if latest else None,
                "core_yoy": f"{latest.core_yoy:.1f}%" if latest else None,
                "core_mom": f"{latest.core_mom:.2f}%" if latest else None,
            } if latest else None,
            "next_release": self.get_next_cpi_release(),
            "forecast": {
                "headline_yoy": f"{forecast.headline_yoy_forecast:.1f}%",
                "headline_range": f"{forecast.headline_yoy_range[0]:.1f}% - {forecast.headline_yoy_range[1]:.1f}%",
                "core_yoy": f"{forecast.core_yoy_forecast:.1f}%",
                "core_range": f"{forecast.core_yoy_range[0]:.1f}% - {forecast.core_yoy_range[1]:.1f}%",
                "confidence": forecast.confidence,
            } if forecast else None,
            "components": components,
            "expectations": expectations,
            "fed_target": "2.0%",
        }


# Singleton
_module: InflationModule | None = None

def get_inflation_module(fred_api_key: str = None) -> InflationModule:
    global _module
    if _module is None:
        _module = InflationModule(fred_api_key)
    return _module
