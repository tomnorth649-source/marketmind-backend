"""Research modules for probability estimation."""
from app.services.research.fed import FedResearchModule, get_fed_module
from app.services.research.inflation import InflationModule, get_inflation_module
from app.services.research.weather import WeatherModule, get_weather_module

__all__ = [
    "FedResearchModule", "get_fed_module",
    "InflationModule", "get_inflation_module",
    "WeatherModule", "get_weather_module",
]
