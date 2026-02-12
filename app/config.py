"""Application configuration with environment separation."""
from functools import lru_cache
from enum import Enum

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class Environment(str, Enum):
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = Field(default=True)
    
    # App
    app_name: str = "MarketMind"
    api_v1_prefix: str = "/api/v1"
    
    # Supabase
    supabase_url: str = Field(default="")
    supabase_anon_key: str = Field(default="")
    supabase_service_key: str = Field(default="")  # For admin operations
    
    # Database - Supabase PostgreSQL
    # Pooler URL (port 6543) for app runtime
    database_url: str = Field(default="sqlite+aiosqlite:///./marketmind.db")
    # Direct URL (port 5432) for migrations
    database_url_direct: str = Field(default="")
    
    # Auth
    secret_key: str = Field(default="dev-secret-change-in-production")
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    
    # Security - stored as comma-separated string in .env
    allowed_origins: str = Field(default="http://localhost:3000,http://localhost:5173")
    rate_limit_per_minute: int = Field(default=60)
    
    # External APIs - Kalshi
    kalshi_api_key: str = Field(default="")
    kalshi_private_key_path: str = Field(default="./kalshi_private_key.pem")
    kalshi_private_key_pem: str = Field(default="")  # PEM string for Railway
    fred_api_key: str = Field(default="")
    polygon_api_key: str = Field(default="")
    openai_api_key: str = Field(default="")
    anthropic_api_key: str = Field(default="")
    
    @field_validator("debug", mode="before")
    @classmethod
    def set_debug(cls, v, info):
        """Disable debug in production."""
        if info.data.get("environment") == Environment.PRODUCTION:
            return False
        return v
    
    @property
    def allowed_origins_list(self) -> list[str]:
        """Get allowed origins as list."""
        return [origin.strip() for origin in self.allowed_origins.split(",")]
    
    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION
    
    @property
    def is_development(self) -> bool:
        return self.environment == Environment.DEVELOPMENT

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
