"""Database connection with Supabase PostgreSQL support."""
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.pool import NullPool

from app.config import get_settings

settings = get_settings()

# Configure engine based on database URL
# Use NullPool for serverless/edge deployments to avoid connection issues
engine_kwargs = {
    "echo": settings.debug and settings.is_development,
}

# For PostgreSQL (Supabase), configure for pgbouncer compatibility
if "postgresql" in settings.database_url:
    engine_kwargs["poolclass"] = NullPool
    # Disable prepared statements for pgbouncer transaction pooling
    engine_kwargs["connect_args"] = {
        "statement_cache_size": 0,
        "prepared_statement_cache_size": 0,
    }

engine = create_async_engine(settings.database_url, **engine_kwargs)

async_session_maker = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(DeclarativeBase):
    """Base class for all models."""
    pass


async def get_db() -> AsyncSession:
    """Dependency to get database session."""
    async with async_session_maker() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()


async def init_db():
    """Initialize database tables.
    
    In production with Supabase, use Alembic migrations instead.
    This is for local development only.
    """
    if settings.is_development and "sqlite" in settings.database_url:
        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
