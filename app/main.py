"""MarketMind API - Production-ready FastAPI application."""
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

from app.config import get_settings
from app.database import init_db
from app.api.v1.router import router as api_v1_router

settings = get_settings()

# Rate limiter
limiter = Limiter(key_func=get_remote_address)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan - startup and shutdown."""
    # Startup
    if settings.is_development:
        await init_db()  # Only auto-create tables in dev
    yield
    # Shutdown - cleanup if needed


app = FastAPI(
    title=settings.app_name,
    description="AI-powered research platform for prediction market traders",
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.debug else None,  # Disable docs in production
    redoc_url="/redoc" if settings.debug else None,
    openapi_url="/openapi.json" if settings.debug else None,
)

# Rate limiting
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS - strict in production
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list if settings.is_production else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "PATCH"],
    allow_headers=["*"],
    max_age=600,  # Cache preflight for 10 minutes
)


@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    """Add security headers to all responses."""
    response = await call_next(request)
    
    if settings.is_production:
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains"
        response.headers["Content-Security-Policy"] = "default-src 'self'"
    
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log requests in development."""
    if settings.is_development:
        print(f"[{request.method}] {request.url.path}")
    return await call_next(request)


# Include API routes
app.include_router(api_v1_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "status": "healthy",
        "environment": settings.environment.value,
    }


@app.get("/health")
async def health():
    """Detailed health check for monitoring."""
    return {
        "status": "healthy",
        "service": "marketmind-api",
        "version": "0.1.0",
    }
