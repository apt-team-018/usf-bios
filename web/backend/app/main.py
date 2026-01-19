# Copyright (c) US Inc. All rights reserved.
"""
USF BIOS Web API - Main Application
Enterprise AI Training & Fine-tuning Platform
"""

import os
import json
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .api import api_router
from .core.config import settings
from .core.database import init_db, engine

# Detected backend URL - stored in memory, served via API endpoint
_detected_backend_url: str | None = None

def _is_internal_request(host: str) -> bool:
    """Check if request is from internal health check (not external browser)"""
    if not host:
        return True
    # Skip localhost, 127.0.0.1, internal docker IPs, container hostnames
    host_lower = host.lower().split(":")[0]  # Remove port
    if host_lower in ("localhost", "127.0.0.1", "0.0.0.0"):
        return True
    if host_lower.startswith("172.") or host_lower.startswith("10.") or host_lower.startswith("192.168."):
        return True
    # Skip container hostnames (hex strings like 043ab22f9f6a)
    if len(host_lower) == 12 and all(c in "0123456789abcdef" for c in host_lower):
        return True
    return False

class DetectExternalUrlMiddleware(BaseHTTPMiddleware):
    """Middleware to detect backend's external URL from EXTERNAL request only"""
    async def dispatch(self, request: Request, call_next):
        global _detected_backend_url
        if _detected_backend_url is None:
            host = request.headers.get("host", "")
            # Only detect from external requests (browser), not internal health checks
            if host and not _is_internal_request(host):
                scheme = request.headers.get("x-forwarded-proto", "https")
                _detected_backend_url = f"{scheme}://{host}"
                print(f"[Backend] External URL detected: {_detected_backend_url}")
        return await call_next(request)

# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise AI Training & Fine-tuning Platform API - Powered by US Inc",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Add URL detection middleware (runs on every request until URL is detected)
app.add_middleware(DetectExternalUrlMiddleware)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "usf-bios-api"}


@app.get("/config")
async def get_config():
    """Return detected backend URL for frontend to use"""
    return {"backendUrl": _detected_backend_url or ""}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    os.makedirs("/app/data", exist_ok=True)
    os.makedirs("/app/data/datasets", exist_ok=True)
    os.makedirs("/app/data/output", exist_ok=True)
    os.makedirs("/app/data/checkpoints", exist_ok=True)
    os.makedirs("/app/data/logs", exist_ok=True)
    os.makedirs("/app/data/terminal_logs", exist_ok=True)
    
    init_db()
    
    from .core.database import get_db_session
    from .services.job_service import JobService
    
    with get_db_session() as db:
        job_service = JobService(db)
        recovered = job_service.recover_interrupted_jobs()
        if recovered:
            print(f"  Recovered {len(recovered)} interrupted job(s)")
            for r in recovered:
                print(f"    - Job {r['job_id']}: {r['name']} -> {r['new_status']}")
    
    print("=" * 60)
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print("  Powered by US Inc")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    engine.dispose()
