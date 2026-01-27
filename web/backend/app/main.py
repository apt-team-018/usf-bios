# Copyright (c) US Inc. All rights reserved.
"""
USF BIOS Web API - Main Application
Enterprise AI Training & Fine-tuning Platform
"""

import os
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

from .api import api_router
from .core.config import settings
from .core.database import init_db, engine
from .core.capabilities import is_system_expired
from .services.system_encrypted_log_service import system_encrypted_log

# System expiration middleware - blocks ALL APIs when expired
class SystemExpirationMiddleware(BaseHTTPMiddleware):
    """
    Middleware to check system expiration status.
    If system is expired, ALL API requests are blocked except health check.
    This cannot be bypassed - middleware runs before any route handler.
    """
    async def dispatch(self, request: Request, call_next):
        # Always allow health check for container orchestration
        if request.url.path == "/health":
            return await call_next(request)
        
        # Check system expiration
        expired, message = is_system_expired()
        if expired:
            return JSONResponse(
                status_code=503,
                content={
                    "error": "system_expired",
                    "message": "System requires upgrade. Please contact support.",
                    "blocked": True
                }
            )
        
        return await call_next(request)


# Create FastAPI application
app = FastAPI(
    title=settings.APP_NAME,
    description="Enterprise AI Training & Fine-tuning Platform API - Powered by US Inc",
    version=settings.APP_VERSION,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# System expiration middleware - MUST be after CORS to handle preflight
app.add_middleware(SystemExpirationMiddleware)

# Include API routes
app.include_router(api_router, prefix="/api")


@app.get("/health")
async def health_check():
    """Health check endpoint for container orchestration"""
    return {"status": "healthy", "service": "usf-bios-api"}


@app.get("/config")
async def get_config(request: Request):
    """Return backend URL directly from request - works with ANY cloud provider"""
    # Get the URL from THIS request's headers - always correct
    host = request.headers.get("host", "")
    scheme = request.headers.get("x-forwarded-proto", request.url.scheme)
    backend_url = f"{scheme}://{host}" if host else ""
    return {"backendUrl": backend_url}


@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    data_dir = os.environ.get("DATA_DIR", "/app/data")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(f"{data_dir}/datasets", exist_ok=True)
    os.makedirs(f"{data_dir}/checkpoints", exist_ok=True)
    os.makedirs(f"{data_dir}/logs", exist_ok=True)
    os.makedirs(f"{data_dir}/terminal_logs", exist_ok=True)
    
    # Create output directory using locked path from system_guard (single source of truth)
    from .core.capabilities import get_system_settings
    os.makedirs(str(get_system_settings().OUTPUT_DIR), exist_ok=True)
    
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
    
    # Initialize system encrypted log service for non-training logs
    # This handles: inference, datasets, models, system operations
    # Structure: /app/data/encrypted_logs/system/YYYY-MM-DD/HH.enc.log
    # Auto-cleanup: removes folders older than 24 hours (runs hourly)
    
    # CRITICAL: Ensure cleanup scheduler is running
    # This is BULLETPROOF - works after restarts, deployments, container recreation
    # - Runs cleanup IMMEDIATELY on startup (catches old logs)
    # - Continues running every hour in background daemon thread
    # - Never crashes - handles all exceptions internally
    # - Singleton pattern - only one cleanup thread per process
    cleanup_running = system_encrypted_log.ensure_cleanup_running()
    
    system_encrypted_log.log_system_startup({
        "app_name": settings.APP_NAME,
        "app_version": settings.APP_VERSION,
        "data_dir": data_dir,
        "output_dir": str(get_system_settings().OUTPUT_DIR),
        "cleanup_scheduler_active": cleanup_running
    })
    
    # Also run manual cleanup on startup as safety net
    cleanup_result = system_encrypted_log.force_cleanup()
    if cleanup_result.get("removed", 0) > 0:
        print(f"  Cleaned up {cleanup_result['removed']} old log folder(s)")
    
    print("=" * 60)
    print(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    print("  Powered by US Inc")
    print("=" * 60)


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    # Log system shutdown (encrypted only)
    system_encrypted_log.log_system_shutdown("normal")
    engine.dispose()
