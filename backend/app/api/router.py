from fastapi import APIRouter
from app.api.endpoints import upload, detection, results

api_router = APIRouter()

api_router.include_router(upload.router, prefix="/upload", tags=["upload"])
api_router.include_router(detection.router, prefix="/detection", tags=["detection"])
api_router.include_router(results.router, prefix="/results", tags=["results"])