"""
Routes package initialization.
"""
from fastapi import APIRouter
from . import detection, users, health

router = APIRouter()

router.include_router(detection.router, prefix="/detection", tags=["Detection"])
router.include_router(users.router, prefix="/users", tags=["Users"])
router.include_router(health.router, prefix="/health", tags=["Health"]) 