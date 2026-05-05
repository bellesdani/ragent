from __future__ import annotations

from fastapi import APIRouter
from app.api.routers import chat, health


router = APIRouter()
router.include_router(health.router)
router.include_router(chat.router)
