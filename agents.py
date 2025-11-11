"""
Agent management endpoints for AI Recruit.
"""

from fastapi import APIRouter
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["agents"])


@router.get("/status")
async def get_agents_status():
    """Get status of all AI agents."""
    logger.info("Get agents status endpoint called")
    return {"message": "Agents status - not implemented yet"}


@router.post("/restart")
async def restart_agents():
    """Restart AI agents."""
    logger.info("Restart agents endpoint called")
    return {"message": "Agent restart - not implemented yet"}