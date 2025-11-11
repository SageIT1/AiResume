"""
File management endpoints for AI Recruit.
"""

from fastapi import APIRouter, UploadFile, File
import logging

logger = logging.getLogger(__name__)

router = APIRouter(tags=["files"])


@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a file."""
    logger.info(f"Upload file endpoint called: {file.filename}")
    return {"message": f"File upload {file.filename} - not implemented yet"}


@router.get("/{file_id}")
async def get_file(file_id: str):
    """Get a file by ID."""
    logger.info(f"Get file {file_id} endpoint called")
    return {"message": f"Get file {file_id} - not implemented yet"}