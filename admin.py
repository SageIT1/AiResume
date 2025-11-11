"""
AI Recruit - Admin API Endpoints
Administrative endpoints for system management.
"""

import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from sqlalchemy.ext.asyncio import AsyncSession

from core.security import get_current_user
from database.session import get_db
from database.seeds.create_default_data import seed_database

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/seed-database")
async def seed_database_endpoint(
    current_user = Depends(get_current_user)
):
    """
    Seed the database with default organizations and users.
    
    This endpoint creates:
    - Default organization for development
    - Default admin user
    
    Useful for setting up development environments.
    """
    try:
        logger.info(f"🌱 Database seeding requested by user: {current_user.username}")
        
        # Check if user has admin role
        if getattr(current_user, 'role', None) != 'admin':
            raise HTTPException(
                status_code=403, 
                detail="Only admin users can seed the database"
            )
        
        # Run seeding
        result = await seed_database()
        
        logger.info("✅ Database seeding completed via API")
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Database seeded successfully",
                "details": {
                    "organization_id": result["organization_id"],
                    "user_id": result["user_id"],
                    "status": result["status"]
                },
                "instructions": [
                    "Default organization and user have been created",
                    "Mock authentication will now use existing organization",
                    "Resume uploads should work without foreign key errors"
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Database seeding failed via API: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "message": "Database seeding failed",
                "error": str(e),
                "troubleshooting": [
                    "Check database connection",
                    "Ensure migrations are up to date",
                    "Verify DATABASE_URL in environment variables"
                ]
            }
        )


@router.get("/database-status")
async def check_database_status(
    current_user = Depends(get_current_user)
):
    """
    Check the current database status and seeding state.
    """
    try:
        from database.seeds.create_default_data import get_default_organization_id, get_default_user_id
        
        # Check if default organization exists
        org_id = await get_default_organization_id()
        user_id = await get_default_user_id()
        
        return JSONResponse(
            status_code=200,
            content={
                "database_status": "connected",
                "seeding_status": "completed",
                "default_organization_id": org_id,
                "default_user_id": user_id,
                "current_user": {
                    "id": str(current_user.id),
                    "organization_id": str(current_user.organization_id),
                    "username": current_user.username,
                    "role": current_user.role
                },
                "foreign_key_status": "✅ organization_id matches existing organization"
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Database status check failed: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "database_status": "error",
                "seeding_status": "incomplete",
                "error": str(e),
                "recommendation": "Run database seeding to fix foreign key issues"
            }
        )


@router.post("/fix-foreign-key-error")
async def fix_foreign_key_error(
    current_user = Depends(get_current_user)
):
    """
    Quick fix for foreign key constraint violations.
    Creates the missing organization that the current user references.
    """
    try:
        logger.info(f"🔧 Foreign key fix requested by user: {current_user.username}")
        
        # Run seeding to ensure organization exists
        result = await seed_database()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Foreign key issue resolved",
                "fix_applied": "Default organization created",
                "organization_id": result["organization_id"],
                "status": "Resume uploads should now work",
                "next_steps": [
                    "Try uploading a resume again",
                    "The foreign key constraint error should be resolved"
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Foreign key fix failed: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "message": "Failed to fix foreign key issue",
                "error": str(e),
                "manual_steps": [
                    "Run: python backend/seed_database.py",
                    "Or check database migrations: alembic upgrade head",
                    "Ensure DATABASE_URL is correct in .env file"
                ]
            }
        )


@router.post("/fix-azure-storage-metadata")
async def fix_azure_storage_metadata(
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Fix resumes that are missing Azure storage metadata.
    This can happen if resumes were uploaded before the Azure integration was properly configured.
    """
    try:
        from sqlalchemy import select, update
        from database.models import Resume, ResumeStatus
        
        logger.info(f"🔧 Azure storage metadata fix requested by user: {current_user.username}")
        
        # Check if user has admin role
        if getattr(current_user, 'role', None) != 'admin':
            raise HTTPException(
                status_code=403, 
                detail="Only admin users can fix Azure storage metadata"
            )
        
        # Find resumes with missing Azure metadata
        result = await db.execute(
            select(Resume).where(
                Resume.organization_id == current_user.organization_id,
                Resume.azure_blob_name.is_(None) | (Resume.azure_blob_name == "")
            )
        )
        problematic_resumes = result.scalars().all()
        
        if not problematic_resumes:
            return JSONResponse(
                status_code=200,
                content={
                    "message": "No resumes found with missing Azure storage metadata",
                    "resumes_checked": 0,
                    "resumes_fixed": 0
                }
            )
        
        fixed_count = 0
        failed_count = 0
        
        for resume in problematic_resumes:
            try:
                # Generate what the blob name should have been
                import os
                from uuid import uuid4
                
                file_ext = os.path.splitext(resume.original_filename)[1]
                # We don't have the original user_id used for upload, so we'll use a generic path
                new_blob_name = f"legacy/{resume.id}{file_ext}"
                
                # Update the resume record
                resume.azure_blob_name = new_blob_name
                resume.azure_container = "resumes"  # Default container
                resume.azure_url = f"https://[storage-account].blob.core.windows.net/resumes/{new_blob_name}"
                
                fixed_count += 1
                logger.info(f"✅ Fixed Azure metadata for resume {resume.id}")
                
            except Exception as e:
                failed_count += 1
                logger.error(f"❌ Failed to fix resume {resume.id}: {str(e)}")
        
        # Commit all changes
        await db.commit()
        
        return JSONResponse(
            status_code=200,
            content={
                "message": "Azure storage metadata fix completed",
                "resumes_checked": len(problematic_resumes),
                "resumes_fixed": fixed_count,
                "resumes_failed": failed_count,
                "details": [
                    f"Fixed {fixed_count} resumes with missing Azure metadata",
                    f"Failed to fix {failed_count} resumes",
                    "Note: This only fixes the database metadata - the actual files may need to be re-uploaded"
                ],
                "recommendations": [
                    "Ask users to re-upload any resumes that still fail to reprocess",
                    "Consider implementing a background job to migrate old files to Azure storage"
                ]
            }
        )
        
    except Exception as e:
        logger.error(f"❌ Azure storage metadata fix failed: {str(e)}")
        
        return JSONResponse(
            status_code=500,
            content={
                "message": "Azure storage metadata fix failed",
                "error": str(e),
                "troubleshooting": [
                    "Check database connection",
                    "Verify admin permissions",
                    "Check application logs for detailed errors"
                ]
            }
        )