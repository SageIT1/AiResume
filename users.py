"""
AI Recruit - User Management Endpoints
Comprehensive user management with role-based access control.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from typing import List, Optional
from uuid import UUID
from fastapi import APIRouter, Depends, HTTPException, status, Query
from sqlalchemy.orm import Session
from sqlalchemy import or_, and_
import logging

from database.sync_session import get_sync_db
from database.models import User, Organization
from core.services.user_service import (
    UserService, 
    UserServiceError, 
    UserAlreadyExistsError
)
from core.security.dependencies import (
    get_current_user, 
    require_admin, 
    require_recruiter,
    require_same_organization
)
from core.schemas.auth import (
    UserResponse,
    UserCreateRequest,
    UserUpdateRequest,
    UserProfileUpdateRequest,
    SuccessResponse,
    PasswordChangeRequest
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["users"])


@router.get("/me", response_model=UserResponse)
async def get_current_user_profile(
    current_user: User = Depends(get_current_user)
):
    """
    Get current user profile.
    
    Returns detailed information about the authenticated user.
    """
    return UserResponse.from_orm(current_user)


@router.put("/me", response_model=UserResponse)
async def update_current_user_profile(
    profile_data: UserProfileUpdateRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_sync_db)
):
    """
    Update current user profile.
    
    Allows users to update their own profile information.
    """
    try:
        user_service = UserService(db)
        updated_user = await user_service.update_user_profile(
            user=current_user,
            profile_data=profile_data
        )

        return UserResponse.from_orm(updated_user)
        
    except UserServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error updating user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Profile update failed due to server error"
        )

@router.post("/{user_id}/reset-password", response_model=SuccessResponse)
async def admin_reset_user_password(
    user_id: UUID,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_sync_db)
):
    """
    Admin resets a user's password to the default and enforces first login change.
    """
    try:
        user: Optional[User] = db.query(User).filter(
            User.id == user_id,
            User.organization_id == current_user.organization_id
        ).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        # Prevent non-super_admin from resetting super_admin
        if user.role == "super_admin" and current_user.role != "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super_admin can reset another super_admin's password"
            )

        from core.config import get_settings
        from core.security.auth import auth_service
        settings = get_settings()
        new_hashed = auth_service.hash_password(settings.DEFAULT_NEW_USER_PASSWORD)
        user.hashed_password = new_hashed
        user.must_change_password = True
        user.password_changed_at = None
        db.commit()
        return SuccessResponse(message="Password reset to default. User must change on next login.")
    except HTTPException:
        raise
    except Exception as e:
        logging.getLogger(__name__).error(f"Error resetting password for user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset user password"
        )



@router.get("/", response_model=List[UserResponse])
async def list_users(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    search: Optional[str] = Query(None, description="Search by name or email"),
    role: Optional[str] = Query(None, description="Filter by role"),
    is_active: Optional[bool] = Query(None, description="Filter by active status"),
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_sync_db)
):
    """
    List users in the organization.
    
    Supports pagination, search, and filtering.
    Requires recruiter role or higher.
    """
    try:
        # Build query
        query = db.query(User).filter(User.organization_id == current_user.organization_id)
        
        # Apply filters
        if search:
            search_filter = or_(
                User.first_name.ilike(f"%{search}%"),
                User.last_name.ilike(f"%{search}%"),
                User.email.ilike(f"%{search}%"),
                User.username.ilike(f"%{search}%")
            )
            query = query.filter(search_filter)
        
        if role:
            query = query.filter(User.role == role)
        
        if is_active is not None:
            query = query.filter(User.is_active == is_active)
        
        # Apply pagination
        offset = (page - 1) * page_size
        users = query.offset(offset).limit(page_size).all()
        
        return [UserResponse.from_orm(user) for user in users]
        
    except Exception as e:
        logger.error(f"Error listing users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve users"
        )


@router.post("/", response_model=UserResponse)
async def create_user(
    user_data: UserCreateRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_sync_db)
):
    """
    Create a new user.
    
    Admin-only endpoint for creating users within the organization.
    """
    try:
        user_service = UserService(db)
        new_user = await user_service.create_user(
            user_data=user_data,
            created_by=current_user
        )
        
        return UserResponse.from_orm(new_user)
        
    except UserAlreadyExistsError as e:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(e)
        )
    except UserServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error creating user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="User creation failed due to server error"
        )


@router.get("/{user_id}", response_model=UserResponse)
async def get_user_by_id(
    user_id: UUID,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_sync_db)
):
    """
    Get user by ID.
    
    Returns detailed user information.
    Requires recruiter role or higher.
    """
    try:
        user = db.query(User).filter(
            User.id == user_id,
            User.organization_id == current_user.organization_id
        ).first()
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )
        
        return UserResponse.from_orm(user)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user"
        )


@router.get("/roles/available")
async def get_available_roles(
    current_user: User = Depends(require_admin)
):
    """
    Get list of available user roles.
    
    Returns roles that can be assigned based on current user's permissions.
    """
    # Define role hierarchy
    all_roles = ["super_admin", "admin", "senior_recruiter", "recruiter", "viewer"]
    
    if current_user.role == "super_admin":
        available_roles = all_roles
    elif current_user.role == "admin":
        available_roles = ["admin", "senior_recruiter", "recruiter", "viewer"]
    else:
        available_roles = ["recruiter", "viewer"]
    
    return {
        "available_roles": available_roles,
        "role_descriptions": {
            "super_admin": "Full system access across all organizations",
            "admin": "Full access within organization",
            "senior_recruiter": "Advanced recruiting features and team management",
            "recruiter": "Standard recruiting features",
            "viewer": "Read-only access to recruiting data"
        }
    }


@router.put("/{user_id}", response_model=UserResponse)
async def update_user_by_id(
    user_id: UUID,
    update_data: UserUpdateRequest,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_sync_db)
):
    """
    Update a user's details (admin-only).
    Ensures the target user belongs to the same organization.
    """
    try:
        user: Optional[User] = db.query(User).filter(
            User.id == user_id,
            User.organization_id == current_user.organization_id
        ).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Prevent changing super_admin by non-super_admin
        if user.role == "super_admin" and current_user.role != "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super_admin can modify another super_admin"
            )

        # Uniqueness checks if email/username changed
        if update_data.email and update_data.email != user.email:
            exists = db.query(User).filter(User.email == update_data.email).first()
            if exists:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Email already in use"
                )
            user.email = update_data.email

        if update_data.username and update_data.username != user.username:
            exists = db.query(User).filter(User.username == update_data.username).first()
            if exists:
                raise HTTPException(
                    status_code=status.HTTP_409_CONFLICT,
                    detail="Username already in use"
                )
            user.username = update_data.username

        if update_data.first_name is not None:
            user.first_name = update_data.first_name
        if update_data.last_name is not None:
            user.last_name = update_data.last_name
        # Keep full_name coherent
        if update_data.first_name is not None or update_data.last_name is not None:
            user.full_name = f"{user.first_name} {user.last_name}".strip()

        if update_data.phone is not None:
            user.phone = update_data.phone
        if update_data.is_active is not None:
            user.is_active = update_data.is_active
        if update_data.permissions is not None:
            user.permissions = update_data.permissions

        # Role update rules
        if update_data.role is not None:
            if current_user.role == "admin" and update_data.role == "super_admin":
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="Admin cannot promote to super_admin"
                )
            user.role = update_data.role

        db.commit()
        db.refresh(user)
        return UserResponse.from_orm(user)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update user"
        )


@router.delete("/{user_id}", response_model=SuccessResponse)
async def delete_user_by_id(
    user_id: UUID,
    current_user: User = Depends(require_admin),
    db: Session = Depends(get_sync_db)
):
    """
    Delete a user (admin-only). Prevent deleting self. Same organization only.
    """
    try:
        if str(current_user.id) == str(user_id):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="You cannot delete your own account"
            )

        user: Optional[User] = db.query(User).filter(
            User.id == user_id,
            User.organization_id == current_user.organization_id
        ).first()
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found"
            )

        # Prevent non-super_admin from deleting super_admin
        if user.role == "super_admin" and current_user.role != "super_admin":
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Only super_admin can delete another super_admin"
            )

        db.delete(user)
        db.commit()
        return SuccessResponse(message="User deleted successfully")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting user {user_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete user"
        )