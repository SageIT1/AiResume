"""
AI Recruit - Authentication Dependencies
FastAPI dependencies for authentication and authorization.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any, Union
from uuid import UUID
from fastapi import Depends, HTTPException, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

from core.config import get_settings
from database.sync_session import get_sync_db
from database.models import User, Organization, UserSession
from .auth import (
    auth_service, 
    AuthenticationError, 
    InvalidCredentialsError,
    TokenExpiredError,
    InvalidTokenError,
    AccountLockedError,
    EmailNotVerifiedError
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Security scheme for JWT token
security = HTTPBearer(auto_error=False)


class AuthenticationManager:
    """
    Centralized authentication and authorization management.
    """
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user_by_id(self, user_id: Union[str, UUID]) -> Optional[User]:
        """
        Get user by ID.
        
        Args:
            user_id: User ID
            
        Returns:
            User object or None if not found
        """
        try:
            return self.db.query(User).filter(User.id == str(user_id)).first()
        except Exception as e:
            logger.error(f"Error fetching user by ID {user_id}: {e}")
            return None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """
        Get user by email.
        
        Args:
            email: User email
            
        Returns:
            User object or None if not found
        """
        try:
            return self.db.query(User).filter(User.email == email).first()
        except Exception as e:
            logger.error(f"Error fetching user by email {email}: {e}")
            return None
    
    def validate_user_access(self, user: User) -> None:
        """
        Validate user can access the system.
        
        Args:
            user: User object
            
        Raises:
            HTTPException: If user cannot access system
        """
        if not user.is_active:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Account is deactivated"
            )
        
        # Check email verification (skip in development mode)
        if not user.is_verified and not settings.DEBUG:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Email address not verified"
            )
        
        # Check if account is locked
        if user.account_locked_until and user.account_locked_until > datetime.now(timezone.utc):
            raise HTTPException(
                status_code=status.HTTP_423_LOCKED,
                detail="Account is temporarily locked due to failed login attempts"
            )
    
    def update_user_activity(self, user: User, request: Request) -> None:
        """
        Update user's last activity timestamp.
        
        Args:
            user: User object
            request: FastAPI request object
        """
        try:
            user.last_activity = datetime.now(timezone.utc)
            self.db.commit()
        except Exception as e:
            logger.error(f"Error updating user activity for {user.id}: {e}")
            self.db.rollback()
    
    def get_active_session(self, user_id: Union[str, UUID], session_token: str) -> Optional[UserSession]:
        """
        Get active session for user.
        
        Args:
            user_id: User ID
            session_token: Session token
            
        Returns:
            UserSession object or None if not found
        """
        try:
            return self.db.query(UserSession).filter(
                UserSession.user_id == str(user_id),
                UserSession.session_token == session_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.now(timezone.utc)
            ).first()
        except Exception as e:
            logger.error(f"Error fetching session for user {user_id}: {e}")
            return None


def get_current_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_sync_db)
) -> User:
    """
    Get current authenticated user from JWT token.
    
    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        Authenticated user object
        
    Raises:
        HTTPException: If authentication fails
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication credentials required",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    try:
        # Decode JWT token
        token_payload = auth_service.decode_access_token(credentials.credentials)
        
        # Get user from database
        auth_manager = AuthenticationManager(db)
        user = auth_manager.get_user_by_id(token_payload["sub"])
        
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
                headers={"WWW-Authenticate": "Bearer"},
            )
        
        # Validate user access
        auth_manager.validate_user_access(user)
        
        # Update user activity
        auth_manager.update_user_activity(user, request)
        
        return user
        
    except TokenExpiredError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token has expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Authentication error: {e}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication failed",
            headers={"WWW-Authenticate": "Bearer"},
        )


def get_current_active_user(
    current_user: User = Depends(get_current_user)
) -> User:
    """
    Get current active user (alias for get_current_user for clarity).
    
    Args:
        current_user: Current authenticated user
        
    Returns:
        Active user object
    """
    return current_user


def get_optional_user(
    request: Request,
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_sync_db)
) -> Optional[User]:
    """
    Get current user if authenticated, otherwise None.
    
    Args:
        request: FastAPI request object
        credentials: HTTP authorization credentials
        db: Database session
        
    Returns:
        User object if authenticated, None otherwise
    """
    if not credentials:
        return None
    
    try:
        return get_current_user(request, credentials, db)
    except HTTPException:
        return None


class RoleChecker:
    """
    Role-based access control checker.
    """
    
    def __init__(self, allowed_roles: List[str]):
        self.allowed_roles = allowed_roles
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has required role.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User object if authorized
            
        Raises:
            HTTPException: If user doesn't have required role
        """
        if current_user.role not in self.allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Access denied. Required roles: {', '.join(self.allowed_roles)}"
            )
        return current_user


class PermissionChecker:
    """
    Permission-based access control checker.
    """
    
    def __init__(self, required_permissions: List[str], require_all: bool = True):
        self.required_permissions = required_permissions
        self.require_all = require_all
    
    def __call__(self, current_user: User = Depends(get_current_user)) -> User:
        """
        Check if user has required permissions.
        
        Args:
            current_user: Current authenticated user
            
        Returns:
            User object if authorized
            
        Raises:
            HTTPException: If user doesn't have required permissions
        """
        user_permissions = set(current_user.permissions or [])
        required_permissions = set(self.required_permissions)
        
        if self.require_all:
            # User must have all required permissions
            if not required_permissions.issubset(user_permissions):
                missing = required_permissions - user_permissions
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Missing permissions: {', '.join(missing)}"
                )
        else:
            # User must have at least one required permission
            if not required_permissions.intersection(user_permissions):
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Access denied. Required permissions: {', '.join(self.required_permissions)}"
                )
        
        return current_user


class OrganizationChecker:
    """
    Organization-based access control checker.
    """
    
    def __init__(self, allow_cross_org: bool = False):
        self.allow_cross_org = allow_cross_org
    
    def __call__(
        self, 
        organization_id: UUID,
        current_user: User = Depends(get_current_user)
    ) -> User:
        """
        Check if user can access resources from specified organization.
        
        Args:
            organization_id: Target organization ID
            current_user: Current authenticated user
            
        Returns:
            User object if authorized
            
        Raises:
            HTTPException: If user cannot access organization
        """
        # Super admins can access any organization
        if current_user.role == "super_admin":
            return current_user
        
        # Check if user belongs to the organization or cross-org access is allowed
        if not self.allow_cross_org and str(current_user.organization_id) != str(organization_id):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Access denied. Cannot access resources from other organizations"
            )
        
        return current_user


# Pre-defined role checkers for common use cases
require_admin = RoleChecker(["super_admin", "admin"])
require_senior_recruiter = RoleChecker(["super_admin", "admin", "senior_recruiter"])
require_recruiter = RoleChecker(["super_admin", "admin", "senior_recruiter", "recruiter"])
require_any_role = RoleChecker(["super_admin", "admin", "senior_recruiter", "recruiter", "viewer"])

# Pre-defined permission checkers
require_user_management = PermissionChecker(["manage_users"])
require_job_management = PermissionChecker(["manage_jobs"])
require_resume_management = PermissionChecker(["manage_resumes"])
require_analytics_access = PermissionChecker(["view_analytics"])

# Organization checker
require_same_organization = OrganizationChecker(allow_cross_org=False)
allow_cross_organization = OrganizationChecker(allow_cross_org=True)
