"""
AI Recruit - Authentication Endpoints
Complete authentication system with registration, login, logout, and token management.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from datetime import datetime, timezone
from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Response
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from sqlalchemy.orm import Session
import logging

from database.sync_session import get_sync_db
from database.models import User, Organization
from core.services.user_service import (
    UserService, 
    UserServiceError, 
    UserAlreadyExistsError,
    InvalidPasswordError
)
from core.security.auth import (
    InvalidCredentialsError,
    AccountLockedError,
    EmailNotVerifiedError,
    TokenExpiredError,
    InvalidTokenError
)
from core.security.dependencies import get_current_user, get_optional_user
from core.schemas.auth import (
    UserRegistrationRequest,
    UserLoginRequest,
    TokenRefreshRequest,
    PasswordChangeRequest,
    PasswordResetRequest,
    PasswordResetConfirm,
    EmailVerificationRequest,
    UserResponse,
    OrganizationResponse,
    AuthTokenResponse,
    AuthErrorResponse,
    SuccessResponse
)

logger = logging.getLogger(__name__)
security = HTTPBearer(auto_error=False)

router = APIRouter(tags=["authentication"])


def get_client_info(request: Request) -> tuple[Optional[str], Optional[str]]:
    """Extract client IP and user agent from request."""
    ip_address = request.client.host if request.client else None
    user_agent = request.headers.get("user-agent")
    return ip_address, user_agent


@router.post("/register", response_model=AuthTokenResponse)
async def register_user(
    registration_data: UserRegistrationRequest,
    request: Request,
    db: Session = Depends(get_sync_db)
):
    """
    Register a new user with organization.
    
    Creates a new user account and organization if needed.
    Returns authentication tokens for immediate login.
    """
    try:
        ip_address, user_agent = get_client_info(request)
        
        user_service = UserService(db)
        user, organization, tokens = await user_service.register_user(
            registration_data=registration_data,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        return AuthTokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user=UserResponse.from_orm(user),
            organization=OrganizationResponse.from_orm(organization)
        )
        
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
        logger.error(f"Unexpected error during registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Registration failed due to server error"
        )


@router.post("/login", response_model=AuthTokenResponse)
async def login_user(
    login_data: UserLoginRequest,
    request: Request,
    db: Session = Depends(get_sync_db)
):
    """
    Authenticate user and create session.
    
    Validates credentials and returns authentication tokens.
    """
    try:
        ip_address, user_agent = get_client_info(request)
        
        user_service = UserService(db)
        user, tokens = await user_service.authenticate_user(
            login_data=login_data,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Get organization
        organization = db.query(Organization).filter(Organization.id == user.organization_id).first()
        
        return AuthTokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user=UserResponse.from_orm(user),
            organization=OrganizationResponse.from_orm(organization)
        )
        
    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except AccountLockedError as e:
        raise HTTPException(
            status_code=status.HTTP_423_LOCKED,
            detail=str(e)
        )
    except EmailNotVerifiedError as e:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=str(e)
        )
    except UserServiceError as e:
        print(f"UserServiceError: {e}")
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during login: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed due to server error"
        )


@router.post("/refresh", response_model=AuthTokenResponse)
async def refresh_token(
    refresh_data: TokenRefreshRequest,
    request: Request,
    db: Session = Depends(get_sync_db)
):
    """
    Refresh authentication token.
    
    Uses refresh token to generate new access and refresh tokens.
    """
    try:
        ip_address, user_agent = get_client_info(request)
        
        user_service = UserService(db)
        user, tokens = await user_service.refresh_user_token(
            refresh_token=refresh_data.refresh_token,
            ip_address=ip_address,
            user_agent=user_agent
        )
        
        # Get organization
        organization = db.query(Organization).filter(Organization.id == user.organization_id).first()
        
        return AuthTokenResponse(
            access_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            token_type=tokens["token_type"],
            expires_in=tokens["expires_in"],
            user=UserResponse.from_orm(user),
            organization=OrganizationResponse.from_orm(organization)
        )
        
    except (InvalidTokenError, TokenExpiredError) as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during token refresh: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed due to server error"
        )


@router.post("/logout", response_model=SuccessResponse)
async def logout_user(
    request: Request,
    revoke_all_sessions: bool = False,
    current_user: User = Depends(get_current_user),
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: Session = Depends(get_sync_db)
):
    """
    Logout user and revoke session(s).
    
    Revokes current session or all user sessions if requested.
    """
    try:
        user_service = UserService(db)
        success = await user_service.logout_user(
            user=current_user,
            access_token=credentials.credentials,
            revoke_all_sessions=revoke_all_sessions
        )
        
        if success:
            return SuccessResponse(
                message="Logged out successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Logout failed"
            )
            
    except Exception as e:
        logger.error(f"Unexpected error during logout: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Logout failed due to server error"
        )


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """
    Get current authenticated user information.
    
    Returns detailed user profile and preferences.
    """
    return UserResponse.from_orm(current_user)


@router.post("/change-password", response_model=SuccessResponse)
async def change_password(
    password_data: PasswordChangeRequest,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_sync_db)
):
    """
    Change user password.
    
    Requires current password for verification.
    """
    try:
        ip_address, user_agent = get_client_info(request)
        
        user_service = UserService(db)
        success = await user_service.change_password(
            user=current_user,
            password_data=password_data,
            ip_address=ip_address
        )
        
        if success:
            return SuccessResponse(
                message="Password changed successfully"
            )
        else:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Password change failed"
            )
            
    except InvalidPasswordError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except UserServiceError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Unexpected error during password change: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Password change failed due to server error"
        )


@router.get("/validate-token")
async def validate_token(
    current_user: Optional[User] = Depends(get_optional_user)
):
    """
    Validate authentication token.
    
    Returns token validation status and user info if valid.
    """
    if current_user:
        return {
            "valid": True,
            "user": UserResponse.from_orm(current_user)
        }
    else:
        return {
            "valid": False,
            "message": "Invalid or expired token"
        }