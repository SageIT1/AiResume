"""
AI Recruit - Authentication Schemas
Pydantic models for authentication requests and responses.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from datetime import datetime, timezone
from typing import Optional, List, Dict, Any
from uuid import UUID
from pydantic import BaseModel, EmailStr, Field, validator
import re

from core.security.auth import PasswordSecurityManager


class UserRegistrationRequest(BaseModel):
    """User registration request schema."""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., min_length=8, max_length=128, description="User password")
    confirm_password: str = Field(..., description="Password confirmation")
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    username: Optional[str] = Field(None, min_length=3, max_length=100, description="Username (optional)")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    
    # Organization information (for new organizations)
    organization_name: Optional[str] = Field(None, max_length=255, description="Organization name")
    organization_domain: Optional[str] = Field(None, max_length=255, description="Organization domain")
    organization_industry: Optional[str] = Field(None, max_length=100, description="Organization industry")
    
    @validator("password")
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        if not PasswordSecurityManager.validate_password_strength(v):
            raise ValueError(
                "Password must be at least 8 characters long and contain uppercase, "
                "lowercase, digit, and special character"
            )
        return v
    
    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Validate password confirmation matches."""
        if 'password' in values and v != values['password']:
            raise ValueError('Passwords do not match')
        return v
    
    @validator("username")
    def validate_username(cls, v):
        """Validate username format."""
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
                raise ValueError('Username can only contain letters, numbers, dots, hyphens, and underscores')
        return v
    
    @validator("phone")
    def validate_phone(cls, v):
        """Validate phone number format."""
        if v is not None:
            # Basic phone validation - can be enhanced based on requirements
            phone_pattern = r'^\+?[\d\s\-\(\)\.]{7,20}$'
            if not re.match(phone_pattern, v):
                raise ValueError('Invalid phone number format')
        return v


class UserLoginRequest(BaseModel):
    """User login request schema."""
    
    email: EmailStr = Field(..., description="User email address")
    password: str = Field(..., description="User password")
    remember_me: bool = Field(False, description="Remember login session")


class TokenRefreshRequest(BaseModel):
    """Token refresh request schema."""
    
    refresh_token: str = Field(..., description="Refresh token")


class PasswordResetRequest(BaseModel):
    """Password reset request schema."""
    
    email: EmailStr = Field(..., description="User email address")


class PasswordResetConfirm(BaseModel):
    """Password reset confirmation schema."""
    
    token: str = Field(..., description="Password reset token")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator("new_password")
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        if not PasswordSecurityManager.validate_password_strength(v):
            raise ValueError(
                "Password must be at least 8 characters long and contain uppercase, "
                "lowercase, digit, and special character"
            )
        return v
    
    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Validate password confirmation matches."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class PasswordChangeRequest(BaseModel):
    """Password change request schema."""
    
    current_password: str = Field(..., description="Current password")
    new_password: str = Field(..., min_length=8, max_length=128, description="New password")
    confirm_password: str = Field(..., description="Password confirmation")
    
    @validator("new_password")
    def validate_password_strength(cls, v):
        """Validate password meets security requirements."""
        if not PasswordSecurityManager.validate_password_strength(v):
            raise ValueError(
                "Password must be at least 8 characters long and contain uppercase, "
                "lowercase, digit, and special character"
            )
        return v
    
    @validator("confirm_password")
    def passwords_match(cls, v, values):
        """Validate password confirmation matches."""
        if 'new_password' in values and v != values['new_password']:
            raise ValueError('Passwords do not match')
        return v


class EmailVerificationRequest(BaseModel):
    """Email verification request schema."""
    
    token: str = Field(..., description="Email verification token")


class UserResponse(BaseModel):
    """User response schema."""
    
    id: UUID
    email: str
    username: str
    first_name: str
    last_name: str
    full_name: str
    role: str
    organization_id: UUID
    avatar_url: Optional[str] = None
    phone: Optional[str] = None
    timezone: str
    is_active: bool
    is_verified: bool
    must_change_password: bool = False
    permissions: List[str]
    preferences: Dict[str, Any]
    ai_agent_preferences: Dict[str, Any]
    default_llm_provider: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    last_login: Optional[datetime] = None
    last_activity: Optional[datetime] = None
    
    class Config:
        from_attributes = True


class OrganizationResponse(BaseModel):
    """Organization response schema."""
    
    id: UUID
    name: str
    domain: str
    description: Optional[str] = None
    industry: Optional[str] = None
    size: Optional[str] = None
    website: Optional[str] = None
    logo_url: Optional[str] = None
    settings: Dict[str, Any]
    is_active: bool
    created_at: datetime
    updated_at: datetime
    
    class Config:
        from_attributes = True


class AuthTokenResponse(BaseModel):
    """Authentication token response schema."""
    
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int
    user: UserResponse
    organization: OrganizationResponse


class TokenValidationResponse(BaseModel):
    """Token validation response schema."""
    
    valid: bool
    user_id: Optional[UUID] = None
    email: Optional[str] = None
    role: Optional[str] = None
    organization_id: Optional[UUID] = None
    expires_at: Optional[datetime] = None


class UserSessionResponse(BaseModel):
    """User session response schema."""
    
    id: UUID
    user_id: UUID
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    device_info: Dict[str, Any]
    is_active: bool
    expires_at: datetime
    last_activity: datetime
    created_at: datetime
    
    class Config:
        from_attributes = True


class UserProfileUpdateRequest(BaseModel):
    """User profile update request schema."""
    
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    phone: Optional[str] = Field(None, max_length=20)
    timezone: Optional[str] = Field(None, max_length=50)
    avatar_url: Optional[str] = Field(None, max_length=500)
    preferences: Optional[Dict[str, Any]] = None
    ai_agent_preferences: Optional[Dict[str, Any]] = None
    default_llm_provider: Optional[str] = Field(None, max_length=50)
    
    @validator("phone")
    def validate_phone(cls, v):
        """Validate phone number format."""
        if v is not None:
            phone_pattern = r'^\+?[\d\s\-\(\)\.]{7,20}$'
            if not re.match(phone_pattern, v):
                raise ValueError('Invalid phone number format')
        return v


class UserCreateRequest(BaseModel):
    """Admin user creation request schema."""
    
    email: EmailStr = Field(..., description="User email address")
    username: str = Field(..., min_length=3, max_length=100, description="Username")
    first_name: str = Field(..., min_length=1, max_length=100, description="First name")
    last_name: str = Field(..., min_length=1, max_length=100, description="Last name")
    role: str = Field(..., description="User role")
    phone: Optional[str] = Field(None, max_length=20, description="Phone number")
    is_active: bool = Field(True, description="User active status")
    send_invitation: bool = Field(True, description="Send invitation email")
    
    @validator("role")
    def validate_role(cls, v):
        """Validate user role."""
        allowed_roles = ["super_admin", "admin", "senior_recruiter", "recruiter", "viewer"]
        if v not in allowed_roles:
            raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v
    
    @validator("username")
    def validate_username(cls, v):
        """Validate username format."""
        if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
            raise ValueError('Username can only contain letters, numbers, dots, hyphens, and underscores')
        return v


class UserUpdateRequest(BaseModel):
    """Admin user update request schema."""
    
    email: Optional[EmailStr] = None
    username: Optional[str] = Field(None, min_length=3, max_length=100)
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    role: Optional[str] = None
    phone: Optional[str] = Field(None, max_length=20)
    is_active: Optional[bool] = None
    permissions: Optional[List[str]] = None
    
    @validator("role")
    def validate_role(cls, v):
        """Validate user role."""
        if v is not None:
            allowed_roles = ["super_admin", "admin", "senior_recruiter", "recruiter", "viewer"]
            if v not in allowed_roles:
                raise ValueError(f"Role must be one of: {', '.join(allowed_roles)}")
        return v
    
    @validator("username")
    def validate_username(cls, v):
        """Validate username format."""
        if v is not None:
            if not re.match(r'^[a-zA-Z0-9_.-]+$', v):
                raise ValueError('Username can only contain letters, numbers, dots, hyphens, and underscores')
        return v


class AuthErrorResponse(BaseModel):
    """Authentication error response schema."""
    
    error: str
    message: str
    details: Optional[Dict[str, Any]] = None


class SuccessResponse(BaseModel):
    """Generic success response schema."""
    
    success: bool = True
    message: str
    data: Optional[Dict[str, Any]] = None
