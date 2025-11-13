"""
AI Recruit - Security Module
Core security utilities and authentication management.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from .auth import (
    auth_service,
    PasswordSecurityManager,
    JWTManager,
    EmailValidator,
    SecurityAuditLogger,
    AuthenticationError,
    InvalidCredentialsError,
    TokenExpiredError,
    InvalidTokenError,
    AccountLockedError,
    EmailNotVerifiedError
)

from .dependencies import (
    get_current_user,
    get_current_active_user,
    get_optional_user,
    RoleChecker,
    PermissionChecker,
    OrganizationChecker,
    require_admin,
    require_senior_recruiter,
    require_recruiter,
    require_any_role,
    require_user_management,
    require_job_management,
    require_resume_management,
    require_analytics_access,
    require_same_organization,
    allow_cross_organization
)

__all__ = [
    # Core authentication service
    "auth_service",
    
    # Security managers
    "PasswordSecurityManager",
    "JWTManager", 
    "EmailValidator",
    "SecurityAuditLogger",
    
    # Exceptions
    "AuthenticationError",
    "InvalidCredentialsError",
    "TokenExpiredError",
    "InvalidTokenError",
    "AccountLockedError",
    "EmailNotVerifiedError",
    
    # Dependencies
    "get_current_user",
    "get_current_active_user",
    "get_optional_user",
    "RoleChecker",
    "PermissionChecker",
    "OrganizationChecker",
    
    # Pre-configured role checkers
    "require_admin",
    "require_senior_recruiter",
    "require_recruiter",
    "require_any_role",
    
    # Pre-configured permission checkers
    "require_user_management",
    "require_job_management",
    "require_resume_management",
    "require_analytics_access",
    
    # Organization checkers
    "require_same_organization",
    "allow_cross_organization"
]