"""
AI Recruit - Authentication & Security Core
Production-ready authentication system with JWT, password security, and session management.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

import secrets
import hashlib
from datetime import datetime, timedelta, timezone
from typing import Optional, Dict, Any, Union
from uuid import UUID, uuid4
import jwt
from passlib.context import CryptContext
from passlib.hash import bcrypt
from email_validator import validate_email, EmailNotValidError
import logging

from core.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# JWT Configuration
JWT_ALGORITHM = settings.ALGORITHM
JWT_SECRET_KEY = settings.SECRET_KEY
ACCESS_TOKEN_EXPIRE_MINUTES = settings.ACCESS_TOKEN_EXPIRE_MINUTES
REFRESH_TOKEN_EXPIRE_MINUTES = settings.REFRESH_TOKEN_EXPIRE_MINUTES


class AuthenticationError(Exception):
    """Base authentication error."""
    pass


class InvalidCredentialsError(AuthenticationError):
    """Invalid login credentials."""
    pass


class TokenExpiredError(AuthenticationError):
    """JWT token has expired."""
    pass


class InvalidTokenError(AuthenticationError):
    """Invalid JWT token."""
    pass


class AccountLockedError(AuthenticationError):
    """Account is locked due to failed login attempts."""
    pass


class EmailNotVerifiedError(AuthenticationError):
    """Email address not verified."""
    pass


class PasswordSecurityManager:
    """
    Advanced password security management with comprehensive validation.
    """
    
    MIN_LENGTH = 8
    MAX_LENGTH = 128
    REQUIRE_UPPERCASE = True
    REQUIRE_LOWERCASE = True
    REQUIRE_DIGITS = True
    REQUIRE_SPECIAL = True
    SPECIAL_CHARS = "!@#$%^&*()_+-=[]{}|;:,.<>?"
    
    @classmethod
    def hash_password(cls, password: str) -> str:
        """
        Hash password using bcrypt with salt.
        
        Args:
            password: Plain text password
            
        Returns:
            Hashed password string
        """
        if not cls.validate_password_strength(password):
            raise ValueError("Password does not meet security requirements")
            
        return pwd_context.hash(password)
    
    @classmethod
    def verify_password(cls, plain_password: str, hashed_password: str) -> bool:
        """
        Verify password against hash.
        
        Args:
            plain_password: Plain text password
            hashed_password: Stored hash
            
        Returns:
            True if password matches
        """
        try:
            return pwd_context.verify(plain_password, hashed_password)
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False
    
    @classmethod
    def validate_password_strength(cls, password: str) -> bool:
        """
        Validate password meets security requirements.
        
        Args:
            password: Password to validate
            
        Returns:
            True if password meets all requirements
        """
        if not password or len(password) < cls.MIN_LENGTH or len(password) > cls.MAX_LENGTH:
            return False
            
        checks = [
            any(c.isupper() for c in password) if cls.REQUIRE_UPPERCASE else True,
            any(c.islower() for c in password) if cls.REQUIRE_LOWERCASE else True,
            any(c.isdigit() for c in password) if cls.REQUIRE_DIGITS else True,
            any(c in cls.SPECIAL_CHARS for c in password) if cls.REQUIRE_SPECIAL else True,
        ]
        
        return all(checks)
    
    @classmethod
    def generate_secure_password(cls, length: int = 16) -> str:
        """
        Generate a secure random password.
        
        Args:
            length: Password length (minimum 12)
            
        Returns:
            Secure random password
        """
        if length < 12:
            length = 12
            
        # Ensure we have at least one of each required character type
        password_chars = []
        
        if cls.REQUIRE_UPPERCASE:
            password_chars.append(secrets.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ"))
        if cls.REQUIRE_LOWERCASE:
            password_chars.append(secrets.choice("abcdefghijklmnopqrstuvwxyz"))
        if cls.REQUIRE_DIGITS:
            password_chars.append(secrets.choice("0123456789"))
        if cls.REQUIRE_SPECIAL:
            password_chars.append(secrets.choice(cls.SPECIAL_CHARS))
        
        # Fill remaining length with random characters
        all_chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789" + cls.SPECIAL_CHARS
        for _ in range(length - len(password_chars)):
            password_chars.append(secrets.choice(all_chars))
        
        # Shuffle the password
        secrets.SystemRandom().shuffle(password_chars)
        
        return ''.join(password_chars)


class JWTManager:
    """
    JWT token management for authentication and authorization.
    """
    
    @classmethod
    def create_access_token(
        cls, 
        user_id: Union[str, UUID], 
        email: str,
        role: str,
        organization_id: Union[str, UUID],
        expires_delta: Optional[timedelta] = None
    ) -> str:
        """
        Create JWT access token.
        
        Args:
            user_id: User ID
            email: User email
            role: User role
            organization_id: Organization ID
            expires_delta: Custom expiration time
            
        Returns:
            JWT token string
        """
        if expires_delta:
            expire = datetime.now(timezone.utc) + expires_delta
        else:
            expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": str(user_id),
            "email": email,
            "role": role,
            "org_id": str(organization_id),
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "access"
        }
        
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    
    @classmethod
    def create_refresh_token(
        cls, 
        user_id: Union[str, UUID],
        session_id: Optional[str] = None
    ) -> str:
        """
        Create JWT refresh token.
        
        Args:
            user_id: User ID
            session_id: Optional session ID
            
        Returns:
            JWT refresh token string
        """
        expire = datetime.now(timezone.utc) + timedelta(minutes=REFRESH_TOKEN_EXPIRE_MINUTES)
        
        to_encode = {
            "sub": str(user_id),
            "exp": expire,
            "iat": datetime.now(timezone.utc),
            "type": "refresh",
            "session_id": session_id or str(uuid4())
        }
        
        encoded_jwt = jwt.encode(to_encode, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)
        return encoded_jwt
    
    @classmethod
    def decode_token(cls, token: str) -> Dict[str, Any]:
        """
        Decode and validate JWT token.
        
        Args:
            token: JWT token string
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid
            TokenExpiredError: If token is expired
        """
        try:
            payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise TokenExpiredError("Token has expired")
        except jwt.JWTError as e:
            raise InvalidTokenError(f"Invalid token: {e}")
    
    @classmethod
    def verify_token_type(cls, token_payload: Dict[str, Any], expected_type: str) -> bool:
        """
        Verify token type matches expected type.
        
        Args:
            token_payload: Decoded token payload
            expected_type: Expected token type ('access' or 'refresh')
            
        Returns:
            True if token type matches
        """
        return token_payload.get("type") == expected_type


class EmailValidator:
    """
    Email validation and verification management.
    """
    
    @classmethod
    def validate_email_format(cls, email: str) -> bool:
        """
        Validate email format.
        
        Args:
            email: Email address to validate
            
        Returns:
            True if email format is valid
        """
        try:
            validate_email(email)
            return True
        except EmailNotValidError:
            return False
    
    @classmethod
    def generate_verification_token(cls) -> str:
        """
        Generate secure email verification token.
        
        Returns:
            Secure random token
        """
        return secrets.token_urlsafe(32)
    
    @classmethod
    def generate_password_reset_token(cls) -> str:
        """
        Generate secure password reset token.
        
        Returns:
            Secure random token
        """
        return secrets.token_urlsafe(32)


class SecurityAuditLogger:
    """
    Security event logging and audit trail management.
    """
    
    @classmethod
    def log_login_attempt(
        cls, 
        email: str, 
        success: bool, 
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        failure_reason: Optional[str] = None
    ):
        """
        Log login attempt for security auditing.
        
        Args:
            email: User email
            success: Whether login was successful
            ip_address: Client IP address
            user_agent: Client user agent
            failure_reason: Reason for failure if applicable
        """
        event_data = {
            "event": "login_attempt",
            "email": email,
            "success": success,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ip_address": ip_address,
            "user_agent": user_agent,
        }
        
        if not success and failure_reason:
            event_data["failure_reason"] = failure_reason
        
        if success:
            logger.info(f"Successful login: {email} from {ip_address}")
        else:
            logger.warning(f"Failed login attempt: {email} from {ip_address} - {failure_reason}")
    
    @classmethod
    def log_password_change(cls, user_id: str, email: str, ip_address: Optional[str] = None):
        """
        Log password change event.
        
        Args:
            user_id: User ID
            email: User email
            ip_address: Client IP address
        """
        logger.info(f"Password changed for user {user_id} ({email}) from {ip_address}")
    
    @classmethod
    def log_account_lockout(cls, email: str, ip_address: Optional[str] = None):
        """
        Log account lockout event.
        
        Args:
            email: User email
            ip_address: Client IP address
        """
        logger.warning(f"Account locked: {email} from {ip_address}")
    
    @classmethod
    def log_suspicious_activity(
        cls, 
        event_type: str, 
        details: Dict[str, Any],
        ip_address: Optional[str] = None
    ):
        """
        Log suspicious security activity.
        
        Args:
            event_type: Type of suspicious activity
            details: Event details
            ip_address: Client IP address
        """
        logger.error(f"Suspicious activity detected: {event_type} from {ip_address} - {details}")


class AuthenticationService:
    """
    Main authentication service orchestrating all security operations.
    """
    
    def __init__(self):
        self.password_manager = PasswordSecurityManager()
        self.jwt_manager = JWTManager()
        self.email_validator = EmailValidator()
        self.audit_logger = SecurityAuditLogger()
    
    def hash_password(self, password: str) -> str:
        """Hash password securely."""
        return self.password_manager.hash_password(password)
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.password_manager.verify_password(plain_password, hashed_password)
    
    def validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements."""
        return self.password_manager.validate_password_strength(password)
    
    def create_tokens(
        self, 
        user_id: Union[str, UUID], 
        email: str,
        role: str,
        organization_id: Union[str, UUID]
    ) -> Dict[str, str]:
        """
        Create access and refresh tokens for user.
        
        Args:
            user_id: User ID
            email: User email
            role: User role
            organization_id: Organization ID
            
        Returns:
            Dictionary with access_token and refresh_token
        """
        access_token = self.jwt_manager.create_access_token(
            user_id=user_id,
            email=email,
            role=role,
            organization_id=organization_id
        )
        
        refresh_token = self.jwt_manager.create_refresh_token(user_id=user_id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "expires_in": ACCESS_TOKEN_EXPIRE_MINUTES * 60
        }
    
    def decode_access_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate access token.
        
        Args:
            token: JWT access token
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid or wrong type
            TokenExpiredError: If token is expired
        """
        payload = self.jwt_manager.decode_token(token)
        
        if not self.jwt_manager.verify_token_type(payload, "access"):
            raise InvalidTokenError("Invalid token type")
        
        return payload
    
    def decode_refresh_token(self, token: str) -> Dict[str, Any]:
        """
        Decode and validate refresh token.
        
        Args:
            token: JWT refresh token
            
        Returns:
            Decoded token payload
            
        Raises:
            InvalidTokenError: If token is invalid or wrong type
            TokenExpiredError: If token is expired
        """
        payload = self.jwt_manager.decode_token(token)
        
        if not self.jwt_manager.verify_token_type(payload, "refresh"):
            raise InvalidTokenError("Invalid token type")
        
        return payload
    
    def validate_email(self, email: str) -> bool:
        """Validate email format."""
        return self.email_validator.validate_email_format(email)
    
    def generate_verification_token(self) -> str:
        """Generate email verification token."""
        return self.email_validator.generate_verification_token()
    
    def generate_password_reset_token(self) -> str:
        """Generate password reset token."""
        return self.email_validator.generate_password_reset_token()


# Global authentication service instance
auth_service = AuthenticationService()
