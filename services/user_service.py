"""
AI Recruit - User Service Layer
Business logic for user management, authentication, and organization handling.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any, Tuple, Union
from uuid import UUID, uuid4
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
import logging

from core.config import get_settings
from database.models import User, Organization, UserSession
from core.security.auth import (
    auth_service,
    AuthenticationError,
    InvalidCredentialsError,
    AccountLockedError,
    EmailNotVerifiedError,
    SecurityAuditLogger
)
from core.schemas.auth import (
    UserRegistrationRequest,
    UserLoginRequest,
    UserCreateRequest,
    UserUpdateRequest,
    UserProfileUpdateRequest,
    PasswordChangeRequest,
    PasswordResetRequest,
    PasswordResetConfirm
)

logger = logging.getLogger(__name__)
settings = get_settings()

# Account lockout configuration
MAX_FAILED_ATTEMPTS = 5
LOCKOUT_DURATION_MINUTES = 30


class UserServiceError(Exception):
    """Base user service error."""
    pass


class UserAlreadyExistsError(UserServiceError):
    """User already exists error."""
    pass


class OrganizationNotFoundError(UserServiceError):
    """Organization not found error."""
    pass


class InvalidPasswordError(UserServiceError):
    """Invalid password error."""
    pass


class UserService:
    """
    Comprehensive user management service with authentication and organization handling.
    """
    
    def __init__(self, db: Session):
        self.db = db
        self.audit_logger = SecurityAuditLogger()
        print(f"UserService initialized")
    
    async def register_user(
        self, 
        registration_data: UserRegistrationRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[User, Organization, Dict[str, str]]:
        """
        Register a new user with organization.
        
        Args:
            registration_data: User registration data
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (User, Organization, tokens)
            
        Raises:
            UserAlreadyExistsError: If user already exists
            UserServiceError: If registration fails
        """
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(User.email == registration_data.email).first()
            if existing_user:
                raise UserAlreadyExistsError("User with this email already exists")
            
            # Check if username is taken (if provided)
            username = registration_data.username or registration_data.email.split('@')[0]
            existing_username = self.db.query(User).filter(User.username == username).first()
            if existing_username:
                # Generate unique username
                base_username = username
                counter = 1
                while existing_username:
                    username = f"{base_username}_{counter}"
                    existing_username = self.db.query(User).filter(User.username == username).first()
                    counter += 1
            
            # Handle organization
            organization = await self._handle_organization_for_registration(registration_data)
            
            # Hash password
            hashed_password = auth_service.hash_password(registration_data.password)
            
            # Generate email verification token
            verification_token = auth_service.generate_verification_token()
            verification_expires = datetime.now(timezone.utc) + timedelta(hours=24)
            
            # Create user
            user = User(
                email=registration_data.email,
                username=username,
                first_name=registration_data.first_name,
                last_name=registration_data.last_name,
                full_name=f"{registration_data.first_name} {registration_data.last_name}",
                hashed_password=hashed_password,
                phone=registration_data.phone,
                organization_id=organization.id,
                email_verification_token=verification_token,
                email_verification_expires=verification_expires,
                role="admin" if not self._has_existing_users(organization.id) else "recruiter",
                is_verified=False,  # Require email verification
                password_changed_at=datetime.now(timezone.utc)
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            # Create authentication tokens (even though not verified, for better UX)
            tokens = auth_service.create_tokens(
                user_id=user.id,
                email=user.email,
                role=user.role,
                organization_id=user.organization_id
            )
            
            # Create session
            await self._create_user_session(user, tokens, ip_address, user_agent)
            
            # Log registration
            logger.info(f"User registered successfully: {user.email} (ID: {user.id})")
            
            # TODO: Send verification email
            # await self._send_verification_email(user, verification_token)
            
            return user, organization, tokens
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Database integrity error during registration: {e}")
            raise UserAlreadyExistsError("User with this email or username already exists")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during user registration: {e}")
            raise UserServiceError(f"Registration failed: {str(e)}")
    
    async def authenticate_user(
        self, 
        login_data: UserLoginRequest,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[User, Dict[str, str]]:
        """
        Authenticate user and create session.
        
        Args:
            login_data: User login data
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (User, tokens)
            
        Raises:
            InvalidCredentialsError: If credentials are invalid
            AccountLockedError: If account is locked
            EmailNotVerifiedError: If email is not verified
        """
        try:
            # Get user by email
            user = self.db.query(User).filter(User.email == login_data.email).first()
            logger.info(f"User: {user}")
            print(f"User: {user}")
            if not user:
                self.audit_logger.log_login_attempt(
                    email=login_data.email,
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    failure_reason="User not found"
                )
                raise InvalidCredentialsError("Invalid email or password")
            
            # Check if account is locked
            if user.account_locked_until and user.account_locked_until > datetime.now(timezone.utc):
                self.audit_logger.log_login_attempt(
                    email=login_data.email,
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    failure_reason="Account locked"
                )
                raise AccountLockedError("Account is temporarily locked due to failed login attempts")
            
            # Verify password
            if not auth_service.verify_password(login_data.password, user.hashed_password):
                await self._handle_failed_login(user, ip_address, user_agent)
                raise InvalidCredentialsError("Invalid email or password")
            
            # Check if user is active
            if not user.is_active:
                self.audit_logger.log_login_attempt(
                    email=login_data.email,
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    failure_reason="Account deactivated"
                )
                raise InvalidCredentialsError("Account is deactivated")
            
            # Check if email is verified (optional based on settings)
            if not user.is_verified and not settings.DEBUG:
                self.audit_logger.log_login_attempt(
                    email=login_data.email,
                    success=False,
                    ip_address=ip_address,
                    user_agent=user_agent,
                    failure_reason="Email not verified"
                )
                raise EmailNotVerifiedError("Please verify your email address before logging in")
            
            # Reset failed login attempts on successful login
            if user.failed_login_attempts > 0:
                user.failed_login_attempts = 0
                user.account_locked_until = None
            
            # Update last login
            user.last_login = datetime.now(timezone.utc)
            user.last_activity = datetime.now(timezone.utc)
            
            self.db.commit()
            
            # Create authentication tokens
            tokens = auth_service.create_tokens(
                user_id=user.id,
                email=user.email,
                role=user.role,
                organization_id=user.organization_id
            )
            
            # Create session
            await self._create_user_session(user, tokens, ip_address, user_agent)
            
            # Log successful login
            self.audit_logger.log_login_attempt(
                email=login_data.email,
                success=True,
                ip_address=ip_address,
                user_agent=user_agent
            )
            
            logger.info(f"User authenticated successfully: {user.email} (ID: {user.id})")
            
            return user, tokens
            
        except (InvalidCredentialsError, AccountLockedError, EmailNotVerifiedError):
            raise
        except Exception as e:
            logger.error(f"Error during user authentication: {e}")
            raise UserServiceError(f"Authentication failed: {str(e)}")
    
    async def refresh_user_token(
        self, 
        refresh_token: str,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> Tuple[User, Dict[str, str]]:
        """
        Refresh user authentication token.
        
        Args:
            refresh_token: Refresh token
            ip_address: Client IP address
            user_agent: Client user agent
            
        Returns:
            Tuple of (User, new_tokens)
            
        Raises:
            InvalidCredentialsError: If refresh token is invalid
        """
        try:
            # Decode refresh token
            token_payload = auth_service.decode_refresh_token(refresh_token)
            
            # Get user
            user = self.db.query(User).filter(User.id == token_payload["sub"]).first()
            if not user or not user.is_active:
                raise InvalidCredentialsError("Invalid refresh token")
            
            # Verify session exists and is active
            session = self.db.query(UserSession).filter(
                UserSession.user_id == user.id,
                UserSession.refresh_token == refresh_token,
                UserSession.is_active == True,
                UserSession.expires_at > datetime.now(timezone.utc)
            ).first()
            
            if not session:
                raise InvalidCredentialsError("Invalid or expired refresh token")
            
            # Create new tokens
            new_tokens = auth_service.create_tokens(
                user_id=user.id,
                email=user.email,
                role=user.role,
                organization_id=user.organization_id
            )
            
            # Update session with new tokens
            session.session_token = new_tokens["access_token"]
            session.refresh_token = new_tokens["refresh_token"]
            session.last_activity = datetime.now(timezone.utc)
            session.expires_at = datetime.now(timezone.utc) + timedelta(
                minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
            )
            
            # Update user activity
            user.last_activity = datetime.now(timezone.utc)
            
            self.db.commit()
            
            logger.info(f"Token refreshed for user: {user.email} (ID: {user.id})")
            
            return user, new_tokens
            
        except Exception as e:
            logger.error(f"Error during token refresh: {e}")
            raise InvalidCredentialsError("Invalid refresh token")
    
    async def logout_user(
        self, 
        user: User, 
        access_token: str,
        revoke_all_sessions: bool = False
    ) -> bool:
        """
        Logout user and revoke session(s).
        
        Args:
            user: User object
            access_token: Current access token
            revoke_all_sessions: Whether to revoke all user sessions
            
        Returns:
            True if logout successful
        """
        try:
            if revoke_all_sessions:
                # Revoke all user sessions
                self.db.query(UserSession).filter(
                    UserSession.user_id == user.id,
                    UserSession.is_active == True
                ).update({
                    "is_active": False,
                    "revoked_at": datetime.now(timezone.utc)
                })
            else:
                # Revoke current session only
                self.db.query(UserSession).filter(
                    UserSession.user_id == user.id,
                    UserSession.session_token == access_token,
                    UserSession.is_active == True
                ).update({
                    "is_active": False,
                    "revoked_at": datetime.now(timezone.utc)
                })
            
            self.db.commit()
            
            logger.info(f"User logged out: {user.email} (ID: {user.id})")
            return True
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during logout for user {user.id}: {e}")
            return False
    
    async def create_user(
        self, 
        user_data: UserCreateRequest, 
        created_by: User
    ) -> User:
        """
        Create a new user (admin function).
        
        Args:
            user_data: User creation data
            created_by: User creating the new user
            
        Returns:
            Created user object
            
        Raises:
            UserAlreadyExistsError: If user already exists
            UserServiceError: If creation fails
        """
        try:
            # Check if user already exists
            existing_user = self.db.query(User).filter(
                (User.email == user_data.email) | (User.username == user_data.username)
            ).first()
            
            if existing_user:
                raise UserAlreadyExistsError("User with this email or username already exists")
            
            # Use default initial password and force change on first login
            temp_password = settings.DEFAULT_NEW_USER_PASSWORD
            hashed_password = auth_service.hash_password(temp_password)
            
            # Generate email verification token
            verification_token = auth_service.generate_verification_token()
            verification_expires = datetime.now(timezone.utc) + timedelta(hours=24)
            
            # Create user
            user = User(
                email=user_data.email,
                username=user_data.username,
                first_name=user_data.first_name,
                last_name=user_data.last_name,
                full_name=f"{user_data.first_name} {user_data.last_name}",
                hashed_password=hashed_password,
                phone=user_data.phone,
                role=user_data.role,
                is_active=user_data.is_active,
                is_verified=False,
                organization_id=created_by.organization_id,
                email_verification_token=verification_token,
                email_verification_expires=verification_expires,
                password_changed_at=datetime.now(timezone.utc),
                must_change_password=True
            )
            
            self.db.add(user)
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"User created by admin: {user.email} (ID: {user.id}) by {created_by.email}")
            
            # TODO: Send invitation email with temporary password
            # if user_data.send_invitation:
            #     await self._send_invitation_email(user, temp_password, verification_token)
            
            return user
            
        except IntegrityError as e:
            self.db.rollback()
            logger.error(f"Database integrity error during user creation: {e}")
            raise UserAlreadyExistsError("User with this email or username already exists")
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error during user creation: {e}")
            raise UserServiceError(f"User creation failed: {str(e)}")
    
    async def update_user_profile(
        self, 
        user: User, 
        profile_data: UserProfileUpdateRequest
    ) -> User:
        """
        Update user profile.
        
        Args:
            user: User to update
            profile_data: Profile update data
            
        Returns:
            Updated user object
        """
        try:
            # Update fields if provided
            if profile_data.first_name is not None:
                user.first_name = profile_data.first_name
            if profile_data.last_name is not None:
                user.last_name = profile_data.last_name
            
            # Update full name if first or last name changed
            if profile_data.first_name is not None or profile_data.last_name is not None:
                user.full_name = f"{user.first_name} {user.last_name}"
            
            if profile_data.phone is not None:
                user.phone = profile_data.phone
            if profile_data.timezone is not None:
                user.timezone = profile_data.timezone
            if profile_data.avatar_url is not None:
                user.avatar_url = profile_data.avatar_url
            if profile_data.preferences is not None:
                user.preferences = profile_data.preferences
            if profile_data.ai_agent_preferences is not None:
                user.ai_agent_preferences = profile_data.ai_agent_preferences
            if profile_data.default_llm_provider is not None:
                user.default_llm_provider = profile_data.default_llm_provider
            
            user.updated_at = datetime.now(timezone.utc)
            
            self.db.commit()
            self.db.refresh(user)
            
            logger.info(f"User profile updated: {user.email} (ID: {user.id})")
            
            return user
            
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error updating user profile {user.id}: {e}")
            raise UserServiceError(f"Profile update failed: {str(e)}")
    
    async def change_password(
        self, 
        user: User, 
        password_data: PasswordChangeRequest,
        ip_address: Optional[str] = None
    ) -> bool:
        """
        Change user password.
        
        Args:
            user: User changing password
            password_data: Password change data
            ip_address: Client IP address
            
        Returns:
            True if password changed successfully
            
        Raises:
            InvalidPasswordError: If current password is incorrect
        """
        try:
            # Verify current password
            if not auth_service.verify_password(password_data.current_password, user.hashed_password):
                raise InvalidPasswordError("Current password is incorrect")
            
            # Hash new password
            new_hashed_password = auth_service.hash_password(password_data.new_password)
            
            # Update password
            user.hashed_password = new_hashed_password
            user.password_changed_at = datetime.now(timezone.utc)
            user.updated_at = datetime.now(timezone.utc)
            # Clear first-login requirement after successful change
            user.must_change_password = False
            
            self.db.commit()
            
            # Log password change
            self.audit_logger.log_password_change(str(user.id), user.email, ip_address)
            
            logger.info(f"Password changed for user: {user.email} (ID: {user.id})")
            
            return True
            
        except InvalidPasswordError:
            raise
        except Exception as e:
            self.db.rollback()
            logger.error(f"Error changing password for user {user.id}: {e}")
            raise UserServiceError(f"Password change failed: {str(e)}")
    
    # Private helper methods
    
    async def _handle_organization_for_registration(
        self, 
        registration_data: UserRegistrationRequest
    ) -> Organization:
        """Handle organization creation or selection during registration."""
        if registration_data.organization_name:
            # Create new organization
            domain = registration_data.organization_domain or registration_data.email.split('@')[1]
            
            # Check if organization with domain already exists
            existing_org = self.db.query(Organization).filter(Organization.domain == domain).first()
            if existing_org:
                return existing_org
            
            organization = Organization(
                name=registration_data.organization_name,
                domain=domain,
                industry=registration_data.organization_industry
            )
            
            self.db.add(organization)
            self.db.flush()  # Get ID without committing
            
            return organization
        else:
            # Use default organization or create one based on email domain
            domain = registration_data.email.split('@')[1]
            existing_org = self.db.query(Organization).filter(Organization.domain == domain).first()
            
            if existing_org:
                return existing_org
            
            # Create organization based on email domain
            organization = Organization(
                name=domain.split('.')[0].title(),
                domain=domain
            )
            
            self.db.add(organization)
            self.db.flush()
            
            return organization
    
    def _has_existing_users(self, organization_id: UUID) -> bool:
        """Check if organization has existing users."""
        return self.db.query(User).filter(User.organization_id == organization_id).count() > 0
    
    async def _create_user_session(
        self, 
        user: User, 
        tokens: Dict[str, str],
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ) -> UserSession:
        """Create user session record."""
        session = UserSession(
            user_id=user.id,
            session_token=tokens["access_token"],
            refresh_token=tokens["refresh_token"],
            ip_address=ip_address,
            user_agent=user_agent,
            expires_at=datetime.now(timezone.utc) + timedelta(
                minutes=settings.REFRESH_TOKEN_EXPIRE_MINUTES
            )
        )
        
        self.db.add(session)
        self.db.commit()
        
        return session
    
    async def _handle_failed_login(
        self, 
        user: User, 
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None
    ):
        """Handle failed login attempt."""
        user.failed_login_attempts = (user.failed_login_attempts or 0) + 1
        
        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.account_locked_until = datetime.now(timezone.utc) + timedelta(
                minutes=LOCKOUT_DURATION_MINUTES
            )
            self.audit_logger.log_account_lockout(user.email, ip_address)
        
        self.db.commit()
        
        self.audit_logger.log_login_attempt(
            email=user.email,
            success=False,
            ip_address=ip_address,
            user_agent=user_agent,
            failure_reason="Invalid password"
        )
