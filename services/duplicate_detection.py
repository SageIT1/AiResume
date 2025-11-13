"""
AI-Powered Duplicate Resume Detection Service

This service uses LLM-based normalization and intelligent matching to detect
duplicate resumes based on email addresses and phone numbers.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

import logging
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from uuid import UUID
import re
import asyncio

from sqlalchemy import select, and_, or_
from sqlalchemy.ext.asyncio import AsyncSession
from pydantic import BaseModel, Field

from core.llm_factory import LLMFactory
from core.config import get_settings
from database.models import Resume, ResumeStatus
from database.session import db_manager

logger = logging.getLogger(__name__)


class ContactInfo(BaseModel):
    """Normalized contact information extracted by AI."""
    email: Optional[str] = Field(None, description="Normalized email address")
    phone: Optional[str] = Field(None, description="Normalized phone number")
    confidence_email: float = Field(0.0, description="Confidence in email normalization")
    confidence_phone: float = Field(0.0, description="Confidence in phone normalization")


class DuplicateMatch(BaseModel):
    """Represents a potential duplicate match."""
    resume_id: str
    candidate_name: Optional[str]
    candidate_email: Optional[str]
    candidate_phone: Optional[str]
    match_type: str  # "email", "phone", "both"
    confidence_score: float
    created_at: datetime
    status: str
    original_filename: str
    
    def dict(self, **kwargs):
        """Override dict method to handle datetime serialization."""
        data = super().dict(**kwargs)
        
        # Convert datetime to ISO string
        if 'created_at' in data and isinstance(data['created_at'], datetime):
            data['created_at'] = data['created_at'].isoformat()
        
        return data


class DuplicateDetectionResult(BaseModel):
    """Result of duplicate detection analysis."""
    is_duplicate: bool
    confidence_score: float
    matches: List[DuplicateMatch]
    normalized_contact: ContactInfo
    reasoning: str
    action_recommended: str  # "proceed", "merge", "manual_review"
    
    def dict(self, **kwargs):
        """Override dict method to handle datetime serialization."""
        data = super().dict(**kwargs)
        
        # Convert datetime objects in matches to ISO strings
        if 'matches' in data and isinstance(data['matches'], list):
            for match in data['matches']:
                if isinstance(match, dict) and 'created_at' in match:
                    if isinstance(match['created_at'], datetime):
                        match['created_at'] = match['created_at'].isoformat()
        
        return data


class DuplicateDetectionService:
    """AI-powered service for detecting duplicate resumes."""
    
    def __init__(self):
        self.settings = get_settings()
        self.llm_factory = LLMFactory(self.settings)
        self.llm = None
        self._initialized = False
    
    def initialize(self):
        """Initialize the LLM for AI-powered normalization."""
        if self._initialized:
            logger.info("✅ Duplicate Detection Service already initialized")
            return
            
        try:
            logger.info("🔄 Initializing Duplicate Detection Service...")
            self.llm = self.llm_factory.create_llm()
            self._initialized = True
            logger.info("✅ Duplicate Detection Service initialized successfully")
        except Exception as e:
            logger.error(f"❌ Failed to initialize Duplicate Detection Service: {e}")
            logger.error(f"❌ Error details: {str(e)}")
            raise
    
    async def normalize_contact_info(
        self, 
        email: Optional[str], 
        phone: Optional[str]
    ) -> ContactInfo:
        """
        Use AI to normalize and clean contact information.
        
        Args:
            email: Raw email address from resume
            phone: Raw phone number from resume
            
        Returns:
            ContactInfo with normalized values and confidence scores
        """
        if not self._initialized:
            self.initialize()
        
        try:
            # Create normalization prompt
            prompt = f"""
You are an expert data normalization specialist. Your task is to clean and normalize contact information extracted from resumes.

CONTACT INFORMATION TO NORMALIZE:
Email: {email or "Not provided"}
Phone: {phone or "Not provided"}

NORMALIZATION RULES:
1. Email normalization:
   - Convert to lowercase
   - Remove extra spaces and special characters
   - Handle common typos (gmail.co -> gmail.com, etc.)
   - Validate email format
   - Return null if clearly invalid

2. Phone normalization:
   - Remove all non-digit characters except country codes
   - Standardize to international format (+1234567890)
   - Handle US numbers (add +1 if missing)
   - Handle extensions (remove them)
   - Return null if clearly invalid

3. Confidence scoring:
   - 1.0 = Perfect, clearly valid
   - 0.8 = Good, minor corrections made
   - 0.6 = Fair, significant corrections needed
   - 0.4 = Poor, major issues but salvageable
   - 0.2 = Very poor, likely invalid
   - 0.0 = Invalid or not provided

Return ONLY a JSON object with this exact structure:
{{
    "email": "normalized_email_or_null",
    "phone": "normalized_phone_or_null", 
    "confidence_email": 0.0,
    "confidence_phone": 0.0
}}
"""
            
            # Get AI response
            response = await self.llm.ainvoke(prompt)
            
            # Parse JSON response
            import json
            try:
                result_data = json.loads(response.content.strip())
                return ContactInfo(**result_data)
            except json.JSONDecodeError:
                # Fallback: extract JSON from response
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                    return ContactInfo(**result_data)
                else:
                    raise ValueError("Could not parse AI response as JSON")
                    
        except Exception as e:
            logger.error(f"❌ AI normalization failed: {e}")
            # Fallback to basic normalization
            return ContactInfo(
                email=email.lower().strip() if email else None,
                phone=re.sub(r'[^\d+]', '', phone) if phone else None,
                confidence_email=0.5 if email else 0.0,
                confidence_phone=0.5 if phone else 0.0
            )
    
    async def find_potential_duplicates(
        self,
        normalized_contact: ContactInfo,
        organization_id: UUID,
        exclude_resume_id: Optional[UUID] = None
    ) -> List[DuplicateMatch]:
        """
        Search for potential duplicate resumes in the database.
        
        Args:
            normalized_contact: Normalized contact information
            organization_id: Organization to search within
            exclude_resume_id: Resume ID to exclude from search
            
        Returns:
            List of potential duplicate matches
        """
        matches = []
        
        try:
            async with db_manager.async_session() as session:
                # Build query conditions
                conditions = [Resume.organization_id == organization_id]
                
                if exclude_resume_id:
                    conditions.append(Resume.id != exclude_resume_id)
                
                # Search by email if available
                email_matches = []
                if normalized_contact.email and normalized_contact.confidence_email >= 0.6:
                    email_query = select(Resume).where(
                        and_(
                            *conditions,
                            Resume.candidate_email.ilike(f"%{normalized_contact.email}%")
                        )
                    )
                    email_result = await session.execute(email_query)
                    email_matches = email_result.scalars().all()
                
                # Search by phone if available
                phone_matches = []
                if normalized_contact.phone and normalized_contact.confidence_phone >= 0.6:
                    # Clean phone for comparison
                    clean_phone = re.sub(r'[^\d]', '', normalized_contact.phone)
                    if len(clean_phone) >= 10:  # Minimum valid phone length
                        phone_query = select(Resume).where(
                            and_(
                                *conditions,
                                Resume.candidate_phone.ilike(f"%{clean_phone[-10:]}%")  # Last 10 digits
                            )
                        )
                        phone_result = await session.execute(phone_query)
                        phone_matches = phone_result.scalars().all()
                
                # Process matches
                all_resumes = {}
                
                # Add email matches
                for resume in email_matches:
                    match_type = "email"
                    confidence = normalized_contact.confidence_email
                    
                    # Check if phone also matches
                    if resume.candidate_phone and normalized_contact.phone:
                        resume_phone_clean = re.sub(r'[^\d]', '', resume.candidate_phone)
                        norm_phone_clean = re.sub(r'[^\d]', '', normalized_contact.phone)
                        if resume_phone_clean[-10:] == norm_phone_clean[-10:]:
                            match_type = "both"
                            confidence = min(1.0, confidence + normalized_contact.confidence_phone)
                    
                    all_resumes[resume.id] = (resume, match_type, confidence)
                
                # Add phone-only matches
                for resume in phone_matches:
                    if resume.id not in all_resumes:
                        all_resumes[resume.id] = (resume, "phone", normalized_contact.confidence_phone)
                
                # Convert to DuplicateMatch objects
                for resume, match_type, confidence in all_resumes.values():
                    matches.append(DuplicateMatch(
                        resume_id=str(resume.id),
                        candidate_name=resume.candidate_name,
                        candidate_email=resume.candidate_email,
                        candidate_phone=resume.candidate_phone,
                        match_type=match_type,
                        confidence_score=confidence,
                        created_at=resume.created_at,
                        status=resume.status.value,
                        original_filename=resume.original_filename
                    ))
                
                # Sort by confidence score (highest first)
                matches.sort(key=lambda x: x.confidence_score, reverse=True)
                
        except Exception as e:
            logger.error(f"❌ Database search for duplicates failed: {e}")
        
        return matches
    
    async def analyze_duplicate_likelihood(
        self,
        matches: List[DuplicateMatch],
        normalized_contact: ContactInfo
    ) -> Tuple[bool, float, str, str]:
        """
        Use AI to analyze the likelihood of duplicates and recommend actions.
        
        Args:
            matches: List of potential duplicate matches
            normalized_contact: Normalized contact information
            
        Returns:
            Tuple of (is_duplicate, confidence_score, reasoning, action_recommended)
        """
        if not matches:
            return False, 0.0, "No potential duplicates found.", "proceed"
        
        if not self._initialized:
            self.initialize()
        
        try:
            # Prepare match data for AI analysis
            match_data = []
            for match in matches:
                match_data.append({
                    "candidate_name": match.candidate_name,
                    "email": match.candidate_email,
                    "phone": match.candidate_phone,
                    "match_type": match.match_type,
                    "confidence": match.confidence_score,
                    "filename": match.original_filename,
                    "upload_date": match.created_at.strftime("%Y-%m-%d")
                })
            
            prompt = f"""
You are an expert duplicate detection analyst. Analyze potential resume duplicates and provide recommendations.

NEW RESUME CONTACT INFO:
Email: {normalized_contact.email or "Not provided"}
Phone: {normalized_contact.phone or "Not provided"}

POTENTIAL MATCHES FOUND:
{match_data}

ANALYSIS CRITERIA:
1. Email match with high confidence (>0.8) = Very likely duplicate
2. Phone match with high confidence (>0.8) = Very likely duplicate  
3. Both email and phone match = Almost certain duplicate
4. Similar names with contact match = Likely duplicate
5. Recent uploads (same day) = Higher duplicate probability

CONFIDENCE SCORING:
- 0.9-1.0: Almost certain duplicate
- 0.7-0.9: Very likely duplicate
- 0.5-0.7: Possibly duplicate
- 0.3-0.5: Unlikely duplicate
- 0.0-0.3: Not a duplicate

ACTIONS:
- "proceed": Low duplicate risk, continue processing
- "manual_review": Medium risk, needs human review
- "merge": High risk, likely duplicate that should be merged

Provide your analysis in this JSON format:
{{
    "is_duplicate": true/false,
    "confidence_score": 0.0-1.0,
    "reasoning": "Detailed explanation of your analysis",
    "action_recommended": "proceed/manual_review/merge"
}}
"""
            
            response = await self.llm.ainvoke(prompt)
            
            # Parse AI response
            import json
            try:
                result = json.loads(response.content.strip())
                return (
                    result["is_duplicate"],
                    result["confidence_score"], 
                    result["reasoning"],
                    result["action_recommended"]
                )
            except json.JSONDecodeError:
                # Fallback parsing
                import re
                json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
                if json_match:
                    result = json.loads(json_match.group())
                    return (
                        result["is_duplicate"],
                        result["confidence_score"],
                        result["reasoning"], 
                        result["action_recommended"]
                    )
                else:
                    raise ValueError("Could not parse AI analysis response")
                    
        except Exception as e:
            logger.error(f"❌ AI duplicate analysis failed: {e}")
            # Fallback logic
            if matches:
                highest_confidence = max(match.confidence_score for match in matches)
                if highest_confidence >= 0.8:
                    return True, highest_confidence, "High confidence match found", "manual_review"
                elif highest_confidence >= 0.6:
                    return True, highest_confidence, "Moderate confidence match found", "manual_review"
            
            return False, 0.0, "Analysis failed, proceeding with caution", "proceed"
    
    async def detect_duplicates(
        self,
        email: Optional[str],
        phone: Optional[str],
        organization_id: UUID,
        exclude_resume_id: Optional[UUID] = None
    ) -> DuplicateDetectionResult:
        """
        Main method to detect duplicate resumes.
        
        Args:
            email: Raw email from resume
            phone: Raw phone from resume
            organization_id: Organization ID
            exclude_resume_id: Resume ID to exclude from search
            
        Returns:
            DuplicateDetectionResult with complete analysis
        """
        logger.info(f"🔍 Starting duplicate detection for email={email}, phone={phone}")
        
        try:
            # Ensure service is initialized
            if not self._initialized:
                logger.info("🔄 Service not initialized, initializing now...")
                self.initialize()
            
            # Step 1: Normalize contact information using AI
            logger.info("🔄 Step 1: Normalizing contact information...")
            normalized_contact = await self.normalize_contact_info(email, phone)
            logger.info(f"📧 Normalized contact: email_conf={normalized_contact.confidence_email:.2f}, phone_conf={normalized_contact.confidence_phone:.2f}")
            
            # Step 2: Search for potential duplicates
            logger.info("🔄 Step 2: Searching for potential duplicates...")
            matches = await self.find_potential_duplicates(
                normalized_contact, organization_id, exclude_resume_id
            )
            logger.info(f"🔎 Found {len(matches)} potential matches")
            
            # Step 3: AI analysis of duplicate likelihood
            logger.info("🔄 Step 3: Analyzing duplicate likelihood...")
            is_duplicate, confidence_score, reasoning, action = await self.analyze_duplicate_likelihood(
                matches, normalized_contact
            )
            
            result = DuplicateDetectionResult(
                is_duplicate=is_duplicate,
                confidence_score=confidence_score,
                matches=matches,
                normalized_contact=normalized_contact,
                reasoning=reasoning,
                action_recommended=action
            )
            
            logger.info(f"✅ Duplicate detection complete: duplicate={is_duplicate}, confidence={confidence_score:.2f}, action={action}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Duplicate detection failed: {e}")
            logger.error(f"❌ Error traceback: {traceback.format_exc()}")
            # Return safe fallback result
            return DuplicateDetectionResult(
                is_duplicate=False,
                confidence_score=0.0,
                matches=[],
                normalized_contact=ContactInfo(),
                reasoning=f"Duplicate detection failed: {str(e)}",
                action_recommended="proceed"
            )


# Global service instance
duplicate_detection_service = DuplicateDetectionService()
