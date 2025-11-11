"""
AI Recruit - Resume API Endpoints
Resume management and AI analysis endpoints.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

import asyncio
import base64
import io
import json
import logging
import mimetypes
import os
import re
import traceback
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any, Tuple
from uuid import UUID, uuid4

from celery.exceptions import SoftTimeLimitExceeded
from docx import Document
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query, Request
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from sqlalchemy import select, func, or_, and_, case
from sqlalchemy.ext.asyncio import AsyncSession

from agents.orchestrator import RecruitmentOrchestrator
from agents.search import SemanticSearchAgent
from agents.similarity import SimilarProfilesAgent
from core.config import get_settings
from core.llm_factory import LLMFactory
from core.security import get_current_user
from core.services.duplicate_detection import duplicate_detection_service
from database.models import User, Resume, ResumeStatus, QualityScore, JobMatch
from database.session import get_db, db_manager
from tasks.resume_processing import (
    process_uploaded_resume_task,
    reprocess_resume_task,
)
from utils.candidate_id_generator import generate_candidate_id
from utils.document_metadata import DocumentMetadataManager
from utils.llm_resume_formatter import LLMResumeFormatter
from utils.parsers.document_parser import DocumentParser
from utils.resume_generator import ResumeGenerator

logger = logging.getLogger(__name__)

# Removed generate_name_variations function - embeddings handle similarity automatically

def normalize_phone_number(phone: str) -> str:
    """
    Normalize phone number by removing all non-digit characters.
    This allows consistent phone number searching regardless of format.
    """
    if not phone:
        return ""
    return re.sub(r'\D', '', phone)

router = APIRouter()


async def perform_initial_duplicate_check(
    file_content: bytes,
    filename: str,
    organization_id: UUID,
    orchestrator: RecruitmentOrchestrator
) -> Tuple[Optional[str], Optional[str], Optional[Dict]]:
    """
    Perform initial duplicate detection using basic text extraction.
    
    Returns:
        Tuple of (email, phone, duplicate_result)
    """
    try:
        # Quick text extraction to get email and phone for duplicate check
        parser = DocumentParser()
        extracted_text = await parser.extract_text(file_content, filename)
        
        if not extracted_text:
            logger.warning("Could not extract text for duplicate detection")
            return None, None, None
        
        # Use AI to extract contact info from text
        settings = get_settings()
        llm_factory = LLMFactory(settings)
        llm = llm_factory.create_llm()
        
        contact_prompt = f"""
Extract the email address and phone number from this resume text.

RESUME TEXT:
{extracted_text[:2000]}  # First 2000 chars for quick processing

Return ONLY a JSON object with this format:
{{
    "email": "extracted_email_or_null",
    "phone": "extracted_phone_or_null"
}}
"""
        
        response = await llm.ainvoke(contact_prompt)
        
        # Parse contact info
        
        try:
            contact_data = json.loads(response.content.strip())
        except json.JSONDecodeError:
            # Fallback parsing
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if json_match:
                contact_data = json.loads(json_match.group())
            else:
                return None, None, None
        
        email = contact_data.get("email")
        phone = contact_data.get("phone")
        
        # Perform duplicate detection if we have contact info
        if email or phone:
            duplicate_result = await duplicate_detection_service.detect_duplicates(
                email=email,
                phone=phone,
                organization_id=organization_id
            )
            return email, phone, duplicate_result.dict()
        
        return email, phone, None
        
    except Exception as e:
        logger.error(f"❌ Initial duplicate check failed: {e}")
        return None, None, None


def process_base64_file(base64_data: str, filename: str, content_type: Optional[str] = None) -> Tuple[bytes, str, str]:
    """
    Process base64 encoded file data and return file content, filename, and content type.
    
    Args:
        base64_data: Base64 encoded file data
        filename: Original filename with extension
        content_type: MIME type of the file (optional, will be detected if not provided)
    
    Returns:
        Tuple of (file_content, filename, content_type)
    
    Raises:
        HTTPException: If base64 data is invalid or file type is not supported
    """
    try:
        # Decode base64 data
        file_content = base64.b64decode(base64_data)
        
        # Validate file size (basic check)
        if len(file_content) == 0:
            raise HTTPException(status_code=400, detail="Empty file data")
        
        # Detect content type if not provided
        if not content_type:
            content_type, _ = mimetypes.guess_type(filename)
            if not content_type:
                # Fallback based on file extension
                ext = filename.split('.')[-1].lower()
                content_type_map = {
                    'pdf': 'application/pdf',
                    'doc': 'application/msword',
                    'docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
                    'txt': 'text/plain'
                }
                content_type = content_type_map.get(ext, 'application/octet-stream')
        
        # Validate file extension
        settings = get_settings()
        allowed_types = settings.ALLOWED_FILE_TYPES
        file_ext = filename.split('.')[-1].lower()
        if file_ext not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"File type .{file_ext} not allowed. Allowed types: {allowed_types}"
            )
        
        # Validate file size
        max_size = settings.MAX_FILE_SIZE_MB * 1024 * 1024
        if len(file_content) > max_size:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit"
            )
        
        return file_content, filename, content_type
        
    except Exception as e:
        if isinstance(e, HTTPException):
            raise
        raise HTTPException(status_code=400, detail=f"Invalid base64 data: {str(e)}")


class ResumeUploadResponse(BaseModel):
    """Response model for resume upload."""
    id: str
    filename: str
    status: str
    azure_url: Optional[str] = None
    upload_timestamp: str
    message: str
    # Duplicate detection information
    is_duplicate: Optional[bool] = None
    duplicate_confidence: Optional[float] = None
    duplicate_matches: Optional[List[Dict[str, Any]]] = None
    duplicate_action: Optional[str] = None
    duplicate_reasoning: Optional[str] = None


class ResumeAnalysisResponse(BaseModel):
    """Response model for resume analysis results."""
    id: str
    candidate_id: str
    status: str
    blacklist: Optional[bool] = False
    # Agent markdown outputs (exposed for UI rendering)
    agent_personal_info_markdown: Optional[str] = None
    agent_experience_markdown: Optional[str] = None
    agent_skills_markdown: Optional[str] = None
    agent_education_markdown: Optional[str] = None
    agent_quality_markdown: Optional[str] = None
    agent_reasoning_markdown: Optional[str] = None
    # Raw agent outputs for debugging/traceability
    agent_raw_outputs: Optional[Dict[str, Any]] = None
    analysis: Optional[Dict[str, Any]] = None
    extracted_text: Optional[str] = None
    quality_score: Optional[float] = None
    confidence_score: Optional[float] = None
    candidate_name: Optional[str] = None
    resume_filename: Optional[str] = None
    processing_time: Optional[float] = None
    error_message: Optional[str] = None


class ResumeListResponse(BaseModel):
    """Response model for resume list."""
    resumes: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_next: bool


class SemanticSearchRequest(BaseModel):
    """Request model for semantic search."""
    query: str = Field(..., description="Natural language search query")
    max_results: int = Field(20, ge=1, le=100, description="Maximum number of results")
    include_analysis: bool = Field(True, description="Include detailed AI analysis in results")


class SemanticSearchResponse(BaseModel):
    """Response model for semantic search."""
    query: str
    search_intent: Dict[str, Any]
    results: List[Dict[str, Any]]
    total_matches: int
    search_time_ms: int
    suggestions: List[str]
    alternative_queries: List[str]


@router.post("/upload", response_model=ResumeUploadResponse)
async def upload_resume(
    request: Request,
    file: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Upload and process a resume using AI agents.
    
    This endpoint supports two upload methods:
    1. File upload: Provide 'file' parameter with multipart/form-data
    2. Base64 upload: Provide 'base64_data', 'filename', and optionally 'content_type' parameters
    
    The endpoint:
    1. Validates the uploaded file or base64 data
    2. Stores it in Azure Blob Storage
    3. Initiates AI analysis pipeline
    4. Returns immediate response with processing status
    """
    try:
        settings = get_settings()
        # Check content type to determine upload method
        content_type = request.headers.get("content-type", "")
        if content_type.startswith("multipart/form-data"):
            # File upload method
            if not file or not file.filename:
                raise HTTPException(status_code=400, detail="No file provided for multipart upload")
            upload_method = "file"
            file_content = await file.read()
            original_filename = file.filename
            detected_content_type = file.content_type
        elif content_type.startswith("application/json"):
            # Base64 upload method - parse JSON body
            try:
                body = await request.json()
                base64_data = body.get("base64_data")
                filename = body.get("filename")
                content_type_param = body.get("content_type")
                if not base64_data or not filename:
                    raise HTTPException(
                        status_code=400, 
                        detail="For JSON uploads, 'base64_data' and 'filename' parameters are required"
                    )
                upload_method = "base64"
                file_content, original_filename, detected_content_type = process_base64_file(
                    base64_data=base64_data,
                    filename=filename,
                    content_type=content_type_param
                )
            except Exception as e:
                if isinstance(e, HTTPException):
                    raise
                raise HTTPException(status_code=400, detail="Invalid JSON data")
        else:
            raise HTTPException(
                status_code=400, 
                detail="Content-Type must be either 'multipart/form-data' or 'application/json'"
            )
        # Validate file size
        if len(file_content) > settings.MAX_FILE_SIZE_MB * 1024 * 1024:
            raise HTTPException(
                status_code=400,
                detail=f"File size exceeds {settings.MAX_FILE_SIZE_MB}MB limit"
            )
        # Get orchestrator from app state
        orchestrator: RecruitmentOrchestrator = request.app.state.orchestrator
        # Perform initial duplicate detection
        logger.info(f"🔍 Performing duplicate detection for {original_filename}")
        email, phone, duplicate_result = await perform_initial_duplicate_check(
            file_content=file_content,
            filename=original_filename,
            organization_id=current_user.organization_id,
            orchestrator=orchestrator
        )
        # Check if duplicate action requires stopping upload
        duplicate_action = None
        duplicate_matches = []
        duplicate_confidence = 0.0
        duplicate_reasoning = "No duplicate detection performed"
        is_duplicate = False
        if duplicate_result:
            is_duplicate = duplicate_result.get("is_duplicate", False)
            duplicate_confidence = duplicate_result.get("confidence_score", 0.0)
            duplicate_matches = [match for match in duplicate_result.get("matches", [])]
            duplicate_action = duplicate_result.get("action_recommended", "proceed")
            duplicate_reasoning = duplicate_result.get("reasoning", "")
            logger.info(f"🔍 Duplicate detection result: duplicate={is_duplicate}, confidence={duplicate_confidence:.2f}, action={duplicate_action}")
            # If high confidence duplicate, return warning but allow upload
            if is_duplicate and duplicate_confidence >= 0.8:
                logger.warning(f"⚠️ High confidence duplicate detected for {original_filename}")
        # Add Sage IT metadata to uploaded resume (DOCX, PDF, and DOC)
        processed_file_content = file_content
        processed_filename = original_filename
        if original_filename.lower().endswith(('.docx', '.pdf', '.doc')):
            try:
                logger.info(f"📄 Adding Sage IT metadata to uploaded {original_filename.split('.')[-1].upper()} resume...")
                metadata_manager = DocumentMetadataManager()
                # Create resume data for metadata
                resume_data = {
                    'candidate_name': 'Unknown',  # Will be updated after AI analysis
                    'extracted_text': '',  # Will be populated during processing
                    'quality_score': 0,  # Will be updated after analysis
                    'processed_date': datetime.now(timezone.utc).isoformat()
                }
                if original_filename.lower().endswith('.docx'):
                    # Handle DOCX files
                    doc = Document(io.BytesIO(file_content))
                    doc_with_metadata = metadata_manager.add_sage_it_metadata(doc, resume_data)
                    # Save the document with metadata to bytes
                    output_buffer = io.BytesIO()
                    doc_with_metadata.save(output_buffer)
                    processed_file_content = output_buffer.getvalue()  
                elif original_filename.lower().endswith('.pdf'):
                    # Handle PDF files
                    processed_file_content = metadata_manager.add_sage_it_metadata_to_pdf(file_content, resume_data)
                elif original_filename.lower().endswith('.doc'):
                    # Handle DOC files - add metadata directly to DOC file
                    logger.info(f"📄 Processing DOC file: {original_filename}")
                    try:
                        processed_file_content = metadata_manager.add_sage_it_metadata_to_doc(file_content, resume_data)
                        # Keep original filename as DOC (no conversion)
                        processed_filename = original_filename
                        logger.info(f"📄 DOC file processed with metadata: {processed_filename}")
                        logger.info(f"📄 DOC file size before: {len(file_content)} bytes, after: {len(processed_file_content)} bytes")
                        # Verify the processing worked
                        if processed_file_content == file_content:
                            logger.warning(f"⚠️ DOC metadata processing returned original content - metadata may not have been applied")
                        else:
                            logger.info(f"✅ DOC metadata processing successful - file size changed")
                    except Exception as doc_error:
                        logger.error(f"❌ DOC metadata processing failed: {doc_error}")
                        # Fall back to original content
                        processed_file_content = file_content
                        processed_filename = original_filename
                logger.info("✅ Sage IT metadata added to uploaded resume")
            except Exception as e:
                logger.warning(f"⚠️ Failed to add Sage IT metadata to uploaded resume: {e}")
                # Continue with original file if metadata addition fails
                processed_file_content = file_content
                processed_filename = original_filename
        # Upload to Azure Storage first (sync path), then queue background analysis
        upload_info = await orchestrator.azure_storage.upload_resume(
            file_content=processed_file_content,
            filename=processed_filename,
            user_id=str(current_user.id),
            metadata={
                "uploaded_by": current_user.username,
                "upload_ip": request.client.host if request.client else None,
                "duplicate_detected": is_duplicate,
                "duplicate_confidence": duplicate_confidence,
                "upload_method": upload_method,
                "sage_it_metadata_added": original_filename.lower().endswith('.docx')
            }
        )
        # Generate unique candidate ID using sync session (same as job ID generation)
        with db_manager.sync_session_context() as sync_db:
            candidate_id = generate_candidate_id(sync_db)
        # Create database record in PROCESSING state
        resume_record = Resume(
            organization_id=current_user.organization_id,
            candidate_id=candidate_id,
            original_filename=processed_filename,  # Use processed filename (may be converted from DOC to DOCX)
            file_size=len(processed_file_content),  # Use processed file size
            content_type=detected_content_type,
            azure_blob_name=upload_info.get("blob_name", ""),
            azure_container=upload_info.get("container", ""),
            azure_url=upload_info.get("url", ""),
            status=ResumeStatus.PROCESSING,
            processing_started_at=datetime.now(timezone.utc),
            uploaded_by_ip=request.client.host if request.client else None,
            # Duplicate detection fields
            is_duplicate=is_duplicate,
            duplicate_confidence_score=duplicate_confidence,
            duplicate_detection_result=duplicate_result or {},
            duplicate_matches=duplicate_matches,
            normalized_email=duplicate_result.get("normalized_contact", {}).get("email") if duplicate_result else email,
            normalized_phone=duplicate_result.get("normalized_contact", {}).get("phone") if duplicate_result else phone,
            duplicate_action_taken=duplicate_action
        )
        db.add(resume_record)
        await db.commit()
        await db.refresh(resume_record)
        # Enqueue background processing task
        process_uploaded_resume_task.delay(
            resume_id=str(resume_record.id),
            blob_name=resume_record.azure_blob_name,
            container=resume_record.azure_container,
            filename=original_filename,
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id),
            metadata={
                "uploaded_by": current_user.username,
                "upload_ip": request.client.host if request.client else None,
                "upload_method": upload_method,
            },
        )
        logger.info(f"✅ Resume upload accepted; background processing queued: {resume_record.id}")
        # Prepare response message based on duplicate detection and metadata addition
        message = "Resume uploaded; background analysis started"
        if original_filename.lower().endswith(('.docx', '.pdf', '.doc')):
            if original_filename.lower().endswith('.doc'):
                message += " (DOC file processed with Sage IT metadata added)"
            else:
                message += " (Sage IT metadata added)"
        if is_duplicate and duplicate_confidence >= 0.8:
            message = f"⚠️ Potential duplicate detected (confidence: {duplicate_confidence:.1%}). Resume uploaded for review."
            if original_filename.lower().endswith(('.docx', '.pdf', '.doc')):
                if original_filename.lower().endswith('.doc'):
                    message += " (DOC file processed with Sage IT metadata added)"
                else:
                    message += " (Sage IT metadata added)"
        elif is_duplicate and duplicate_confidence >= 0.6:
            message = f"⚠️ Possible duplicate detected (confidence: {duplicate_confidence:.1%}). Resume uploaded for analysis."
            if original_filename.lower().endswith(('.docx', '.pdf', '.doc')):
                if original_filename.lower().endswith('.doc'):
                    message += " (DOC file processed with Sage IT metadata added)"
                else:
                    message += " (Sage IT metadata added)"
        return ResumeUploadResponse(
            id=str(resume_record.id),
            filename=original_filename,
            status=resume_record.status.value,
            azure_url=resume_record.azure_url,
            upload_timestamp=resume_record.created_at.isoformat(),
            message=message,
            # Duplicate detection information
            is_duplicate=is_duplicate,
            duplicate_confidence=duplicate_confidence,
            duplicate_matches=[
                {
                    "resume_id": match.get("resume_id"),
                    "candidate_name": match.get("candidate_name"),
                    "match_type": match.get("match_type"),
                    "confidence": match.get("confidence_score"),
                    "filename": match.get("original_filename")
                }
                for match in duplicate_matches[:3]  # Limit to top 3 matches
            ] if duplicate_matches else None,
            duplicate_action=duplicate_action,
            duplicate_reasoning=duplicate_reasoning
        )      
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resume upload failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Resume upload failed")


@router.get("/list", response_model=ResumeListResponse)
async def list_resumes(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(10, ge=1, le=1000, description="Items per page"),
    status: Optional[str] = Query(None, description="Filter by status"),
    search: Optional[str] = Query(None, description="Search by candidate name or email"),
    sort_by: Optional[str] = Query(None, description="Sort field: created_at, candidate_name, status, quality_score, processing_completed_at, years_experience, candidate_email, upload_date, experience"),
    sort_order: Optional[str] = Query("desc", description="Sort order: asc or desc"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List resumes with filtering and pagination.
    """
    try:
        logger.info(f"🔍 Listing resumes for user {current_user.id} (page {page})")
        
        # Build base query using async SQLAlchemy syntax
        query = select(Resume).where(Resume.organization_id == current_user.organization_id)
        
        # Apply filters
        if status:
            query = query.where(Resume.status == status)
        
        if search:
            search_filter = f"%{search}%"
            # Normalize search term for phone number comparison
            normalized_search = normalize_phone_number(search)
            
            # Build search conditions
            search_conditions = [
                Resume.candidate_name.ilike(search_filter),
                Resume.candidate_email.ilike(search_filter),
                Resume.original_filename.ilike(search_filter),
                Resume.ai_analysis['personal_info']['professional_title'].astext.ilike(search_filter),
                Resume.candidate_id.ilike(search_filter)
            ]
            
            # Add phone number search with normalization
            if normalized_search:
                # Search in both original phone format and normalized format
                search_conditions.extend([
                    Resume.candidate_phone.ilike(search_filter),  # Original format search
                    func.regexp_replace(Resume.candidate_phone, r'\D', '', 'g').ilike(f"%{normalized_search}%")  # Normalized search
                ])
            else:
                # If search term has no digits, just do regular phone search
                search_conditions.append(Resume.candidate_phone.ilike(search_filter))
            
            query = query.where(or_(*search_conditions))
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0
        
        # Apply sorting
        valid_sort_fields = {
            "created_at": Resume.created_at,
            "candidate_name": Resume.candidate_name,
            "status": Resume.status,
            "processing_completed_at": Resume.processing_completed_at,
            "years_experience": Resume.years_of_experience,
            "candidate_email": Resume.candidate_email,
            "upload_date": Resume.created_at,  # Alias for created_at
            "experience": Resume.years_of_experience  # Alias for years_experience
        }
        
        # Apply sorting only if sort_by is provided
        if sort_by:
            # Handle quality_score sorting specially since it's an enum
            if sort_by == "quality_score":
                # Sort by quality_score enum with custom ordering
                quality_order = {
                    QualityScore.EXCELLENT: 4,
                    QualityScore.GOOD: 3,
                    QualityScore.AVERAGE: 2,
                    QualityScore.POOR: 1
                }
                
                # Use CASE statement to convert enum to numeric for sorting
                quality_sort_field = case(
                    (Resume.quality_score == QualityScore.EXCELLENT, 4),
                    (Resume.quality_score == QualityScore.GOOD, 3),
                    (Resume.quality_score == QualityScore.AVERAGE, 2),
                    (Resume.quality_score == QualityScore.POOR, 1),
                    else_=0
                )
                
                if sort_order.lower() == "asc":
                    query = query.order_by(quality_sort_field.asc())
                else:
                    query = query.order_by(quality_sort_field.desc())
            else:
                sort_field = valid_sort_fields.get(sort_by, Resume.created_at)
                
                if sort_order.lower() == "asc":
                    query = query.order_by(sort_field.asc())
                else:
                    query = query.order_by(sort_field.desc())
        else:
            # Default sorting by created_at desc when no sort_by is provided
            query = query.order_by(Resume.created_at.desc())
        
        # Apply pagination
        offset = (page - 1) * page_size
        paginated_query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(paginated_query)
        resumes = result.scalars().all()
        
        # Format response
        resume_list = []
        for resume in resumes:
            resume_data = {
                "id": str(resume.id),
                "candidate_id": resume.candidate_id,
                "filename": resume.original_filename,
                "candidate_name": resume.candidate_name,
                "candidate_email": resume.candidate_email,
                "candidate_phone": resume.candidate_phone,
                "status": resume.status.value,
                "blacklist": resume.blacklist or False,
                "quality_score": resume.quality_score.value if resume.quality_score else (resume.ai_analysis.get("quality_assessment", {}).get("overall_score") if resume.ai_analysis else None),
                "years_experience": float(resume.years_of_experience) if resume.years_of_experience is not None else 0.0,
                "total_years_experience": resume.ai_analysis.get("career_analysis", {}).get("total_years_experience", ""),  # Use years_of_experience for both
                "location": resume.candidate_location,  # Use candidate_location
                "upload_date": resume.created_at.isoformat(),
                "processing_completed": resume.processing_completed_at.isoformat() if resume.processing_completed_at else None,
                "professional_title": resume.ai_analysis.get("personal_info", {}).get("professional_title") if resume.ai_analysis else None,
            }
            resume_list.append(resume_data)
        
        has_next = (page * page_size) < total
        
        logger.info(f"✅ Listed {len(resume_list)} resumes (page {page})")
        
        return ResumeListResponse(
            resumes=resume_list,
            total=total,
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to list resumes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve resumes")


@router.post("/search/semantic", response_model=SemanticSearchResponse)
async def semantic_search_resumes(
    search_request: SemanticSearchRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Advanced NLP-powered semantic search for resumes.
    
    This endpoint:
    1. Analyzes natural language queries using AI
    2. Extracts search intent and entities
    3. Performs semantic similarity matching
    4. Ranks results using AI reasoning
    5. Provides explanations and suggestions
    
    Examples:
    - "Find React developers with 5+ years experience"
    - "Senior engineers with leadership background in fintech"
    - "Data scientists with machine learning and Python skills"
    - "Frontend developers who know TypeScript and have startup experience"
    """
    try:
        start_time = datetime.now(timezone.utc)
        settings = get_settings()
        
        logger.info(f"🔍 Starting semantic search: '{search_request.query}' for user {current_user.id}")
        
        # Get orchestrator and initialize search agent
        orchestrator: RecruitmentOrchestrator = request.app.state.orchestrator
        
        # Initialize semantic search agent (reuse existing embeddings if available)
        reuse_embeddings = None
        if hasattr(orchestrator, "vector_store") and orchestrator.vector_store:
            try:
                reuse_embeddings = getattr(orchestrator.vector_store, "embeddings", None)
            except Exception:
                reuse_embeddings = None

        search_agent = SemanticSearchAgent(
            llm=orchestrator.llm if hasattr(orchestrator, 'llm') else None,
            embeddings=reuse_embeddings,
            settings=settings
        )
        await search_agent.initialize()
        
        resumes = []
        # Use vector similarity via Qdrant for accurate semantic search
        if hasattr(orchestrator, "vector_store") and orchestrator.vector_store:
            try:
                logger.info(f"🔍 Using Qdrant vector search for query: '{search_request.query}'")
                # Optimize Qdrant search: use moderate top_k for UI/UX searches
                # UI/UX searches need more candidates due to skill variations
                optimized_top_k = min(15, search_request.max_results * 2)
                vector_hits = await orchestrator.vector_store.search_resumes(
                    query_text=search_request.query,
                    top_k=optimized_top_k,
                    org_id=str(current_user.organization_id)
                )
                hit_ids = [hit.id for hit in vector_hits]
                logger.info(f"📊 Qdrant returned {len(hit_ids)} resume IDs")
                
                if hit_ids:
                    # Fetch these resumes preserving order
                    result = await db.execute(
                        select(Resume).where(
                            Resume.organization_id == current_user.organization_id,
                            Resume.id.in_(hit_ids),
                            Resume.status == ResumeStatus.ANALYZED,
                            Resume.ai_analysis.isnot(None)
                        )
                    )
                    fetched = result.scalars().all()
                    logger.info(f"📊 PostgreSQL returned {len(fetched)} valid resumes from Qdrant IDs")
                    
                    id_to_resume = {str(r.id): r for r in fetched}
                    resumes = [id_to_resume[i] for i in hit_ids if i in id_to_resume]
                    logger.info(f"✅ Using {len(resumes)} Qdrant-ordered resumes for semantic search")
                else:
                    logger.warning("⚠️ Qdrant returned no results, falling back to DB scan")
            except Exception as e:
                logger.error(f"❌ Vector search failed, falling back to DB scan: {e}")
                logger.error(traceback.format_exc())
        else:
            logger.warning("⚠️ No vector store available, using DB scan")
        
        logger.info(f"🔍 Found {len(resumes)} resumes for semantic search")
        
        if not resumes:
            return SemanticSearchResponse(
                query=search_request.query,
                search_intent={
                    "intent_type": "no_data",
                    "confidence_score": 0.0,
                    "search_strategy": "no_resumes_available"
                },
                results=[],
                total_matches=0,
                search_time_ms=0,
                suggestions=["Upload and analyze some resumes first"],
                alternative_queries=[]
            )
        
        # Convert resumes to searchable format
        resume_data = []
        for resume in resumes:
            resume_dict = {
                "id": resume.id,
                "candidate_name": resume.candidate_name,
                "candidate_email": resume.candidate_email,
                "current_position": resume.current_position,
                "current_company": resume.current_company,
                "years_of_experience": resume.years_of_experience,
                "professional_summary": resume.professional_summary,
                "technical_skills": resume.technical_skills,
                "soft_skills": resume.soft_skills,
                "ai_analysis": resume.ai_analysis,
                "quality_score": resume.ai_analysis.get("quality_assessment", {}).get("overall_score") if resume.ai_analysis else None,
                "upload_date": resume.created_at.isoformat(),
                "processing_completed": resume.processing_completed_at.isoformat() if resume.processing_completed_at else None,
                "professional_title": resume.ai_analysis.get("personal_info", {}).get("professional_title") if resume.ai_analysis else None,
            }
            resume_data.append(resume_dict)
        
        # Perform semantic search
        search_results = await search_agent.semantic_search(
            query=search_request.query,
            resume_data=resume_data,
            max_results=search_request.max_results
        )
        
        # Format results for API response
        formatted_results = []
        for result in search_results.results:
            formatted_result = {
                "id": result.resume_id,
                "candidate_name": result.candidate_name,
                "relevance_score": result.relevance_score,
                "skill_match_score": result.skill_match_score,
                "experience_match_score": result.experience_match_score,
                "semantic_similarity": result.semantic_similarity,
                "matched_skills": result.matched_skills,
                "matched_experience": result.matched_experience,
            }
            
            # Add full resume data if requested
            if search_request.include_analysis:
                original_resume = next((r for r in resume_data if str(r["id"]) == result.resume_id), None)
                if original_resume:
                    formatted_result.update({
                        "candidate_email": original_resume["candidate_email"],
                        "current_position": original_resume["current_position"],
                        "current_company": original_resume["current_company"],
                        "years_of_experience": original_resume["years_of_experience"],
                        "quality_score": original_resume["quality_score"],
                        "upload_date": original_resume["upload_date"],
                        "processing_completed": original_resume["processing_completed"]
                    })
            
            formatted_results.append(formatted_result)
        
        search_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        
        logger.info(f"✅ Semantic search completed: {len(formatted_results)} results in {search_time}ms")
        
        return SemanticSearchResponse(
            query=search_results.query,
            search_intent=search_results.search_intent.dict(),
            results=formatted_results,
            total_matches=search_results.total_matches,
            search_time_ms=search_time,
            suggestions=search_results.suggestions,
            alternative_queries=search_results.alternative_queries
        )
        
    except Exception as e:
        logger.error(f"❌ Semantic search failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Semantic search failed: {str(e)}")


@router.get("/{resume_id}/analysis", response_model=ResumeAnalysisResponse)
async def get_resume_analysis(
    resume_id: UUID,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed analysis results for a specific resume.
    """
    try:
        # Get resume using async SQLAlchemy syntax
        query = select(Resume).where(
            Resume.id == resume_id,
            Resume.organization_id == current_user.organization_id
        )
        result = await db.execute(query)
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Build response
        response = ResumeAnalysisResponse(
            id=str(resume.id),
            candidate_id=resume.candidate_id,
            status=resume.status.value,
            blacklist=resume.blacklist or False,
            agent_personal_info_markdown=resume.agent_personal_info_markdown,
            agent_experience_markdown=resume.agent_experience_markdown,
            agent_skills_markdown=resume.agent_skills_markdown,
            agent_education_markdown=resume.agent_education_markdown,
            agent_quality_markdown=resume.agent_quality_markdown,
            agent_reasoning_markdown=resume.agent_reasoning_markdown,
            agent_raw_outputs=resume.agent_raw_outputs,
            analysis=resume.ai_analysis,
            extracted_text=resume.extracted_text,
            quality_score=resume.ai_analysis.get("quality_assessment", {}).get("overall_score") if resume.ai_analysis else None,
            confidence_score=resume.processing_confidence,
            candidate_name=resume.candidate_name if resume.candidate_name else None,
            resume_filename=resume.original_filename if resume.original_filename else None,
            processing_time=(
                (resume.processing_completed_at - resume.processing_started_at).total_seconds()
                if resume.processing_completed_at and resume.processing_started_at
                else None
            ),
            error_message=resume.processing_error
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get resume analysis: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to retrieve analysis")


@router.post("/{resume_id}/reprocess")
async def reprocess_resume(
    resume_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Reprocess a resume using the latest AI models.
    """
    try:
        # Get resume
        query = select(Resume).where(
            Resume.id == resume_id,
            Resume.organization_id == current_user.organization_id
        )
        result = await db.execute(query)
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Check Azure storage information
        if not resume.azure_blob_name or not resume.azure_blob_name.strip():
            logger.error(f"❌ Resume {resume_id} missing Azure blob name - cannot reprocess")
            raise HTTPException(
                status_code=400,
                detail="Resume missing Azure storage information. This may be an old upload that predates Azure integration."
            )

        # Set status to PROCESSING and commit before queueing
        resume.status = ResumeStatus.PROCESSING
        resume.processing_started_at = datetime.now(timezone.utc)
        resume.processing_error = None
        await db.commit()

        # Enqueue background reprocessing
        reprocess_resume_task.delay(
            resume_id=str(resume.id),
            blob_name=resume.azure_blob_name,
            container=resume.azure_container,
            filename=resume.original_filename,
            user_id=str(current_user.id),
            organization_id=str(current_user.organization_id),
            metadata={
                "reprocessed_by": current_user.username,
                "original_resume_id": str(resume.id),
            },
        )

        logger.info(f"✅ Resume reprocessing queued: {resume_id}")

        return JSONResponse(
            content={
                "message": "Resume reprocessing initiated successfully",
                "status": resume.status.value,
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resume reprocessing failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Resume reprocessing failed")


@router.delete("/{resume_id}")
async def delete_resume(
    resume_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Delete a resume and its associated files.
    """
    try:
        # Get resume
        query = select(Resume).where(
            Resume.id == resume_id,
            Resume.organization_id == current_user.organization_id
        )
        result = await db.execute(query)
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Delete related job matches first (due to foreign key constraints)
        job_matches_query = select(JobMatch).where(JobMatch.resume_id == resume_id)
        job_matches_result = await db.execute(job_matches_query)
        job_matches = job_matches_result.scalars().all()
        
        if job_matches:
            logger.info(f"🗑️ Deleting {len(job_matches)} related job matches for resume {resume_id}")
            for job_match in job_matches:
                await db.delete(job_match)
        
        # Delete from Azure Storage
        orchestrator: RecruitmentOrchestrator = request.app.state.orchestrator
        azure_storage = orchestrator.azure_storage
        
        try:
            await azure_storage.delete_file(
                resume.azure_container,
                resume.azure_blob_name
            )
        except Exception as e:
            logger.warning(f"Failed to delete file from Azure Storage: {str(e)}")
        
        # Delete resume from database
        await db.delete(resume)
        await db.commit()
        
        # Delete embedding from vector store
        try:
            orchestrator: RecruitmentOrchestrator = request.app.state.orchestrator
            if hasattr(orchestrator, "vector_store") and orchestrator.vector_store:
                await orchestrator.vector_store.delete_resume_embedding(str(resume_id))
        except Exception as e:
            logger.warning(f"Failed to delete resume embedding from vector store: {str(e)}")
        
        logger.info(f"✅ Resume deleted successfully: {resume_id}")
        
        return JSONResponse(
            content={"message": "Resume deleted successfully"}
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resume deletion failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Resume deletion failed")


@router.get("/{resume_id}/download")
async def download_resume(
    resume_id: UUID,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Download the original resume file directly.
    """
    try:
        logger.info(f"📄 Downloading original resume for ID: {resume_id}")
        
        # Get resume
        query = select(Resume).where(
            Resume.id == resume_id,
            Resume.organization_id == current_user.organization_id
        )
        result = await db.execute(query)
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Download file content directly from Azure Storage
        from core.azure_storage import AzureStorageService
        from core.config import get_settings
        
        settings = get_settings()
        azure_service = AzureStorageService(settings)
        await azure_service.initialize()
        
        # Download the file content
        file_content = await azure_service.download_file(
            container_name=resume.azure_container,
            blob_name=resume.azure_blob_name
        )
        
        # Check if this was originally a DOC file that was converted to DOCX
        # We can determine this by checking if the original filename ends with .docx
        # but the content is actually a DOCX file (which means it was converted)
        is_converted_doc = False
        if resume.original_filename.lower().endswith('.docx'):
            # Check if this might be a converted DOC file by looking at the content
            # DOCX files start with PK (ZIP signature)
            if file_content.startswith(b'PK'):
                # This is a DOCX file, but we need to check if it was converted from DOC
                # We can't easily determine this without additional metadata, so we'll assume
                # that if it's a DOCX file, it might have been converted and has Sage IT metadata
                is_converted_doc = True
                logger.info(f"📄 Detected potentially converted DOC file: {resume.original_filename}")
        
        # Determine content type based on file extension
        file_extension = resume.original_filename.lower().split('.')[-1]
        if file_extension == 'pdf':
            media_type = "application/pdf"
        elif file_extension in ['docx', 'doc']:
            media_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        elif file_extension == 'txt':
            media_type = "text/plain"
        else:
            media_type = "application/octet-stream"
        
        logger.info(f"✅ Original resume downloaded: {resume.original_filename}")
        
        # Return file content directly
        return Response(
            content=file_content,
            media_type=media_type,
            headers={
                "Content-Disposition": f"attachment; filename=\"{resume.original_filename}\"",
                "Content-Type": media_type
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resume download failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Resume download failed")


class ResumeComparisonRequest(BaseModel):
    """Request model for resume comparison."""
    resume_ids: List[str]
    comparison_criteria: Optional[Dict[str, float]] = None


class ComparisonResult(BaseModel):
    """Individual comparison result."""
    resume_id: str
    candidate_name: str
    overall_score: float
    category_scores: Dict[str, float]
    strengths: List[str]
    weaknesses: List[str]
    ranking_position: int


class ResumeComparisonResponse(BaseModel):
    """Response model for resume comparison."""
    comparison_id: str
    total_candidates: int
    comparison_criteria: Dict[str, float]
    results: List[ComparisonResult]
    winner_categories: Dict[str, str]
    skill_overlap_analysis: Dict[str, Any]
    ai_insights: Dict[str, Any]
    execution_time: float


@router.post("/compare", response_model=ResumeComparisonResponse)
async def compare_resumes(
    comparison_request: ResumeComparisonRequest,
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Compare multiple resumes using AI analysis.
    
    This endpoint:
    1. Validates selected resumes
    2. Runs AI-powered comparison analysis via LLM on stored analyses
    3. Provides detailed side-by-side comparison
    4. Ranks candidates across multiple criteria
    """
    try:
        start_time = datetime.now(timezone.utc)

        if len(comparison_request.resume_ids) < 2:
            raise HTTPException(status_code=400, detail="At least 2 resumes required for comparison")

        if len(comparison_request.resume_ids) > 4:
            raise HTTPException(status_code=400, detail="Maximum 4 resumes can be compared at once")

        logger.info(f"🔄 Starting AI resume comparison for {len(comparison_request.resume_ids)} candidates")

        # Fetch resumes and ensure they belong to the org and are analyzed
        result = await db.execute(
            select(Resume).where(
                Resume.organization_id == current_user.organization_id,
                Resume.id.in_(comparison_request.resume_ids),
                Resume.status == ResumeStatus.ANALYZED,
                Resume.ai_analysis.isnot(None)
            )
        )
        resumes = result.scalars().all()

        if len(resumes) != len(set(comparison_request.resume_ids)):
            raise HTTPException(status_code=404, detail="One or more resumes not found or not analyzed")

        # Build compact candidate summaries for the LLM
        candidates: List[Dict[str, Any]] = []
        for r in resumes:
            analysis = r.ai_analysis or {}
            personal = analysis.get("personal_info", {})
            career = analysis.get("career_analysis", {})
            quality = analysis.get("quality_assessment", {})
            skills = analysis.get("skills", analysis.get("technical_skills", {}))

            candidate_summary = {
                "resume_id": str(r.id),
                "candidate_name": personal.get("full_name") or r.candidate_name or r.original_filename,
                "years_experience": career.get("total_years_experience") or r.years_of_experience,
                "current_position": career.get("current_position"),
                "current_company": career.get("current_company"),
                "highest_education": analysis.get("education", {}).get("highest_degree"),
                "key_achievements": analysis.get("career_analysis", {}).get("notable_achievements", [])[:5],
                "technical_skills": skills.get("technical", skills) if isinstance(skills, dict) else skills,
                "soft_skills": analysis.get("soft_skills", []),
                "quality_overall_score": quality.get("overall_score"),
                "quality_dimensions": {
                    "completeness": quality.get("completeness_score"),
                    "relevance": quality.get("relevance_score"),
                    "presentation": quality.get("presentation_score"),
                    "impact": quality.get("impact_score"),
                },
            }
            candidates.append(candidate_summary)

        # Prepare comparison criteria (weights)
        weights = comparison_request.comparison_criteria or {
            "technical_skills": 0.3,
            "experience": 0.25,
            "education": 0.2,
            "leadership": 0.15,
            "communication": 0.1,
        }

        # Create LLM and ask for structured comparison
        settings = get_settings()
        llm_factory = LLMFactory(settings)
        llm = llm_factory.create_llm()

        system_prompt = (
            "You are an expert AI recruiter. Compare candidates strictly using the provided analyses. "
            "Return only valid JSON matching the requested schema, with numeric scores in [0,1]."
        )

        schema_hint = {
            "results": [
                {
                    "resume_id": "string",
                    "candidate_name": "string",
                    "overall_score": 0.0,
                    "category_scores": {
                        "technical_skills": 0.0,
                        "experience": 0.0,
                        "education": 0.0,
                        "leadership": 0.0,
                        "communication": 0.0
                    },
                    "strengths": ["string"],
                    "weaknesses": ["string"],
                    "ranking_position": 1
                }
            ],
            "winner_categories": {
                "technical_skills": "resume_id",
                "experience": "resume_id",
                "education": "resume_id",
                "leadership": "resume_id",
                "communication": "resume_id"
            },
            "skill_overlap_analysis": {
                "common_skills": ["string"],
                "unique_skills_by_candidate": {"candidate_name": ["string"]},
                "skill_gap_analysis": "string",
                "training_recommendations": ["string"]
            },
            "ai_insights": {
                "recommendation": "string",
                "key_differentiators": ["string"],
                "hiring_strategy": {
                    "immediate_hire": "string or null",
                    "development_potential": ["string"],
                    "additional_screening": ["string"]
                }
            }
        }

        user_prompt = (
            "Compare the following candidates. Use the weights to compute overall_score. "
            "Do not invent facts beyond the provided analyses."
            f"\n\nWeights: {json.dumps(weights)}"
            f"\n\nCandidates: {json.dumps(candidates) }"
            f"\n\nReturn ONLY JSON matching this schema (no extra text): {json.dumps(schema_hint)}"
        )

        response = await llm.ainvoke([
            ("system", system_prompt),
            ("user", user_prompt),
        ])

        try:
            content = response.content if hasattr(response, "content") else str(response)
            parsed = json.loads(content)
        except Exception as parse_error:
            logger.error(f"❌ Failed to parse LLM comparison JSON: {parse_error}\nRaw: {getattr(response, 'content', None)}")
            raise HTTPException(status_code=500, detail="AI comparison failed to produce valid JSON")

        # Validate and map to pydantic models
        results: List[ComparisonResult] = []
        for item in parsed.get("results", []):
            results.append(
                ComparisonResult(
                    resume_id=item.get("resume_id"),
                    candidate_name=item.get("candidate_name"),
                    overall_score=float(item.get("overall_score", 0.0)),
                    category_scores={
                        "technical_skills": float(item.get("category_scores", {}).get("technical_skills", 0.0)),
                        "experience": float(item.get("category_scores", {}).get("experience", 0.0)),
                        "education": float(item.get("category_scores", {}).get("education", 0.0)),
                        "leadership": float(item.get("category_scores", {}).get("leadership", 0.0)),
                        "communication": float(item.get("category_scores", {}).get("communication", 0.0)),
                    },
                    strengths=item.get("strengths", [])[:10],
                    weaknesses=item.get("weaknesses", [])[:10],
                    ranking_position=int(item.get("ranking_position", 0)),
                )
            )

        # Fallback: if LLM didn't include rankings, compute from overall_score
        if any(r.ranking_position in (0, None) for r in results):
            results_sorted = sorted(results, key=lambda r: r.overall_score, reverse=True)
            for idx, r in enumerate(results_sorted, start=1):
                r.ranking_position = idx
            results = results_sorted

        winner_categories = parsed.get("winner_categories", {})
        skill_overlap_analysis = parsed.get("skill_overlap_analysis", {})
        ai_insights = parsed.get("ai_insights", {})

        execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

        comparison_id = f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        logger.info(f"✅ Resume comparison completed: {comparison_id} ({execution_time:.2f}s)")

        return ResumeComparisonResponse(
            comparison_id=comparison_id,
            total_candidates=len(comparison_request.resume_ids),
            comparison_criteria=weights,
            results=results,
            winner_categories=winner_categories,
            skill_overlap_analysis=skill_overlap_analysis,
            ai_insights=ai_insights,
            execution_time=execution_time,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Resume comparison failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Resume comparison failed")


@router.post("/{resume_id}/check-processing")
async def check_processing_status(
    resume_id: str,
    current_user = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
    request: Request = Request
):
    """
    Check the current processing status of a resume and attempt to complete processing if stuck.
    """
    try:
        # Get resume record
        result = await db.execute(
            select(Resume).where(
                Resume.id == resume_id,
                Resume.organization_id == current_user.organization_id
            )
        )
        resume_record = result.scalar_one_or_none()
        
        if not resume_record:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        logger.info(f"🔍 Checking processing status for resume: {resume_id}")
        
        # Get orchestrator to check active sessions
        orchestrator: RecruitmentOrchestrator = request.app.state.orchestrator
        
        # Check if there are any active sessions for this resume
        active_sessions = getattr(orchestrator, 'active_sessions', {})
        
        session_info = None
        for session_id, session_data in active_sessions.items():
            if session_data.get("type") == "resume_processing":
                session_info = {
                    "session_id": session_id,
                    "status": session_data.get("status"),
                    "started_at": session_data.get("started_at"),
                    "type": session_data.get("type")
                }
                break
        
        processing_time = None
        if resume_record.processing_started_at:
            end_time = resume_record.processing_completed_at or datetime.now(timezone.utc)
            processing_time = (end_time - resume_record.processing_started_at).total_seconds()
        
        return {
            "resume_id": str(resume_record.id),
            "filename": resume_record.original_filename,
            "current_status": resume_record.status.value,
            "processing_started_at": resume_record.processing_started_at.isoformat() if resume_record.processing_started_at else None,
            "processing_completed_at": resume_record.processing_completed_at.isoformat() if resume_record.processing_completed_at else None,
            "processing_time_seconds": processing_time,
            "has_analysis": bool(resume_record.ai_analysis),
            "processing_confidence": resume_record.processing_confidence,
            "active_session": session_info,
            "recommendations": [
                "If stuck at PROCESSING for >60 seconds, try reprocessing",
                "Check logs for any errors during processing",
                "Verify Azure OpenAI configuration is working"
            ] if resume_record.status == ResumeStatus.PROCESSING else []
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to check processing status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to check processing status: {str(e)}")


# Duplicate Management Endpoints

class DuplicateReviewRequest(BaseModel):
    """Request model for reviewing duplicate resumes."""
    action: str = Field(..., description="Action to take: 'merge', 'keep_both', 'delete_duplicate'")
    primary_resume_id: Optional[str] = Field(None, description="ID of resume to keep when merging")
    notes: Optional[str] = Field(None, description="Review notes")


class DuplicateListResponse(BaseModel):
    """Response model for listing duplicate resumes."""
    duplicates: List[Dict[str, Any]]
    total: int
    page: int
    page_size: int
    has_next: bool


@router.get("/duplicates", response_model=DuplicateListResponse)
async def list_duplicate_resumes(
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(20, ge=1, le=100, description="Items per page"),
    confidence_threshold: float = Query(0.6, ge=0.0, le=1.0, description="Minimum confidence threshold"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    List resumes that have been flagged as potential duplicates.
    """
    try:
        logger.info(f"🔍 Listing duplicate resumes for user {current_user.id} (page {page})")
        
        # Build base query for duplicates
        query = select(Resume).where(
            and_(
                Resume.organization_id == current_user.organization_id,
                Resume.is_duplicate == True,
                Resume.duplicate_confidence_score >= confidence_threshold
            )
        ).order_by(Resume.duplicate_confidence_score.desc(), Resume.created_at.desc())
        
        # Get total count
        count_query = select(func.count()).select_from(query.subquery())
        count_result = await db.execute(count_query)
        total = count_result.scalar() or 0
        
        # Apply pagination
        offset = (page - 1) * page_size
        paginated_query = query.offset(offset).limit(page_size)
        
        # Execute query
        result = await db.execute(paginated_query)
        resumes = result.scalars().all()
        
        # Format response
        duplicates = []
        for resume in resumes:
            duplicate_data = {
                "id": str(resume.id),
                "candidate_name": resume.candidate_name,
                "candidate_email": resume.candidate_email,
                "candidate_phone": resume.candidate_phone,
                "original_filename": resume.original_filename,
                "is_duplicate": resume.is_duplicate,
                "duplicate_confidence_score": resume.duplicate_confidence_score,
                "duplicate_matches": resume.duplicate_matches or [],
                "duplicate_action_taken": resume.duplicate_action_taken,
                "normalized_email": resume.normalized_email,
                "normalized_phone": resume.normalized_phone,
                "created_at": resume.created_at.isoformat() if resume.created_at else None,
                "status": resume.status.value if resume.status else None,
                "blacklist": resume.blacklist or False
            }
            duplicates.append(duplicate_data)
        
        has_next = (page * page_size) < total
        
        return DuplicateListResponse(
            duplicates=duplicates,
            total=total,
            page=page,
            page_size=page_size,
            has_next=has_next
        )
        
    except Exception as e:
        logger.error(f"❌ Failed to list duplicate resumes: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to list duplicate resumes")


@router.post("/{resume_id}/review-duplicate")
async def review_duplicate_resume(
    resume_id: str,
    review_request: DuplicateReviewRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Review and take action on a duplicate resume.
    """
    try:
        logger.info(f"📋 Reviewing duplicate resume {resume_id} with action: {review_request.action}")
        
        # Get the resume
        result = await db.execute(
            select(Resume).where(
                Resume.id == resume_id,
                Resume.organization_id == current_user.organization_id
            )
        )
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Update review information
        resume.duplicate_action_taken = review_request.action
        resume.duplicate_reviewed_by = current_user.id
        resume.duplicate_reviewed_at = datetime.now(timezone.utc)
        
        response_message = ""
        
        if review_request.action == "merge":
            if not review_request.primary_resume_id:
                raise HTTPException(status_code=400, detail="primary_resume_id required for merge action")
            
            # Get the primary resume
            primary_result = await db.execute(
                select(Resume).where(
                    Resume.id == review_request.primary_resume_id,
                    Resume.organization_id == current_user.organization_id
                )
            )
            primary_resume = primary_result.scalar_one_or_none()
            
            if not primary_resume:
                raise HTTPException(status_code=404, detail="Primary resume not found")
            
            # Mark current resume as merged (but don't delete)
            resume.status = ResumeStatus.FAILED  # Use FAILED status to indicate merged/inactive
            resume.processing_error = f"Merged with resume {review_request.primary_resume_id}"
            response_message = f"Resume marked as merged with {review_request.primary_resume_id}"
            
        elif review_request.action == "keep_both":
            # Mark as reviewed but keep both resumes
            resume.is_duplicate = False  # Remove duplicate flag
            response_message = "Resume marked to keep both copies"
            
        elif review_request.action == "delete_duplicate":
            # Mark as deleted (soft delete)
            resume.status = ResumeStatus.FAILED
            resume.processing_error = "Marked as duplicate and deleted"
            response_message = "Resume marked as deleted duplicate"
            
        else:
            raise HTTPException(status_code=400, detail="Invalid action. Must be 'merge', 'keep_both', or 'delete_duplicate'")
        
        await db.commit()
        
        logger.info(f"✅ Duplicate review completed for {resume_id}: {review_request.action}")
        
        return {
            "message": response_message,
            "resume_id": resume_id,
            "action_taken": review_request.action,
            "reviewed_by": current_user.username,
            "reviewed_at": datetime.now(timezone.utc).isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to review duplicate resume: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to review duplicate resume")


@router.get("/{resume_id}/duplicate-details")
async def get_duplicate_details(
    resume_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Get detailed information about a resume's duplicate detection results.
    """
    try:
        # Get the resume
        result = await db.execute(
            select(Resume).where(
                Resume.id == resume_id,
                Resume.organization_id == current_user.organization_id
            )
        )
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Get detailed match information
        matches_with_details = []
        if resume.duplicate_matches:
            for match in resume.duplicate_matches:
                match_resume_id = match.get("resume_id")
                if match_resume_id:
                    match_result = await db.execute(
                        select(Resume).where(Resume.id == match_resume_id)
                    )
                    match_resume = match_result.scalar_one_or_none()
                    
                    if match_resume:
                        match_details = {
                            **match,
                            "current_status": match_resume.status.value if match_resume.status else None,
                            "upload_date": match_resume.created_at.isoformat() if match_resume.created_at else None,
                            "azure_url": match_resume.azure_url
                        }
                        matches_with_details.append(match_details)
        
        return {
            "resume_id": resume_id,
            "is_duplicate": resume.is_duplicate,
            "duplicate_confidence_score": resume.duplicate_confidence_score,
            "duplicate_detection_result": resume.duplicate_detection_result or {},
            "matches": matches_with_details,
            "normalized_contact": {
                "email": resume.normalized_email,
                "phone": resume.normalized_phone
            },
            "action_taken": resume.duplicate_action_taken,
            "reviewed_by": str(resume.duplicate_reviewed_by) if resume.duplicate_reviewed_by else None,
            "reviewed_at": resume.duplicate_reviewed_at.isoformat() if resume.duplicate_reviewed_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get duplicate details: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get duplicate details")


# Duplicate Search Endpoints

class DuplicateSearchRequest(BaseModel):
    """Request model for searching duplicate resumes."""
    name: Optional[str] = Field(None, description="Candidate name to search for")
    email: Optional[str] = Field(None, description="Email address to search for")
    phone: Optional[str] = Field(None, description="Phone number to search for")
    fuzzy_threshold: float = Field(0.8, ge=0.0, le=1.0, description="Fuzzy matching threshold (0.0-1.0)")
    max_results: int = Field(10, ge=1, le=50, description="Maximum number of results to return")


class DuplicateSearchResult(BaseModel):
    """Individual duplicate search result."""
    resume_id: str
    candidate_name: Optional[str]
    candidate_email: Optional[str]
    candidate_phone: Optional[str]
    original_filename: str
    created_at: datetime
    status: str
    match_score: float
    match_type: str  # "name", "email", "phone", "multiple"
    confidence: float
    blacklist: bool = False  # Whether this resume is blacklisted


class DuplicateSearchResponse(BaseModel):
    """Response model for duplicate search."""
    results: List[DuplicateSearchResult]
    total_found: int
    search_criteria: Dict[str, Any]
    search_performed_at: datetime


@router.post("/search-duplicates", response_model=DuplicateSearchResponse)
async def search_duplicate_resumes(
    search_request: DuplicateSearchRequest = None,
    file: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Search for duplicate resumes using vector similarity search in Qdrant.
    
    This endpoint supports two modes:
    1. Manual search: Provide search_request with name, email, phone
    2. File upload: Provide file to extract information and search for duplicates
    
    This endpoint performs intelligent duplicate detection using:
    1. Vector embeddings for semantic similarity matching
    2. AI-powered contact information normalization
    3. Hybrid search combining vector similarity with exact matching
    4. Configurable similarity thresholds
    
    Args:
        search_request: Search criteria including name, email, phone (optional)
        file: Resume file to extract information from (optional)
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        List of potential duplicate resumes with match scores and confidence levels
    """
    try:
        # Determine search mode and extract search criteria
        if file and file.filename:
            # File upload mode - extract information from file
            logger.info(f"🔍 Starting file-based duplicate search for file: {file.filename}")
            
            # Read file content
            file_content = await file.read()
            
            # Extract text content including headers
            from utils.parsers.document_parser import DocumentParser
            parser = DocumentParser()
            
            # Parse the document to extract text
            parse_result = await parser.parse_document(
                file_content=file_content,
                filename=file.filename
            )
            
            extracted_text = parse_result["text"]
            logger.info(f"📄 Extracted text length: {len(extracted_text)} characters")
            
            # Extract candidate information using AI
            from agents.orchestrator import RecruitmentOrchestrator
            orchestrator = RecruitmentOrchestrator()
            await orchestrator.initialize()
            
            # Use AI to extract personal information
            # Use the orchestrator's resume agent to extract personal info
            personal_info = await orchestrator.resume_agent._extract_personal_info(extracted_text)
            
            # Create search request from extracted information
            search_request = DuplicateSearchRequest(
                name=personal_info.full_name,
                email=personal_info.email,
                phone=personal_info.phone,
                max_results=10,
                fuzzy_threshold=0.2
            )
            
            logger.info(f"👤 Extracted info - Name: {search_request.name}, Email: {search_request.email}, Phone: {search_request.phone}")
            
        elif search_request:
            # Manual search mode
            logger.info(f"🔍 Starting manual duplicate search for user {current_user.id}")
            logger.info(f"Search criteria: name='{search_request.name}', email='{search_request.email}', phone='{search_request.phone}'")
        else:
            raise HTTPException(
                status_code=400, 
                detail="Either search_request or file must be provided"
            )
        
        # Validate that at least one search criteria is provided
        if not any([search_request.name, search_request.email, search_request.phone]):
            raise HTTPException(
                status_code=400, 
                detail="At least one search criteria (name, email, or phone) must be provided"
            )
        # Initialize orchestrator to get vector store (if not already done in file mode)
        if not file or not file.filename:
            from agents.orchestrator import RecruitmentOrchestrator
            orchestrator = RecruitmentOrchestrator()
            await orchestrator.initialize()
        if not hasattr(orchestrator, "vector_store") or not orchestrator.vector_store:
            raise HTTPException(
                status_code=503,
                detail="Vector store not available. Please ensure Qdrant is properly configured."
            )
        # Initialize duplicate detection service for normalization
        duplicate_service = duplicate_detection_service
        duplicate_service.initialize()
        # Normalize contact information using AI
        normalized_contact = await duplicate_service.normalize_contact_info(
            search_request.email, 
            search_request.phone
        )
        logger.info(f"📧 Normalized contact: email='{normalized_contact.email}', phone='{normalized_contact.phone}'")
        # Simple and effective: Let embeddings handle similarity
        search_components = []
        if search_request.name:
            search_components.append(f"Name: {search_request.name}")
        if normalized_contact.email:
            search_components.append(f"Email: {normalized_contact.email}")
        if normalized_contact.phone:
            search_components.append(f"Phone: {normalized_contact.phone}")
        search_query = " | ".join(search_components)
        logger.info(f"🔍 Vector search query: {search_query}")
        # Let embeddings handle the similarity - they're designed for this!
        # Remove org filter to check if there are any resumes at all
        vector_results = await orchestrator.vector_store.search_resumes(
            query_text=search_query,
            top_k=search_request.max_results * 3,  # Get more results for better coverage
            org_id=None  # No organization filter - search all resumes
        )
        logger.info(f"🎯 Vector search returned {len(vector_results)} candidates")
        logger.info(f"🔍 Organization ID: {current_user.organization_id}")
        logger.info(f"🔍 Search query: '{search_query}'")
        # Debug: Log first few results for troubleshooting
        for i, result in enumerate(vector_results[:3]):
            logger.info(f"🔍 Vector result {i+1}: id={result.id}, score={result.score:.3f}, payload={result.payload}")
        
        # Filter and process results
        results = []
        seen_resume_ids = set()
        
        for vector_result in vector_results:
            resume_id = vector_result.id
            similarity_score = vector_result.score
            payload = vector_result.payload
            # Skip if we've already processed this resume
            if resume_id in seen_resume_ids:
                continue
            seen_resume_ids.add(resume_id)
            # Get full resume data from database (no organization filter for now)
            resume_query = select(Resume).where(Resume.id == resume_id)
            resume_result = await db.execute(resume_query)
            resume = resume_result.scalar_one_or_none()
            if not resume:
                logger.info(f"🔍 Skipping resume {resume_id} - not found in DB")
                continue
            # Trust the embedding similarity score - that's what it's designed for!
            match_confidence = similarity_score
            
            # Only include results above minimum threshold
            if match_confidence >= (0.35):  # Lower threshold for vector search
                logger.info(f"✅ Including candidate: {resume.candidate_name}")
                results.append(DuplicateSearchResult(
                    resume_id=str(resume.id),
                    candidate_name=resume.candidate_name,
                    candidate_email=resume.candidate_email,
                    candidate_phone=resume.candidate_phone,
                    original_filename=resume.original_filename,
                    created_at=resume.created_at,
                    status=resume.status.value,
                    match_score=match_confidence,
                    match_type="vector",  # Simple default
                    confidence=similarity_score,
                    blacklist=resume.blacklist or False
                ))
            else:
                logger.info(f"❌ Excluding candidate: {resume.candidate_name} (below threshold)")
        
        # Sort by match score (highest first)
        results.sort(key=lambda x: x.match_score, reverse=True)
        
        # Limit results
        results = results[:search_request.max_results]
        
        logger.info(f"✅ Vector-based duplicate search completed: found {len(results)} results")
        
        response = DuplicateSearchResponse(
            results=results,
            total_found=len(results),
            search_criteria={
                "name": search_request.name,
                "email": search_request.email,
                "phone": search_request.phone,
                "fuzzy_threshold": search_request.fuzzy_threshold,
                "normalized_email": normalized_contact.email,
                "normalized_phone": normalized_contact.phone,
                "search_query": search_query,
                "vector_search_enabled": True
            },
            search_performed_at=datetime.now(timezone.utc)
        )
        
        logger.info(f"🔍 Returning duplicate search response: total_found={response.total_found}, results_count={len(response.results)}")
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Vector-based duplicate search failed: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Duplicate search failed: {str(e)}")


@router.post("/blacklist-duplicates")
async def blacklist_duplicates(
    blacklist_request: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Blacklist multiple resumes to prevent them from showing in future searches and matches.
    
    Args:
        blacklist_request: Dictionary containing list of resume IDs to blacklist
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Success message with count of blacklisted resumes
    """
    try:
        # Extract resume IDs from request
        resume_ids = blacklist_request.get("resume_ids", [])
        
        if not resume_ids:
            return {"message": "No resume IDs provided", "blacklisted_count": 0}
        
        logger.info(f"🚫 Blacklisting {len(resume_ids)} resumes")
        
        blacklisted_count = 0
        already_blacklisted = []
        not_found = []
        blacklisted_resumes = []
        
        for resume_id in resume_ids:
            try:
                # Get the resume
                resume = await db.get(Resume, resume_id)
                if not resume:
                    not_found.append(resume_id)
                    continue
                
                # Check if resume is already blacklisted
                if resume.blacklist:
                    already_blacklisted.append(resume_id)
                    continue
                
                # Set blacklist field to True
                resume.blacklist = True
                blacklisted_count += 1
                blacklisted_resumes.append(resume_id)
                
            except Exception as e:
                logger.error(f"❌ Failed to blacklist resume {resume_id}: {str(e)}")
                continue
        
        # Commit all changes
        await db.commit()
        
        logger.info(f"✅ Successfully blacklisted {blacklisted_count} resumes")
        
        return {
            "message": f"Successfully blacklisted {blacklisted_count} resumes",
            "blacklisted_count": blacklisted_count,
            "blacklisted_resume_ids": blacklisted_resumes,
            "already_blacklisted": already_blacklisted,
            "not_found": not_found,
            "blacklisted_at": datetime.now(timezone.utc).isoformat(),
            "blacklisted_by": str(current_user.id)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to blacklist resumes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to blacklist resumes: {str(e)}")


@router.post("/unblacklist-duplicates")
async def unblacklist_duplicates(
    unblacklist_request: Dict[str, Any],
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Unblacklist multiple resumes to allow them to show in future searches and matches.
    
    Args:
        unblacklist_request: Dictionary containing list of resume IDs to unblacklist
        db: Database session
        current_user: Current authenticated user
        
    Returns:
        Success message with count of unblacklisted resumes
    """
    try:
        # Extract resume IDs from request
        resume_ids = unblacklist_request.get("resume_ids", [])
        
        if not resume_ids:
            return {"message": "No resume IDs provided", "unblacklisted_count": 0}
        
        logger.info(f"✅ Unblacklisting {len(resume_ids)} resumes")
        
        unblacklisted_count = 0
        not_blacklisted = []
        not_found = []
        unblacklisted_resumes = []
        
        for resume_id in resume_ids:
            try:
                # Get the resume
                resume = await db.get(Resume, resume_id)
                if not resume:
                    not_found.append(resume_id)
                    continue
                
                # Check if resume is actually blacklisted
                if not resume.blacklist:
                    not_blacklisted.append(resume_id)
                    continue
                
                # Set blacklist field to False
                resume.blacklist = False
                unblacklisted_count += 1
                unblacklisted_resumes.append(resume_id)
                
            except Exception as e:
                logger.error(f"❌ Failed to unblacklist resume {resume_id}: {str(e)}")
                continue
        
        # Commit all changes
        await db.commit()
        
        logger.info(f"✅ Successfully unblacklisted {unblacklisted_count} resumes")
        
        return {
            "message": f"Successfully unblacklisted {unblacklisted_count} resumes",
            "unblacklisted_count": unblacklisted_count,
            "unblacklisted_resume_ids": unblacklisted_resumes,
            "not_blacklisted": not_blacklisted,
            "not_found": not_found,
            "unblacklisted_at": datetime.now(timezone.utc).isoformat(),
            "unblacklisted_by": str(current_user.id)
        }
        
    except Exception as e:
        logger.error(f"❌ Failed to unblacklist resumes: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to unblacklist resumes: {str(e)}")




@router.get("/{resume_id}/similar", response_model=Dict[str, Any])
async def get_similar_profiles(
    resume_id: UUID,
    request: Request,
    max_results: int = Query(default=10, ge=1, le=20, description="Maximum number of similar profiles to return"),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Find profiles similar to the specified resume based on skills, experience, and companies.
    
    This endpoint:
    1. Analyzes the target resume's characteristics
    2. Compares against all other resumes in the organization
    3. Uses AI-powered similarity matching across multiple dimensions
    4. Returns ranked similar profiles with explanations
    
    Similarity factors:
    - Technical and soft skills overlap
    - Career experience and progression
    - Shared companies or similar company types  
    - Similar job titles and responsibilities
    - Educational background alignment
    - Industry experience similarity
    - Career seniority level matching
    """
    try:
        start_time = datetime.now(timezone.utc)
        settings = get_settings()
        
        logger.info(f"🔍 Finding similar profiles for resume {resume_id} for user {current_user.id}")
        
        # Get the target resume
        result = await db.execute(
            select(Resume).where(
                Resume.id == resume_id,
                Resume.organization_id == current_user.organization_id,
                Resume.status == ResumeStatus.ANALYZED,
                Resume.ai_analysis.isnot(None)
            )
        )
        target_resume = result.scalar_one_or_none()
        
        if not target_resume:
            raise HTTPException(
                status_code=404, 
                detail="Resume not found or not analyzed yet"
            )
        
        # Get all other analyzed resumes in the organization for comparison
        result = await db.execute(
            select(Resume).where(
                Resume.organization_id == current_user.organization_id,
                Resume.status == ResumeStatus.ANALYZED,
                Resume.ai_analysis.isnot(None),
                Resume.id != resume_id  # Exclude the target resume
            )
        )
        candidate_resumes = result.scalars().all()
        
        logger.info(f"🔍 Comparing against {len(candidate_resumes)} candidate resumes")
        
        if len(candidate_resumes) == 0:
            return {
                "target_resume_id": str(resume_id),
                "target_candidate_name": target_resume.ai_analysis.get("personal_info", {}).get("full_name", "Unknown"),
                "similar_profiles": [],
                "total_candidates_analyzed": 0,
                "search_time_ms": 0,
                "message": "No other analyzed resumes found for comparison"
            }
        
        # Get orchestrator and initialize similarity agent
        orchestrator: RecruitmentOrchestrator = request.app.state.orchestrator
        
        # Initialize similar profiles agent
        similarity_agent = SimilarProfilesAgent(
            llm=orchestrator.llm if hasattr(orchestrator, 'llm') else None,
            embeddings=getattr(orchestrator.vector_store, "embeddings", None) if hasattr(orchestrator, "vector_store") else None,
            settings=settings
        )
        await similarity_agent.initialize()
        
        # Convert resume objects to dictionaries for the agent - include all relevant fields
        def resume_to_dict(resume):
            return {
                "id": str(resume.id),
                "candidate_name": resume.candidate_name or resume.ai_analysis.get("personal_info", {}).get("full_name", "Unknown"),
                "ai_analysis": resume.ai_analysis,
                # Direct database fields (prioritized by similarity agent)
                "technical_skills": resume.technical_skills or [],
                "soft_skills": resume.soft_skills or [],
                "years_of_experience": resume.years_of_experience,
                "current_position": resume.current_position,
                "current_company": resume.current_company,
                "work_experience": resume.work_experience or [],
                "education": resume.education or [],
                "highest_degree": resume.highest_degree,
                # Fallback values from ai_analysis if direct fields are empty
                "professional_summary": resume.professional_summary or resume.ai_analysis.get("personal_info", {}).get("professional_title", "")
            }
        
        target_resume_dict = resume_to_dict(target_resume)
        
        candidate_dicts = []
        for resume in candidate_resumes:
            candidate_dicts.append(resume_to_dict(resume))
        
        # Find similar profiles
        similarity_response = await similarity_agent.find_similar_profiles(
            target_resume=target_resume_dict,
            candidate_pool=candidate_dicts,
            max_results=max_results
        )
        
        search_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        
        logger.info(f"✅ Found {len(similarity_response.similar_profiles)} similar profiles in {search_time}ms")
        
        # Convert response to dictionary format
        return {
            "target_resume_id": similarity_response.target_resume_id,
            "target_candidate_name": similarity_response.target_candidate_name,
            "similar_profiles": [
                {
                    "resume_id": profile.resume_id,
                    "candidate_name": profile.candidate_name,
                    "current_position": profile.current_position,
                    "current_company": profile.current_company,
                    "years_experience": profile.years_experience,
                    "overall_similarity": profile.overall_similarity,
                    "similarity_factors": {
                        "skills_similarity": profile.similarity_factors.skills_similarity,
                        "experience_similarity": profile.similarity_factors.experience_similarity,
                        "company_overlap": profile.similarity_factors.company_overlap,
                        "role_similarity": profile.similarity_factors.role_similarity,
                        "education_similarity": profile.similarity_factors.education_similarity,
                        "industry_similarity": profile.similarity_factors.industry_similarity,
                        "seniority_similarity": profile.similarity_factors.seniority_similarity
                    },
                    "shared_skills": profile.shared_skills,
                    "shared_companies": profile.shared_companies,
                    "similar_roles": profile.similar_roles
                }
                for profile in similarity_response.similar_profiles
            ],
            "total_candidates_analyzed": similarity_response.total_candidates_analyzed,
            "search_time_ms": search_time,
            "similarity_criteria": similarity_response.similarity_criteria
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to find similar profiles: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to find similar profiles")


# generate_clean_resume function:
@router.get("/{resume_id}/generate-clean-resume")
async def generate_clean_resume(
    resume_id: str,
    request: Request,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Generate standardized resume using LLM to reformat extracted text.
    Converts any resume format into your specific standardized template.
    On first call, generates and stores the formatted resume.
    On subsequent calls, serves the cached formatted resume.
    """
    try:
        logger.info(f"📄 Processing formatted resume request for ID: {resume_id}")
        
        # Get resume from database
        result = await db.execute(
            select(Resume).where(
                Resume.id == resume_id,
                Resume.organization_id == current_user.organization_id
            )
        )
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Check if formatted resume already exists
        if resume.formatted_resume_blob_name:
            logger.info(f"📄 Found cached formatted resume for {resume.candidate_name}")
            
            try:
                # Download from Azure Storage
                from core.azure_storage import AzureStorageService
                from core.config import get_settings
                
                settings = get_settings()
                azure_service = AzureStorageService(settings)
                await azure_service.initialize()
                
                # Download the file
                file_content = await azure_service.download_file(
                    container_name="formatted-resumes",
                    blob_name=resume.formatted_resume_blob_name
                )
                
                # Generate filename
                candidate_name = resume.candidate_name or "Professional"
                timestamp = resume.formatted_resume_generated_at.strftime("%Y%m%d_%H%M%S") if resume.formatted_resume_generated_at else "cached"
                filename = f"{candidate_name}_Formatted_Resume_{timestamp}.docx"
                
                logger.info(f"✅ Served cached formatted resume: {filename}")
                
                return Response(
                    content=file_content,
                    media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    headers={
                        "Content-Disposition": f"attachment; filename={filename}",
                        "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                        "X-Cache-Status": "HIT"
                    }
                )
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to download cached formatted resume: {str(e)}")
                # Fall through to regenerate
                pass
        
        # Validate extracted text exists
        if not resume.extracted_text:
            raise HTTPException(
                status_code=400,
                detail="Resume text not available. Please ensure resume has been processed."
            )
        
        if len(resume.extracted_text.strip()) < 50:
            raise HTTPException(
                status_code=400,
                detail="Resume text too short to format."
            )
        
        # Generate new formatted resume
        logger.info(f"🤖 Generating new formatted resume for {resume.candidate_name}")
        
        # Initialize LLM formatter
        formatter = LLMResumeFormatter()
        
        # Get candidate name
        candidate_name = resume.candidate_name or "Professional"
        
        # Generate filename
        filename = formatter.generate_filename(candidate_name)
        
        # Create output directory
        output_dir = "generated_resumes"
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, filename)
        
        # Metadata
        metadata = {
            'quality_score': resume.quality_score or 0,
            'processed_date': resume.processing_completed_at
        }
        
        # Format using LLM (makes AI call to restructure)
        logger.info("🤖 Calling LLM to reformat resume into standardized template...")
        generated_path = await formatter.format_resume(
            extracted_text=resume.extracted_text,
            candidate_name=candidate_name,
            output_path=output_path,
            metadata=metadata
        )
        logger.info("✅ Resume formatted successfully")
        
        # Read generated file
        with open(generated_path, 'rb') as file:
            file_content = file.read()
        
        # Store in Azure Storage and database
        try:
            from core.azure_storage import AzureStorageService
            from core.config import get_settings
            from datetime import datetime, timezone
            
            settings = get_settings()
            azure_service = AzureStorageService(settings)
            await azure_service.initialize()
            
            # Generate unique blob name
            blob_name = f"formatted-resumes/{resume_id}/{filename}"
            
            # Upload to Azure Storage using the correct container
            container_name = "formatted-resumes"
            blob_client = azure_service.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Prepare metadata
            blob_metadata = {
                "original_filename": filename,
                "document_type": "formatted-resume",
                "resume_id": str(resume_id),
                "candidate_name": candidate_name,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "file_size": str(len(file_content)),
                "content_type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            }
            
            # Upload file
            await blob_client.upload_blob(
                data=file_content,
                overwrite=True,
                metadata=blob_metadata
            )
            
            logger.info(f"✅ Formatted resume uploaded to Azure Storage: {blob_name}")
            
            # Update database with formatted resume info
            resume.formatted_resume_blob_name = blob_name
            resume.formatted_resume_generated_at = datetime.now(timezone.utc)
            resume.formatted_resume_file_size = len(file_content)
            resume.formatted_resume_llm_provider = "openai"  # Get from formatter if available
            resume.formatted_resume_llm_model = "gpt-4"  # Get from formatter if available
            
            await db.commit()
            
            logger.info(f"✅ Formatted resume stored in Azure Storage: {blob_name}")
            
        except Exception as e:
            logger.error(f"⚠️ Failed to store formatted resume: {str(e)}")
            # Continue to serve the file even if storage fails
        
        # Clean up local file
        os.remove(generated_path)
        
        # Return file
        return Response(
            content=file_content,
            media_type="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            headers={
                "Content-Disposition": f"attachment; filename={filename}",
                "Content-Type": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                "X-Cache-Status": "MISS"
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to format resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to format resume: {str(e)}")


@router.get("/{resume_id}/generate-clean-resume/status")
async def get_resume_generation_status(
    resume_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get the status of resume generation.
    This endpoint can be polled by the frontend to check progress.
    """
    try:
        # For now, we'll return a simple status
        # In a production system, you might want to store generation status in the database
        result = await db.execute(
            select(Resume).where(
                Resume.id == resume_id,
                Resume.organization_id == current_user.organization_id
            )
        )
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        # Check if resume has been processed
        if resume.status == ResumeStatus.ANALYZED:
            return {
                "status": "ready",
                "message": "Resume is ready for generation",
                "resume_id": str(resume.id),
                "candidate_name": resume.candidate_name,
                "has_formatted_resume": bool(resume.formatted_resume_blob_name),
                "formatted_resume_generated_at": resume.formatted_resume_generated_at.isoformat() if resume.formatted_resume_generated_at else None,
                "formatted_resume_file_size": resume.formatted_resume_file_size,
                "formatted_resume_llm_provider": resume.formatted_resume_llm_provider,
                "formatted_resume_llm_model": resume.formatted_resume_llm_model
            }
        elif resume.status == ResumeStatus.PROCESSING:
            return {
                "status": "processing",
                "message": "Resume is still being processed",
                "resume_id": str(resume.id)
            }
        else:
            return {
                "status": "error",
                "message": f"Resume processing failed: {resume.processing_error or 'Unknown error'}",
                "resume_id": str(resume.id)
            }
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to get resume generation status: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to get resume generation status")


@router.delete("/{resume_id}/formatted-resume")
async def delete_formatted_resume(
    resume_id: str,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Delete the cached formatted resume to force regeneration.
    """
    try:
        result = await db.execute(
            select(Resume).where(
                Resume.id == resume_id,
                Resume.organization_id == current_user.organization_id
            )
        )
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        if not resume.formatted_resume_blob_name:
            raise HTTPException(status_code=404, detail="No formatted resume found")
        
        # Delete from Azure Storage
        try:
            from core.azure_storage import AzureStorageService
            from core.config import get_settings
            
            settings = get_settings()
            azure_service = AzureStorageService(settings)
            await azure_service.initialize()
            
            # Delete the blob
            blob_client = azure_service.blob_service_client.get_blob_client(
                container="formatted-resumes",
                blob=resume.formatted_resume_blob_name
            )
            await blob_client.delete_blob()
            
            logger.info(f"✅ Deleted formatted resume from Azure Storage: {resume.formatted_resume_blob_name}")
            
        except Exception as e:
            logger.warning(f"⚠️ Failed to delete formatted resume from Azure Storage: {str(e)}")
            # Continue to clear database fields
        
        # Clear database fields
        resume.formatted_resume_blob_name = None
        resume.formatted_resume_generated_at = None
        resume.formatted_resume_file_size = None
        resume.formatted_resume_llm_provider = None
        resume.formatted_resume_llm_model = None
        
        await db.commit()
        
        logger.info(f"✅ Cleared formatted resume data for resume {resume_id}")
        
        return {
            "message": "Formatted resume deleted successfully",
            "resume_id": str(resume_id)
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Failed to delete formatted resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete formatted resume: {str(e)}")

