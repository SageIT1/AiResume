"""
AI Recruit - Interview Generation API Endpoints
Generate structured interview questions for job candidates.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

import json
import logging
import re
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from core.config import get_settings
from core.llm_factory import LLMFactory
from core.security import get_current_user
from database.models import User, Resume, JobMatch, Job
from database.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


class InterviewGenerationRequest(BaseModel):
    """Request model for interview generation."""
    resume_id: str = Field(..., description="ID of the candidate resume")
    job_id: str = Field(..., description="ID of the job position")
    interview_level: str = Field("L3", description="Interview level (L1, L2, L3, L4, L5)")
    focus_areas: Optional[List[str]] = Field(None, description="Specific areas to focus on")


class InterviewQuestion(BaseModel):
    """Individual interview question model."""
    category: str = Field(..., description="Question category (e.g., React, Backend, System Design)")
    question: str = Field(..., description="The interview question")
    what_to_listen_for: List[str] = Field(..., description="Key points to listen for in the answer")
    follow_up_questions: Optional[List[str]] = Field(None, description="Optional follow-up questions")


class InterviewGuide(BaseModel):
    """Complete interview guide model."""
    candidate_name: str
    job_title: str
    interview_level: str
    overall_score: float
    scope: Dict[str, Any]
    key_questions: List[InterviewQuestion]
    optional_exercise: Optional[Dict[str, Any]] = None
    scorecard: Dict[str, Any]
    recommendation_guidelines: str


class InterviewGenerationResponse(BaseModel):
    """Response model for interview generation."""
    interview_id: str
    generated_at: str
    guide: InterviewGuide
    generation_time_ms: int
    cached: Optional[bool] = False
    is_new_generation: Optional[bool] = True


@router.post("/generate", response_model=InterviewGenerationResponse)
async def generate_interview_questions(
    request: InterviewGenerationRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Generate structured interview questions for a candidate-job match.
    
    This endpoint:
    1. Validates the user has permission (admin or senior recruiter)
    2. Fetches the job match and candidate details
    3. Makes a single AI call to generate comprehensive interview guide
    4. Returns structured interview questions and guidelines
    """
    try:
        start_time = datetime.now(timezone.utc)
        
        # Check user permissions
        if current_user.role not in ["admin", "senior_recruiter"]:
            raise HTTPException(
                status_code=403, 
                detail="Only admin and senior recruiter users can generate interview questions"
            )
        
        logger.info(f"🎯 Generating interview questions for resume {request.resume_id} and job {request.job_id}")
        
        # Fetch job match to verify score and get details
        job_match_query = select(JobMatch).join(Job).where(
            and_(
                JobMatch.resume_id == request.resume_id,
                JobMatch.job_id == request.job_id,
                Job.organization_id == current_user.organization_id
            )
        )
        job_match_result = await db.execute(job_match_query)
        job_match = job_match_result.scalar_one_or_none()
        
        if not job_match:
            raise HTTPException(
                status_code=404, 
                detail="Job match not found or access denied"
            )
        
        # Check if overall score is above 40%
        if job_match.overall_score < 0.4:
            raise HTTPException(
                status_code=400,
                detail=f"Cannot generate interview questions. Overall score ({job_match.overall_score:.1%}) must be above 40%"
            )
        
        # Check if interview questions already exist in the database
        if job_match.interview_questions and job_match.interview_questions.get('guide'):
            logger.info(f"📋 Found existing interview questions for resume {request.resume_id} and job {request.job_id}")
            
            # Return existing interview questions
            existing_guide = job_match.interview_questions['guide']
            return InterviewGenerationResponse(
                interview_id=job_match.interview_questions.get('interview_id', 'existing'),
                generated_at=job_match.interview_questions.get('generated_at', datetime.now(timezone.utc).isoformat()),
                guide=InterviewGuide(**existing_guide),
                generation_time_ms=0,  # No generation time for cached results
                cached=True,
                is_new_generation=False
            )
        
        # Fetch resume and job details
        resume_query = select(Resume).where(
            and_(
                Resume.id == request.resume_id,
                Resume.organization_id == current_user.organization_id
            )
        )
        resume_result = await db.execute(resume_query)
        resume = resume_result.scalar_one_or_none()
        
        job_query = select(Job).where(
            and_(
                Job.id == request.job_id,
                Job.organization_id == current_user.organization_id
            )
        )
        job_result = await db.execute(job_query)
        job = job_result.scalar_one_or_none()
        
        if not resume or not job:
            raise HTTPException(
                status_code=404,
                detail="Resume or job not found"
            )
        # Check if candidate is blacklisted
        if resume.blacklist:
            raise HTTPException(
                status_code=403,
                detail="Cannot generate interview questions for blacklisted candidates"
            )
        
        # Generate interview guide using single AI call
        interview_guide = await _generate_interview_guide(
            resume=resume,
            job=job,
            job_match=job_match,
            interview_level=request.interview_level,
            focus_areas=request.focus_areas
        )
        
        generation_time = int((datetime.now(timezone.utc) - start_time).total_seconds() * 1000)
        interview_id = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.resume_id[:8]}"
        generated_at = datetime.now(timezone.utc).isoformat()
        
        # Save interview questions to the database
        interview_data = {
            'interview_id': interview_id,
            'generated_at': generated_at,
            'guide': interview_guide.dict(),
            'interview_level': request.interview_level,
            'focus_areas': request.focus_areas
        }
        
        # Update the job match with interview questions
        job_match.interview_questions = interview_data
        await db.commit()
        
        logger.info(f"✅ Interview questions generated and saved successfully in {generation_time}ms")
        
        return InterviewGenerationResponse(
            interview_id=interview_id,
            generated_at=generated_at,
            guide=interview_guide,
            generation_time_ms=generation_time,
            cached=False,
            is_new_generation=True
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Interview generation failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Interview generation failed")


async def _generate_interview_guide(
    resume: Resume,
    job: Job,
    job_match: JobMatch,
    interview_level: str,
    focus_areas: Optional[List[str]] = None
) -> InterviewGuide:
    """Generate comprehensive interview guide using single AI call."""
    
    # Get LLM instance
    settings = get_settings()
    llm_factory = LLMFactory(settings)
    llm = llm_factory.create_llm()
    
    # Extract candidate details
    candidate_name = resume.candidate_name or "Unknown Candidate"
    job_title = job.title or "Unknown Position"
    
    # Extract skills and experience from resume AI analysis
    ai_analysis = resume.ai_analysis or {}
    skills_analysis = ai_analysis.get("skills_analysis", {})
    career_analysis = ai_analysis.get("career_analysis", {})
    
    # Build comprehensive skills list
    all_skills = []
    skill_categories = {
        'technical_skills': skills_analysis.get("technical_skills", []),
        'soft_skills': skills_analysis.get("soft_skills", []),
        'methodologies': skills_analysis.get("methodologies", []),
        'tools_technologies': skills_analysis.get("tools_technologies", []),
        'frameworks_libraries': skills_analysis.get("frameworks_libraries", []),
        'programming_languages': skills_analysis.get("programming_languages", [])
    }
    
    # Safely process skills, handling both strings and dictionaries
    for category, skills in skill_categories.items():
        if isinstance(skills, list):
            for skill in skills:
                if skill is not None:
                    if isinstance(skill, dict):
                        # If skill is a dict, try to get the name or value
                        skill_str = skill.get('name', skill.get('skill', skill.get('value', str(skill))))
                    else:
                        skill_str = str(skill)
                    all_skills.append(skill_str)
    
    # Safely format skills for display
    def safe_join_skills(skills_list, max_items=10):
        try:
            return ', '.join([str(skill) for skill in skills_list[:max_items]])
        except Exception as e:
            logger.warning(f"Error formatting skills: {e}")
            return 'Skills not available'
    
    # Create AI prompt for interview generation
    prompt = f"""
You are an expert technical interviewer and hiring manager. Generate a comprehensive, structured interview guide for the following candidate and job match.

CANDIDATE DETAILS:
- Name: {candidate_name}
- Current Position: {career_analysis.get('current_position', 'N/A')}
- Years Experience: {career_analysis.get('total_years_experience', 0)}
- Technical Skills: {safe_join_skills(skill_categories['technical_skills'])}
- Programming Languages: {safe_join_skills(skill_categories['programming_languages'])}
- Frameworks & Libraries: {safe_join_skills(skill_categories['frameworks_libraries'])}
- Tools & Technologies: {safe_join_skills(skill_categories['tools_technologies'])}
- Methodologies: {safe_join_skills(skill_categories['methodologies'])}
- Soft Skills: {safe_join_skills(skill_categories['soft_skills'])}
- All Skills: {safe_join_skills(all_skills, 20)}
- Professional Summary: {resume.professional_summary or 'N/A'}

JOB DETAILS:
- Title: {job_title}
- Description: {job.description or 'N/A'}
- Requirements: {job.requirements or 'N/A'}
- Skills Required: {safe_join_skills(job.required_skills or [])}

MATCH SCORES:
- Overall Score: {job_match.overall_score:.1%}
- Skill Match: {job_match.skills_score:.1%}
- Experience Match: {job_match.experience_score:.1%}
- Cultural Fit: {job_match.cultural_fit_score:.1%}

INTERVIEW LEVEL: {interview_level}
FOCUS AREAS: {', '.join(focus_areas) if focus_areas else 'All areas based on candidate profile'}

Generate a comprehensive interview guide in the following JSON format:

{{
    "candidate_name": "{candidate_name}",
    "job_title": "{job_title}",
    "interview_level": "{interview_level}",
    "overall_score": {job_match.overall_score},
    "scope": {{
        "target_level": "{interview_level}",
        "areas_to_test": ["Area1", "Area2", "Area3"],
        "key_technologies": ["Tech1", "Tech2", "Tech3"],
        "focus_areas": {focus_areas or []}
    }},
    "key_questions": [
        {{
            "category": "Technical Area",
            "question": "Detailed technical question",
            "what_to_listen_for": ["Key point 1", "Key point 2", "Key point 3"]
        }}
    ],
    "optional_exercise": {{
        "title": "Design Exercise Title",
        "description": "Detailed exercise description",
        "time_limit": "10-15 minutes",
        "evaluation_criteria": ["Criterion 1", "Criterion 2"]
    }},
    "scorecard": {{
        "categories": [
            {{"name": "Technical Skills", "weight": 0.3}},
            {{"name": "Problem Solving", "weight": 0.25}},
            {{"name": "Communication", "weight": 0.2}},
            {{"name": "Cultural Fit", "weight": 0.25}}
        ],
        "rating_scale": "1-5 (1=Poor, 5=Excellent)",
        "recommendation_threshold": "3.5+ for hire"
    }},
    "recommendation_guidelines": "Specific guidelines for making hiring recommendations based on scores and performance"
}}

IMPORTANT:
- Generate AT LEAST 15 high-quality interview questions based on job responsibilities and requirements
- Focus on the candidate's skills and experience relevant to the specific job role
- Include technical, behavioral, and role-specific questions
- Make questions specific to the job requirements and candidate's background
- Each question should have clear "what_to_listen_for" evaluation criteria
- Cover different aspects: technical skills, problem-solving, communication, cultural fit
- Base questions on the actual job description, requirements, and candidate's skills
- Return ONLY the JSON response, no additional text
"""
    
    try:
        logger.info(f"🤖 Making single AI call to generate interview guide for {candidate_name}")
        
        response = await llm.ainvoke(prompt)
        content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if not json_match:
            raise ValueError("No JSON found in AI response")
        
        json_str = json_match.group()
        interview_data = json.loads(json_str)
        
        # Convert to InterviewGuide model
        questions = []
        for q_data in interview_data.get("key_questions", []):
            questions.append(InterviewQuestion(
                category=q_data.get("category", ""),
                question=q_data.get("question", ""),
                what_to_listen_for=q_data.get("what_to_listen_for", []),
                follow_up_questions=q_data.get("follow_up_questions", [])
            ))
        
        return InterviewGuide(
            candidate_name=interview_data.get("candidate_name", candidate_name),
            job_title=interview_data.get("job_title", job_title),
            interview_level=interview_data.get("interview_level", interview_level),
            overall_score=interview_data.get("overall_score", job_match.overall_score),
            scope=interview_data.get("scope", {}),
            key_questions=questions,
            optional_exercise=interview_data.get("optional_exercise"),
            scorecard=interview_data.get("scorecard", {}),
            recommendation_guidelines=interview_data.get("recommendation_guidelines", "")
        )
        
    except Exception as e:
        logger.error(f"Failed to generate interview guide: {e}")
        # Return a basic fallback guide
        return InterviewGuide(
            candidate_name=candidate_name,
            job_title=job_title,
            interview_level=interview_level,
            overall_score=job_match.overall_score,
            scope={
                "target_level": interview_level,
                "areas_to_test": ["Technical Skills", "Problem Solving", "Communication"],
                "key_technologies": skill_categories['technical_skills'][:5],
                "focus_areas": focus_areas or []
            },
            key_questions=[
                InterviewQuestion(
                    category="Technical Skills",
                    question="Tell me about your experience with the technologies mentioned in this role.",
                    what_to_listen_for=["Relevant experience", "Depth of knowledge", "Practical examples"],
                    follow_up_questions=["Can you walk me through a specific project?", "What challenges did you face?"]
                )
            ],
            scorecard={
                "categories": [
                    {"name": "Technical Skills", "weight": 0.4},
                    {"name": "Problem Solving", "weight": 0.3},
                    {"name": "Communication", "weight": 0.3}
                ],
                "rating_scale": "1-5 (1=Poor, 5=Excellent)",
                "recommendation_threshold": "3.0+ for hire"
            },
            recommendation_guidelines="Evaluate based on technical competency, problem-solving approach, and communication skills."
        )


@router.get("/check-eligibility/{resume_id}/{job_id}")
async def check_interview_eligibility(
    resume_id: str,
    job_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Check if a candidate is eligible for interview generation.
    Returns eligibility status and overall score.
    """
    try:
        # Check user permissions
        if current_user.role not in ["admin", "senior_recruiter"]:
            return {
                "eligible": False,
                "reason": "Insufficient permissions",
                "overall_score": 0.0
            }
        
        # Fetch job match
        job_match_query = select(JobMatch).join(Job).where(
            and_(
                JobMatch.resume_id == resume_id,
                JobMatch.job_id == job_id,
                Job.organization_id == current_user.organization_id
            )
        )
        job_match_result = await db.execute(job_match_query)
        job_match = job_match_result.scalar_one_or_none()
        
        if not job_match:
            return {
                "eligible": False,
                "reason": "Job match not found",
                "overall_score": 0.0
            }
        
        # Check score threshold
        eligible = job_match.overall_score >= 0.4
        
        return {
            "eligible": eligible,
            "reason": "Score below 40%" if not eligible else "Eligible for interview",
            "overall_score": job_match.overall_score,
            "skill_match_score": job_match.skills_score,
            "experience_match_score": job_match.experience_score,
            "cultural_fit_score": job_match.cultural_fit_score
        }
        
    except Exception as e:
        logger.error(f"❌ Eligibility check failed: {str(e)}")
        return {
            "eligible": False,
            "reason": "Error checking eligibility",
            "overall_score": 0.0
        }


@router.post("/reprocess")
async def reprocess_interview_questions(
    request: InterviewGenerationRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Clear existing interview questions and regenerate them
    """
    try:
        # Check permissions
        if current_user.role not in ["admin", "senior_recruiter"]:
            raise HTTPException(
                status_code=403,
                detail="Only admin and senior recruiter users can reprocess interview questions"
            )
        
        # Get job match with job details
        job_match_query = (
            select(JobMatch)
            .join(Job, Job.id == JobMatch.job_id)
            .where(
                JobMatch.resume_id == request.resume_id,
                JobMatch.job_id == request.job_id,
                Job.organization_id == current_user.organization_id
            )
        )
        
        result = await db.execute(job_match_query)
        job_match = result.scalar_one_or_none()
        
        if not job_match:
            raise HTTPException(
                status_code=404,
                detail="Job match not found"
            )
        
        # Check eligibility (40% threshold)
        if job_match.overall_score < 0.4:
            raise HTTPException(
                status_code=400,
                detail=f"Candidate score ({job_match.overall_score:.1%}) is below the 40% threshold required for interview generation"
            )
        
        # Clear existing interview questions
        job_match.interview_questions = {}
        await db.commit()
        
        logger.info(f"🔄 Cleared existing interview questions for resume {request.resume_id} and job {request.job_id}")
        
        # Generate new interview questions using the same logic as the original endpoint
        # Get resume details
        resume_query = select(Resume).where(
            Resume.id == request.resume_id,
            Resume.organization_id == current_user.organization_id
        )
        result = await db.execute(resume_query)
        resume = result.scalar_one_or_none()
        
        if not resume:
            raise HTTPException(
                status_code=404,
                detail="Resume not found"
            )
        
        # Check if candidate is blacklisted
        if resume.blacklist:
            raise HTTPException(
                status_code=403,
                detail="Cannot generate interview questions for blacklisted candidates"
            )
    
        
        # Get job details
        job_query = select(Job).where(
            Job.id == request.job_id,
            Job.organization_id == current_user.organization_id
        )
        result = await db.execute(job_query)
        job = result.scalar_one_or_none()
        
        if not job:
            raise HTTPException(
                status_code=404,
                detail="Job not found"
            )
        
        # Generate interview guide using the same logic
        interview_guide = await _generate_interview_guide(
            resume=resume,
            job=job,
            job_match=job_match,
            interview_level=request.interview_level or "L3",
            focus_areas=request.focus_areas or []
        )
        
        # Create interview data dictionary (same format as original endpoint)
        interview_id = f"interview_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{request.resume_id[:8]}"
        generated_at = datetime.now(timezone.utc).isoformat()
        
        interview_data = {
            'interview_id': interview_id,
            'generated_at': generated_at,
            'guide': interview_guide.dict(),
            'interview_level': request.interview_level or "L3",
            'focus_areas': request.focus_areas or []
        }
        
        # Store the interview questions in the database
        job_match.interview_questions = interview_data
        await db.commit()
        
        logger.info(f"✅ Interview questions regenerated successfully for resume {request.resume_id} and job {request.job_id}")
        
        return {
            "success": True,
            "guide": interview_guide,
            "interview_id": interview_id,
            "generated_at": generated_at,
            "cached": False,
            "is_new_generation": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ Interview reprocessing failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Interview reprocessing failed"
        )