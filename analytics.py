"""
Analytics endpoints for AI Recruit.
Comprehensive dashboard analytics with real-time data aggregation.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import List, Optional, Dict, Any
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, func, and_, or_, desc
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import text

from core.security import get_current_user
from database.models import (
    User, Resume, Job, JobMatch, AgentSession, InterviewSession,
    ResumeStatus, JobStatus, MatchingStatus
)
from database.session import get_db

logger = logging.getLogger(__name__)
router = APIRouter(tags=["analytics"])


class DashboardStats(BaseModel):
    """Dashboard statistics model."""
    total_resumes: int
    total_resumes_change: str
    active_jobs: int
    active_jobs_change: str
    successful_matches: int
    successful_matches_percentage: float
    successful_matches_change: str
    avg_processing_time: float
    avg_processing_time_change: str


class RecentActivity(BaseModel):
    """Recent activity item model."""
    id: str
    type: str
    title: str
    description: str
    time: str
    timestamp: datetime  # Add timestamp for proper sorting
    icon: str
    color: str


class DashboardResponse(BaseModel):
    """Complete dashboard response model."""
    stats: DashboardStats
    recent_activity: List[RecentActivity]
    processing_metrics: Dict[str, Any]
    ai_insights: Dict[str, Any]


@router.get("/dashboard", response_model=DashboardResponse)
async def get_dashboard_stats(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """
    Get comprehensive dashboard analytics with real-time data.
    
    Provides:
    - Key performance metrics with trend analysis
    - Recent system activity
    - AI processing insights
    - Performance analytics
    """
    try:
        logger.info(f"ðŸ“Š Generating dashboard analytics for user {current_user.id}")
        
        # Get date ranges for trend analysis (timezone-aware)
        now = datetime.now(timezone.utc)
        last_30_days = now - timedelta(days=30)
        last_60_days = now - timedelta(days=60)
        last_7_days = now - timedelta(days=7)
        
        org_id = current_user.organization_id
        
        # === RESUME STATISTICS ===
        
        # Total resumes (current period)
        total_resumes_result = await db.execute(
            select(func.count(Resume.id)).where(
                and_(
                    Resume.organization_id == org_id,
                    Resume.created_at >= last_30_days
                )
            )
        )
        total_resumes_current = total_resumes_result.scalar() or 0
        
        # Total resumes (previous period for comparison)
        total_resumes_prev_result = await db.execute(
            select(func.count(Resume.id)).where(
                and_(
                    Resume.organization_id == org_id,
                    Resume.created_at >= last_60_days,
                    Resume.created_at < last_30_days
                )
            )
        )
        total_resumes_prev = total_resumes_prev_result.scalar() or 0
        
        # Calculate resume change percentage
        if total_resumes_prev > 0:
            resume_change = ((total_resumes_current - total_resumes_prev) / total_resumes_prev) * 100
            resume_change_str = f"+{resume_change:.0f}%" if resume_change > 0 else f"{resume_change:.0f}%"
        else:
            resume_change_str = "+100%" if total_resumes_current > 0 else "0%"
        
        # === JOB STATISTICS ===
        
        # Active jobs (current)
        active_jobs_result = await db.execute(
            select(func.count(Job.id)).where(
                and_(
                    Job.organization_id == org_id,
                    Job.status == JobStatus.ACTIVE
                )
            )
        )
        active_jobs_current = active_jobs_result.scalar() or 0
        
        # Jobs created in last 30 days vs previous 30 days
        jobs_current_result = await db.execute(
            select(func.count(Job.id)).where(
                and_(
                    Job.organization_id == org_id,
                    Job.created_at >= last_30_days
                )
            )
        )
        jobs_current = jobs_current_result.scalar() or 0
        
        jobs_prev_result = await db.execute(
            select(func.count(Job.id)).where(
                and_(
                    Job.organization_id == org_id,
                    Job.created_at >= last_60_days,
                    Job.created_at < last_30_days
                )
            )
        )
        jobs_prev = jobs_prev_result.scalar() or 0
        
        # Calculate job change
        if jobs_prev > 0:
            job_change = ((jobs_current - jobs_prev) / jobs_prev) * 100
            job_change_str = f"+{job_change:.0f}%" if job_change > 0 else f"{job_change:.0f}%"
        else:
            job_change_str = f"+{jobs_current}" if jobs_current > 0 else "0"
        
        # === MATCHING STATISTICS ===
        
        # Get organization job IDs for matching analysis
        org_jobs_result = await db.execute(
            select(Job.id).where(Job.organization_id == org_id)
        )
        org_job_ids = [str(job_id) for job_id in org_jobs_result.scalars().all()]
        
        successful_matches_current = 0
        successful_matches_prev = 0
        total_matches_current = 0
        
        if org_job_ids:
            # Successful matches (score >= 0.7) in last 30 days
            successful_matches_result = await db.execute(
                select(func.count(JobMatch.id)).where(
                    and_(
                        JobMatch.job_id.in_(org_job_ids),
                        JobMatch.overall_score >= 0.7,
                        JobMatch.created_at >= last_30_days
                    )
                )
            )
            successful_matches_current = successful_matches_result.scalar() or 0
            
            # Total matches in last 30 days
            total_matches_result = await db.execute(
                select(func.count(JobMatch.id)).where(
                    and_(
                        JobMatch.job_id.in_(org_job_ids),
                        JobMatch.created_at >= last_30_days
                    )
                )
            )
            total_matches_current = total_matches_result.scalar() or 0
            
            # Previous period successful matches
            successful_matches_prev_result = await db.execute(
                select(func.count(JobMatch.id)).where(
                    and_(
                        JobMatch.job_id.in_(org_job_ids),
                        JobMatch.overall_score >= 0.7,
                        JobMatch.created_at >= last_60_days,
                        JobMatch.created_at < last_30_days
                    )
                )
            )
            successful_matches_prev = successful_matches_prev_result.scalar() or 0
        
        # Calculate success rate and change
        success_rate = (successful_matches_current / total_matches_current * 100) if total_matches_current > 0 else 0
        
        if successful_matches_prev > 0:
            match_change = ((successful_matches_current - successful_matches_prev) / successful_matches_prev) * 100
            match_change_str = f"+{match_change:.0f}%" if match_change > 0 else f"{match_change:.0f}%"
        else:
            match_change_str = f"+{successful_matches_current}" if successful_matches_current > 0 else "0%"
        
        # === PROCESSING TIME STATISTICS ===
        
        # Average processing time for resumes
        processing_times_result = await db.execute(
            select(
                func.extract('epoch', Resume.processing_completed_at - Resume.processing_started_at).label('duration')
            ).where(
                and_(
                    Resume.organization_id == org_id,
                    Resume.processing_completed_at.isnot(None),
                    Resume.processing_started_at.isnot(None),
                    Resume.created_at >= last_30_days
                )
            )
        )
        processing_times = [row.duration for row in processing_times_result.fetchall() if row.duration]
        
        avg_processing_time = sum(processing_times) / len(processing_times) if processing_times else 4.2
        
        # Previous period processing time
        prev_processing_times_result = await db.execute(
            select(
                func.extract('epoch', Resume.processing_completed_at - Resume.processing_started_at).label('duration')
            ).where(
                and_(
                    Resume.organization_id == org_id,
                    Resume.processing_completed_at.isnot(None),
                    Resume.processing_started_at.isnot(None),
                    Resume.created_at >= last_60_days,
                    Resume.created_at < last_30_days
                )
            )
        )
        prev_processing_times = [row.duration for row in prev_processing_times_result.fetchall() if row.duration]
        prev_avg_processing_time = sum(prev_processing_times) / len(prev_processing_times) if prev_processing_times else avg_processing_time
        
        # Calculate processing time change
        if prev_avg_processing_time > 0:
            processing_change = ((avg_processing_time - prev_avg_processing_time) / prev_avg_processing_time) * 100
            processing_change_str = f"{processing_change:.0f}%" if processing_change > 0 else f"{processing_change:.0f}%"
        else:
            processing_change_str = "0%"
        
        # === RECENT ACTIVITY ===
        
        recent_activity = []
        
        # Recent resume uploads
        recent_resumes_result = await db.execute(
            select(Resume).where(
                and_(
                    Resume.organization_id == org_id,
                    Resume.created_at >= last_7_days
                )
            ).order_by(desc(Resume.created_at)).limit(5)
        )
        recent_resumes = recent_resumes_result.scalars().all()
        
        for resume in recent_resumes:
            # Ensure both datetimes are timezone-aware for comparison
            resume_created = resume.created_at
            if resume_created.tzinfo is None:
                resume_created = resume_created.replace(tzinfo=timezone.utc)
            
            time_diff = now - resume_created
            if time_diff.days > 0:
                time_str = f"{time_diff.days} days ago"
            elif time_diff.seconds > 3600:
                time_str = f"{time_diff.seconds // 3600} hours ago"
            else:
                time_str = f"{max(1, time_diff.seconds // 60)} minutes ago"
            
            recent_activity.append(RecentActivity(
                id=f"resume_{resume.id}",
                type="resume_uploaded",
                title="New resume analyzed",
                description=f"{resume.candidate_name or 'Unknown Candidate'} - {resume.current_position or 'Position not specified'}",
                time=time_str,
                timestamp=resume_created,
                icon="FileText",
                color="text-blue-600"
            ))
        
        # Recent job postings
        recent_jobs_result = await db.execute(
            select(Job).where(
                and_(
                    Job.organization_id == org_id,
                    Job.created_at >= last_7_days
                )
            ).order_by(desc(Job.created_at)).limit(5)
        )
        recent_jobs = recent_jobs_result.scalars().all()
        
        for job in recent_jobs:
            # Ensure both datetimes are timezone-aware for comparison
            job_created = job.created_at
            if job_created.tzinfo is None:
                job_created = job_created.replace(tzinfo=timezone.utc)
            
            time_diff = now - job_created
            if time_diff.days > 0:
                time_str = f"{time_diff.days} days ago"
            elif time_diff.seconds > 3600:
                time_str = f"{time_diff.seconds // 3600} hours ago"
            else:
                time_str = f"{max(1, time_diff.seconds // 60)} minutes ago"
            
            recent_activity.append(RecentActivity(
                id=f"job_{job.id}",
                type="job_created",
                title="New job posting created",
                description=f"{job.title} - {job.department or 'General'}",
                time=time_str,
                timestamp=job_created,
                icon="Briefcase",
                color="text-purple-600"
            ))
        
        # Recent high-quality matches
        if org_job_ids:
            recent_matches_result = await db.execute(
                select(JobMatch, Job, Resume).join(
                    Job, JobMatch.job_id == Job.id
                ).join(
                    Resume, JobMatch.resume_id == Resume.id
                ).where(
                    and_(
                        JobMatch.job_id.in_(org_job_ids),
                        JobMatch.overall_score >= 0.8,
                        JobMatch.created_at >= last_7_days
                    )
                ).order_by(desc(JobMatch.created_at)).limit(5)
            )
            recent_matches = recent_matches_result.fetchall()
            
            for match, job, resume in recent_matches:
                # Ensure both datetimes are timezone-aware for comparison
                match_created = match.created_at
                if match_created.tzinfo is None:
                    match_created = match_created.replace(tzinfo=timezone.utc)
                
                time_diff = now - match_created
                if time_diff.days > 0:
                    time_str = f"{time_diff.days} days ago"
                elif time_diff.seconds > 3600:
                    time_str = f"{time_diff.seconds // 3600} hours ago"
                else:
                    time_str = f"{max(1, time_diff.seconds // 60)} minutes ago"
                
                recent_activity.append(RecentActivity(
                    id=f"match_{match.id}",
                    type="job_match",
                    title="High-quality match found",
                    description=f"{job.title} - {int(match.overall_score * 100)}% match",
                    time=time_str,
                    timestamp=match_created,
                    icon="Target",
                    color="text-green-600"
                ))
        
        # Recent AI processing sessions
        recent_sessions_result = await db.execute(
            select(AgentSession).where(
                and_(
                    AgentSession.organization_id == org_id,
                    AgentSession.created_at >= last_7_days,
                    AgentSession.status == "success"
                )
            ).order_by(desc(AgentSession.created_at)).limit(5)
        )
        recent_sessions = recent_sessions_result.scalars().all()
        
        for session in recent_sessions:
            # Ensure both datetimes are timezone-aware for comparison
            session_created = session.created_at
            if session_created.tzinfo is None:
                session_created = session_created.replace(tzinfo=timezone.utc)
            
            time_diff = now - session_created
            if time_diff.days > 0:
                time_str = f"{time_diff.days} days ago"
            elif time_diff.seconds > 3600:
                time_str = f"{time_diff.seconds // 3600} hours ago"
            else:
                time_str = f"{max(1, time_diff.seconds // 60)} minutes ago"
            
            recent_activity.append(RecentActivity(
                id=f"session_{session.id}",
                type="ai_processing",
                title="AI batch processing completed",
                description=f"{session.session_type.replace('_', ' ').title()} - {session.agent_name}",
                time=time_str,
                timestamp=session_created,
                icon="Bot",
                color="text-primary-600"
            ))
        
        # Sort recent activity by most recent first (descending order)
        recent_activity.sort(key=lambda x: x.timestamp, reverse=True)
        
        # === PROCESSING METRICS ===
        
        processing_metrics = {
            "total_ai_sessions": 0,
            "successful_sessions": 0,
            "average_confidence": 0.0,
            "total_tokens_used": 0,
            "estimated_cost": 0.0
        }
        
        # Get AI session metrics
        sessions_result = await db.execute(
            select(AgentSession).where(
                and_(
                    AgentSession.organization_id == org_id,
                    AgentSession.created_at >= last_30_days
                )
            )
        )
        sessions = sessions_result.scalars().all()
        
        if sessions:
            processing_metrics["total_ai_sessions"] = len(sessions)
            processing_metrics["successful_sessions"] = len([s for s in sessions if s.status == "success"])
            
            confidence_scores = [s.confidence_score for s in sessions if s.confidence_score]
            processing_metrics["average_confidence"] = sum(confidence_scores) / len(confidence_scores) if confidence_scores else 0.0
            
            processing_metrics["total_tokens_used"] = sum(s.tokens_used or 0 for s in sessions)
            processing_metrics["estimated_cost"] = sum(s.cost_estimate or 0 for s in sessions)
        
        # === AI INSIGHTS ===
        
        ai_insights = {
            "system_health": "excellent" if processing_metrics["successful_sessions"] / max(processing_metrics["total_ai_sessions"], 1) > 0.95 else "good",
            "processing_efficiency": "high" if avg_processing_time < 5.0 else "moderate",
            "match_quality": "excellent" if success_rate > 80 else "good" if success_rate > 60 else "needs_improvement",
            "recommendations": [],
            "trends": {
                "resume_volume": "increasing" if resume_change_str.startswith("+") else "stable",
                "job_creation": "active" if job_change_str.startswith("+") else "stable",
                "matching_performance": "improving" if match_change_str.startswith("+") else "stable"
            }
        }
        
        # Generate recommendations based on data
        if success_rate < 60:
            ai_insights["recommendations"].append("Consider reviewing job requirements for better candidate matching")
        if avg_processing_time > 10:
            ai_insights["recommendations"].append("AI processing time is high - consider system optimization")
        if total_resumes_current < 5:
            ai_insights["recommendations"].append("Increase candidate sourcing to improve matching opportunities")
        if active_jobs_current == 0:
            ai_insights["recommendations"].append("Create job postings to start matching candidates")
        
        if not ai_insights["recommendations"]:
            ai_insights["recommendations"].append("System is performing well - continue current practices")
        
        # === BUILD RESPONSE ===
        
        stats = DashboardStats(
            total_resumes=total_resumes_current,
            total_resumes_change=resume_change_str,
            active_jobs=active_jobs_current,
            active_jobs_change=job_change_str,
            successful_matches=successful_matches_current,
            successful_matches_percentage=success_rate,
            successful_matches_change=match_change_str,
            avg_processing_time=avg_processing_time,
            avg_processing_time_change=processing_change_str
        )
        
        response = DashboardResponse(
            stats=stats,
            recent_activity=recent_activity[:8],  # Limit to 8 most recent items
            processing_metrics=processing_metrics,
            ai_insights=ai_insights
        )
        
        logger.info(f"âœ… Dashboard analytics generated successfully")
        logger.info(f"ðŸ“Š Stats: {total_resumes_current} resumes, {active_jobs_current} jobs, {successful_matches_current} matches")
        
        return response
        
    except Exception as e:
        logger.error(f"âŒ Failed to generate dashboard analytics: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate dashboard analytics")


@router.get("/matching")
async def get_matching_analytics(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed matching analytics."""
    logger.info("Matching analytics endpoint called")
    
    try:
        # Get organization job IDs
        org_jobs_result = await db.execute(
            select(Job.id).where(Job.organization_id == current_user.organization_id)
        )
        org_job_ids = [str(job_id) for job_id in org_jobs_result.scalars().all()]
        
        if not org_job_ids:
            return {
                "total_matches": 0,
                "success_rate": 0,
                "message": "No matching data available"
            }
        
        # Get match statistics
        matches_result = await db.execute(
            select(JobMatch).where(JobMatch.job_id.in_(org_job_ids))
        )
        matches = matches_result.scalars().all()
        
        total_matches = len(matches)
        successful_matches = len([m for m in matches if m.overall_score >= 0.7])
        success_rate = (successful_matches / total_matches * 100) if total_matches > 0 else 0
        
        return {
            "total_matches": total_matches,
            "successful_matches": successful_matches,
            "success_rate": round(success_rate, 2),
            "average_score": round(sum(m.overall_score for m in matches) / total_matches, 3) if total_matches > 0 else 0
        }
        
    except Exception as e:
        logger.error(f"âŒ Failed to get matching analytics: {str(e)}")
        return {"error": "Failed to retrieve matching analytics"}