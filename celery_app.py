"""
AI Recruit - Celery Application Configuration
PostgreSQL-based task queue for background AI processing.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

import logging
import asyncio
import os
from celery import Celery
from kombu import Exchange, Queue

from core.config import get_settings

# Disable LangSmith tracing to avoid 403 errors in Celery workers
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ.pop("LANGSMITH_API_KEY", None)

logger = logging.getLogger(__name__)

# Get settings
settings = get_settings()

# Create Celery app
celery_app = Celery(
    "celery_app",
    broker=settings.CELERY_BROKER_URL,
    backend=settings.CELERY_RESULT_BACKEND,
    include=[
        "tasks.resume_processing",
        "tasks.job_matching", 
        "tasks.quality_assessment",
        "tasks.analytics",
        "tasks.job_processing"
    ]
)

# Configure Celery
celery_app.conf.update(
    # Task serialization
    task_serializer=settings.CELERY_TASK_SERIALIZER,
    result_serializer=settings.CELERY_RESULT_SERIALIZER,
    accept_content=settings.CELERY_ACCEPT_CONTENT,
    
    # Timezone settings
    timezone=settings.CELERY_TIMEZONE,
    enable_utc=settings.CELERY_ENABLE_UTC,
    
    # Event loop configuration - use solo pool to avoid event loop conflicts
    worker_pool='solo',  # Use solo pool to avoid event loop conflicts with async code
    worker_concurrency=1,  # Single worker to avoid conflicts
    
    # Task routing
    task_routes={
        'tasks.resume_processing.*': {'queue': 'resume_analysis'},
        'tasks.job_matching.*': {'queue': 'job_matching'},
        'tasks.quality_assessment.*': {'queue': 'quality_assessment'},
        'tasks.analytics.*': {'queue': 'analytics'},
        'tasks.job_processing.*': {'queue': 'job_analysis'},
    },
    
    # Queues
    task_default_queue='default',
    task_queues=(
        Queue('default', Exchange('default'), routing_key='default'),
        Queue('resume_analysis', Exchange('resume_analysis'), routing_key='resume_analysis'),
        Queue('job_matching', Exchange('job_matching'), routing_key='job_matching'),
        Queue('quality_assessment', Exchange('quality_assessment'), routing_key='quality_assessment'),
        Queue('analytics', Exchange('analytics'), routing_key='analytics'),
        Queue('job_analysis', Exchange('job_analysis'), routing_key='job_analysis'),
    ),
    
    # Worker configuration
    worker_prefetch_multiplier=1,
    task_acks_late=True,
    worker_disable_rate_limits=False,
    
    # Task execution (derive from PROCESSING_TIMEOUT_SECONDS)
    task_soft_time_limit=settings.PROCESSING_TIMEOUT_SECONDS,
    task_time_limit=settings.PROCESSING_TIMEOUT_SECONDS * 2,
    task_max_retries=3,
    task_default_retry_delay=60,
    
    # Result backend settings
    result_expires=3600,       # 1 hour
    result_persistent=True,
    
    # Beat schedule (for periodic tasks)
    beat_schedule={
        'cleanup-old-results': {
            'task': 'tasks.analytics.cleanup_old_results',
            'schedule': 3600.0,  # Run every hour
        },
        'update-analytics': {
            'task': 'tasks.analytics.update_analytics_cache',
            'schedule': 300.0,   # Run every 5 minutes
        },
    },
    
    # Monitoring
    worker_send_task_events=True,
    task_send_sent_event=True,
    
    # Security
    worker_hijack_root_logger=False,
    worker_log_color=False,
)

# Configure logging
@celery_app.on_configure.connect
def configure_logging(**kwargs):
    """Configure logging for Celery."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

@celery_app.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery functionality."""
    logger.info(f'Request: {self.request!r}')
    return 'Debug task completed successfully'

# Health check task
@celery_app.task
def health_check():
    """Health check task for monitoring."""
    return {
        "status": "healthy",
        "message": "Celery worker is operational",
        "broker": settings.CELERY_BROKER_URL.split('@')[0] + '@***',  # Hide credentials
        "backend": settings.CELERY_RESULT_BACKEND.split('@')[0] + '@***'
    }

if __name__ == '__main__':
    celery_app.start()