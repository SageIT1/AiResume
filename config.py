"""
AI Recruit - Agentic AI Platform Configuration
Comprehensive settings management with support for multiple LLM providers and Azure storage.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from functools import lru_cache
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timezone
import json
import os

from pydantic_settings import BaseSettings
from pydantic import Field, validator
from pydantic.networks import AnyHttpUrl


class Settings(BaseSettings):
    """
    Application settings with comprehensive LLM provider support.
    All configuration comes from environment variables.
    """
    
    # ===========================================
    # APPLICATION SETTINGS
    # ===========================================
    ENVIRONMENT: Literal["development", "staging", "production"] = "development"
    APP_NAME: str = "ai-recruit"
    API_VERSION: str = "v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 4
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"
    
    # ===========================================
    # SECURITY & AUTHENTICATION
    # ===========================================
    SECRET_KEY: str = Field(..., env="SECRET_KEY")
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    REFRESH_TOKEN_EXPIRE_MINUTES: int = 10080
    ALLOWED_ORIGINS: List[str] = ["http://localhost:3000", "http://localhost:8000"]
    ALLOWED_HOSTS: List[str] = ["localhost", "127.0.0.1"]
    DEFAULT_NEW_USER_PASSWORD: str = Field("ChangeMe123!", env="DEFAULT_NEW_USER_PASSWORD")
    
    @validator("ALLOWED_ORIGINS", pre=True)
    def assemble_cors_origins(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    @validator("ALLOWED_HOSTS", pre=True)
    def assemble_allowed_hosts(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    # ===========================================
    # DATABASE CONFIGURATION
    # ===========================================
    DATABASE_URL: str = Field(..., env="DATABASE_URL")
    DATABASE_POOL_SIZE: int = 10
    DATABASE_MAX_OVERFLOW: int = 20
    DATABASE_POOL_TIMEOUT: int = 30
    
    # ===========================================
    # CELERY CONFIGURATION (PostgreSQL-based)
    # ===========================================
    CELERY_BROKER_URL: str = Field(..., env="CELERY_BROKER_URL")
    CELERY_RESULT_BACKEND: str = Field(..., env="CELERY_RESULT_BACKEND")
    CELERY_TASK_SERIALIZER: str = "json"
    CELERY_RESULT_SERIALIZER: str = "json"
    CELERY_ACCEPT_CONTENT: List[str] = ["json"]
    CELERY_TIMEZONE: str = "UTC"
    CELERY_ENABLE_UTC: bool = True
    
    # ===========================================
    # AZURE STORAGE CONFIGURATION
    # ===========================================
    AZURE_STORAGE_ACCOUNT_NAME: str = Field(..., env="AZURE_STORAGE_ACCOUNT_NAME")
    AZURE_STORAGE_ACCOUNT_KEY: str = Field(..., env="AZURE_STORAGE_ACCOUNT_KEY")
    AZURE_STORAGE_CONNECTION_STRING: str = Field(..., env="AZURE_STORAGE_CONNECTION_STRING")
    AZURE_BLOB_CONTAINER_RESUMES: str = "resumes"
    AZURE_BLOB_CONTAINER_DOCUMENTS: str = "documents"
    AZURE_BLOB_CONTAINER_PROCESSED: str = "processed-files"
    AZURE_BLOB_CONTAINER_FORMATTED_RESUMES: str = "formatted-resumes"
    AZURE_STORAGE_URL_EXPIRY_HOURS: int = 24
    
    # ===========================================
    # LLM PROVIDER CONFIGURATION
    # ===========================================
    LLM_PROVIDER: Literal[
        "openai", "azure_openai", "anthropic", "together", 
        "huggingface", "google", "cohere", "ollama"
    ] = "openai"
    LLM_FALLBACK_PROVIDERS: List[str] = ["anthropic", "azure_openai"]
    
    @validator("LLM_FALLBACK_PROVIDERS", pre=True)
    def assemble_fallback_providers(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = None
    OPENAI_MODEL: str = "gpt-4.1"
    OPENAI_TEMPERATURE: float = 0.1
    OPENAI_MAX_TOKENS: int = 4096
    OPENAI_ORGANIZATION: Optional[str] = None
    
    # Azure OpenAI Configuration
    AZURE_OPENAI_API_KEY: Optional[str] = None
    AZURE_OPENAI_ENDPOINT: Optional[str] = None
    AZURE_OPENAI_API_VERSION: str = "2024-02-15-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4.1"  # This should be your actual deployment name
    AZURE_OPENAI_MODEL: str = "gpt-4.1"  # Valid Azure OpenAI model name
    
    # Anthropic Configuration
    ANTHROPIC_API_KEY: Optional[str] = None
    ANTHROPIC_MODEL: str = "claude-3-sonnet-20240229"
    ANTHROPIC_TEMPERATURE: float = 0.1
    ANTHROPIC_MAX_TOKENS: int = 4096
    
    # Google AI Configuration
    GOOGLE_API_KEY: Optional[str] = None
    GOOGLE_MODEL: str = "gemini-1.5-pro"
    GOOGLE_TEMPERATURE: float = 0.1
    GOOGLE_MAX_TOKENS: int = 4096
    
    # Cohere Configuration
    COHERE_API_KEY: Optional[str] = None
    COHERE_MODEL: str = "command-r-plus"
    COHERE_TEMPERATURE: float = 0.1
    COHERE_MAX_TOKENS: int = 4096
    
    # Together AI Configuration
    TOGETHER_API_KEY: Optional[str] = None
    TOGETHER_MODEL: str = "meta-llama/Llama-2-70b-chat-hf"
    TOGETHER_TEMPERATURE: float = 0.1
    TOGETHER_MAX_TOKENS: int = 4096
    
    # Hugging Face Configuration
    HUGGINGFACE_API_KEY: Optional[str] = None
    HUGGINGFACE_MODEL: str = "microsoft/DialoGPT-large"
    HUGGINGFACE_TEMPERATURE: float = 0.1
    HUGGINGFACE_MAX_TOKENS: int = 4096
    
    # Ollama Configuration (Local)
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "llama3:8b"
    OLLAMA_TEMPERATURE: float = 0.1
    OLLAMA_MAX_TOKENS: int = 4096
    
    # ===========================================
    # EMBEDDING PROVIDER CONFIGURATION
    # ===========================================
    EMBEDDING_PROVIDER: Literal[
        "openai", "azure_openai", "huggingface", "sentence_transformers"
    ] = "openai"
    EMBEDDING_MODEL: str = "text-embedding-3-large"
    EMBEDDING_DIMENSIONS: int = 3072
    
    # Azure OpenAI Embeddings
    AZURE_EMBEDDING_DEPLOYMENT_NAME: str = "text-embedding-3-large-deployment"
    AZURE_EMBEDDING_MODEL: str = "text-embedding-3-large"
    
    # Hugging Face Embeddings
    HUGGINGFACE_EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # ===========================================
    # VECTOR DATABASE CONFIGURATION
    # ===========================================
    VECTOR_DB_PROVIDER: Literal["chroma", "pinecone", "qdrant", "weaviate"] = "qdrant"
    
    # ChromaDB Configuration
    CHROMA_HOST: str = "localhost"
    CHROMA_PORT: int = 8000
    CHROMA_COLLECTION_RESUMES: str = "resume_embeddings"
    CHROMA_COLLECTION_JOBS: str = "job_embeddings"
    
    # Pinecone Configuration
    PINECONE_API_KEY: Optional[str] = None
    PINECONE_ENVIRONMENT: Optional[str] = None
    PINECONE_INDEX_NAME: str = "ai-recruit-vectors"
    
    # Qdrant Configuration
    QDRANT_URL: str = "http://localhost:6333"
    QDRANT_API_KEY: Optional[str] = None
    QDRANT_COLLECTION_NAME: str = "ai_recruit_vectors"
    
    # Weaviate Configuration
    WEAVIATE_URL: str = "http://localhost:8080"
    WEAVIATE_API_KEY: Optional[str] = None
    WEAVIATE_CLASS_NAME: str = "AiRecruitDocuments"
    
    # ===========================================
    # AGENT ORCHESTRATION
    # ===========================================
    LANGCHAIN_API_KEY: Optional[str] = None
    LANGCHAIN_PROJECT: str = "ai-recruit-agents"
    LANGCHAIN_TRACING_V2: bool = False
    LANGSMITH_API_KEY: Optional[str] = None
    LANGFUSE_SECRET_KEY: Optional[str] = None
    LANGFUSE_PUBLIC_KEY: Optional[str] = None
    LANGFUSE_HOST: str = "https://cloud.langfuse.com"
    
    # Agent Configuration
    MAX_AGENT_RETRIES: int = 3
    AGENT_TIMEOUT_SECONDS: int = 30
    AGENT_CONCURRENCY_LIMIT: int = 10
    ENABLE_AGENT_MEMORY: bool = True
    AGENT_MEMORY_TTL_HOURS: int = 24
    
    # ===========================================
    # CREW AI CONFIGURATION
    # ===========================================
    CREW_AI_MODEL: str = "gpt-4.1"
    CREW_AI_TEMPERATURE: float = 0.1
    CREW_AI_MAX_ITERATIONS: int = 5
    CREW_AI_MEMORY_ENABLED: bool = True
    CREW_AI_VERBOSE: bool = True
    
    # ===========================================
    # MONITORING & OBSERVABILITY
    # ===========================================
    WANDB_API_KEY: Optional[str] = None
    WANDB_PROJECT: str = "ai-recruit"
    WANDB_ENTITY: Optional[str] = None
    
    PROMETHEUS_PORT: int = 9090
    PROMETHEUS_ENABLED: bool = True
    
    APPLICATIONINSIGHTS_CONNECTION_STRING: Optional[str] = None
    
    # ===========================================
    # EMAIL CONFIGURATION
    # ===========================================
    SMTP_HOST: str = "smtp.gmail.com"
    SMTP_PORT: int = 587
    SMTP_USER: Optional[str] = None
    SMTP_PASSWORD: Optional[str] = None
    EMAIL_FROM: str = "noreply@airecruit.com"
    EMAIL_FROM_NAME: str = "AI Recruit Platform"
    
    # ===========================================
    # FILE PROCESSING
    # ===========================================
    MAX_FILE_SIZE_MB: int = 50
    ALLOWED_FILE_TYPES: List[str] = ["pdf", "doc", "docx", "txt"]
    TEMP_UPLOAD_DIR: str = "/tmp/ai-recruit/uploads"
    PROCESSING_TIMEOUT_SECONDS: int = 600
    
    @validator("ALLOWED_FILE_TYPES", pre=True)
    def assemble_file_types(cls, v):
        if isinstance(v, str):
            return json.loads(v)
        return v
    
    # ===========================================
    # RATE LIMITING
    # ===========================================
    RATE_LIMIT_REQUESTS_PER_MINUTE: int = 60
    RATE_LIMIT_BURST: int = 10
    RATE_LIMIT_ENABLED: bool = True
    
    # ===========================================
    # FRONTEND CONFIGURATION
    # ===========================================
    FRONTEND_URL: str = "http://localhost:3000"
    FRONTEND_BUILD_PATH: str = "../frontend/build"
    STATIC_FILES_PATH: str = "/static"
    
    # ===========================================
    # DEVELOPMENT & TESTING
    # ===========================================
    TEST_DATABASE_URL: Optional[str] = None
    PYTEST_WORKERS: int = 4
    MOCK_LLM_RESPONSES: bool = False
    ENABLE_API_DOCS: bool = True
    
    # ===========================================
    # PRODUCTION SPECIFIC
    # ===========================================
    SENTRY_DSN: Optional[str] = None
    HTTPS_ONLY: bool = False
    SECURE_COOKIES: bool = False
    SESSION_TIMEOUT_MINUTES: int = 480
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
    def get_llm_config(self) -> Dict[str, Any]:
        """Get configuration for the selected LLM provider."""
        provider_configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": self.OPENAI_MODEL,
                "temperature": self.OPENAI_TEMPERATURE,
                "max_tokens": self.OPENAI_MAX_TOKENS,
                "organization": self.OPENAI_ORGANIZATION,
            },
            "azure_openai": {
                "api_key": self.AZURE_OPENAI_API_KEY,
                "endpoint": self.AZURE_OPENAI_ENDPOINT,
                "api_version": self.AZURE_OPENAI_API_VERSION,
                "deployment_name": self.AZURE_OPENAI_DEPLOYMENT_NAME,
                "model": self.AZURE_OPENAI_MODEL,
                "temperature": self.OPENAI_TEMPERATURE,
                "max_tokens": self.OPENAI_MAX_TOKENS,
            },
            "anthropic": {
                "api_key": self.ANTHROPIC_API_KEY,
                "model": self.ANTHROPIC_MODEL,
                "temperature": self.ANTHROPIC_TEMPERATURE,
                "max_tokens": self.ANTHROPIC_MAX_TOKENS,
            },
            "google": {
                "api_key": self.GOOGLE_API_KEY,
                "model": self.GOOGLE_MODEL,
                "temperature": self.GOOGLE_TEMPERATURE,
                "max_tokens": self.GOOGLE_MAX_TOKENS,
            },
            "cohere": {
                "api_key": self.COHERE_API_KEY,
                "model": self.COHERE_MODEL,
                "temperature": self.COHERE_TEMPERATURE,
                "max_tokens": self.COHERE_MAX_TOKENS,
            },
            "together": {
                "api_key": self.TOGETHER_API_KEY,
                "model": self.TOGETHER_MODEL,
                "temperature": self.TOGETHER_TEMPERATURE,
                "max_tokens": self.TOGETHER_MAX_TOKENS,
            },
            "huggingface": {
                "api_key": self.HUGGINGFACE_API_KEY,
                "model": self.HUGGINGFACE_MODEL,
                "temperature": self.HUGGINGFACE_TEMPERATURE,
                "max_tokens": self.HUGGINGFACE_MAX_TOKENS,
            },
            "ollama": {
                "base_url": self.OLLAMA_BASE_URL,
                "model": self.OLLAMA_MODEL,
                "temperature": self.OLLAMA_TEMPERATURE,
                "max_tokens": self.OLLAMA_MAX_TOKENS,
            },
        }
        
        return {
            "provider": self.LLM_PROVIDER,
            "config": provider_configs.get(self.LLM_PROVIDER, {}),
            "fallback_providers": self.LLM_FALLBACK_PROVIDERS,
        }
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get configuration for the selected embedding provider."""
        provider_configs = {
            "openai": {
                "api_key": self.OPENAI_API_KEY,
                "model": self.EMBEDDING_MODEL,
                "dimensions": self.EMBEDDING_DIMENSIONS,
            },
            "azure_openai": {
                "api_key": self.AZURE_OPENAI_API_KEY,
                "endpoint": self.AZURE_OPENAI_ENDPOINT,
                "api_version": self.AZURE_OPENAI_API_VERSION,
                "deployment_name": self.AZURE_EMBEDDING_DEPLOYMENT_NAME,
                "model": self.AZURE_EMBEDDING_MODEL,
            },
            "huggingface": {
                "api_key": self.HUGGINGFACE_API_KEY,
                "model": self.HUGGINGFACE_EMBEDDING_MODEL,
            },
            "sentence_transformers": {
                "model": self.HUGGINGFACE_EMBEDDING_MODEL,
            },
        }
        
        return {
            "provider": self.EMBEDDING_PROVIDER,
            "config": provider_configs.get(self.EMBEDDING_PROVIDER, {}),
        }
    
    def get_vector_db_config(self) -> Dict[str, Any]:
        """Get configuration for the selected vector database."""
        provider_configs = {
            "chroma": {
                "host": self.CHROMA_HOST,
                "port": self.CHROMA_PORT,
                "collection_resumes": self.CHROMA_COLLECTION_RESUMES,
                "collection_jobs": self.CHROMA_COLLECTION_JOBS,
            },
            "pinecone": {
                "api_key": self.PINECONE_API_KEY,
                "environment": self.PINECONE_ENVIRONMENT,
                "index_name": self.PINECONE_INDEX_NAME,
            },
            "qdrant": {
                "url": self.QDRANT_URL,
                "api_key": self.QDRANT_API_KEY,
                "collection_name": self.QDRANT_COLLECTION_NAME,
            },
            "weaviate": {
                "url": self.WEAVIATE_URL,
                "api_key": self.WEAVIATE_API_KEY,
                "class_name": self.WEAVIATE_CLASS_NAME,
            },
        }
        
        return {
            "provider": self.VECTOR_DB_PROVIDER,
            "config": provider_configs.get(self.VECTOR_DB_PROVIDER, {}),
        }
    
    def get_azure_storage_config(self) -> Dict[str, Any]:
        """Get Azure storage configuration."""
        return {
            "account_name": self.AZURE_STORAGE_ACCOUNT_NAME,
            "account_key": self.AZURE_STORAGE_ACCOUNT_KEY,
            "connection_string": self.AZURE_STORAGE_CONNECTION_STRING,
            "containers": {
                "resumes": self.AZURE_BLOB_CONTAINER_RESUMES,
                "documents": self.AZURE_BLOB_CONTAINER_DOCUMENTS,
                "processed": self.AZURE_BLOB_CONTAINER_PROCESSED,
                "formatted-resumes": self.AZURE_BLOB_CONTAINER_FORMATTED_RESUMES,
            },
            "url_expiry_hours": self.AZURE_STORAGE_URL_EXPIRY_HOURS,
        }
    
    @staticmethod
    def get_current_timestamp() -> str:
        """Get current timestamp in ISO format."""
        return datetime.now(timezone.utc).isoformat()


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()