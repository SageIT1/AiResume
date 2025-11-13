"""
Vector store abstraction and Qdrant implementation for persistent resume embeddings.

Design:
- Qdrant collections keyed by settings.QDRANT_COLLECTION_NAME (default: ai_recruit_vectors)
- Payload schema stores resume metadata for filtering and hydration
- Embeddings from Settings.get_embedding_config() via LangChain embeddings
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel
from qdrant_client.http.models import PointStruct, PointIdsList

from core.config import Settings

logger = logging.getLogger(__name__)


class VectorSearchResult(BaseModel):
    id: str
    score: float
    payload: Dict[str, Any]


class VectorStore:
    """Abstract interface for vector store operations."""

    async def initialize(self) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    async def upsert_resume_embedding(
        self,
        resume_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    async def delete_resume_embedding(self, resume_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search_resumes(
        self, query_text: str, top_k: int = 20, org_id: Optional[str] = None
    ) -> List[VectorSearchResult]:  # pragma: no cover
        raise NotImplementedError

    async def upsert_job_embedding(
        self,
        job_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:  # pragma: no cover - interface only
        raise NotImplementedError

    async def delete_job_embedding(self, job_id: str) -> None:  # pragma: no cover
        raise NotImplementedError

    async def search_jobs(
        self, query_text: str, top_k: int = 20, org_id: Optional[str] = None
    ) -> List[VectorSearchResult]:  # pragma: no cover
        raise NotImplementedError


class QdrantVectorStore(VectorStore):
    """Qdrant-backed vector store for resumes."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = None
        self.collection_name = settings.QDRANT_COLLECTION_NAME
        self.embeddings = None
        self.dimensions = settings.EMBEDDING_DIMENSIONS

    async def initialize(self) -> None:
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.http.models import Distance, VectorParams
            from langchain_openai import AzureOpenAIEmbeddings, OpenAIEmbeddings

            # Initialize Qdrant client
            self.client = QdrantClient(
                url=self.settings.QDRANT_URL,
                api_key=self.settings.QDRANT_API_KEY,
                timeout=30,
            )

            # Ensure collection exists
            existing = [c.name for c in self.client.get_collections().collections]
            if self.collection_name not in existing:
                logger.info(
                    f"Creating Qdrant collection '{self.collection_name}' with dim={self.dimensions}"
                )
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.dimensions,
                        distance=Distance.COSINE,
                    ),
                )

            # Ensure payload indexes for frequently filtered fields
            try:
                from qdrant_client.http import models as rest

                # 'type' is used in every search filter
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="type",
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )
                logger.info("✅ Created payload index on key 'type'")
            except Exception as e:
                # Index may already exist or cloud constraints may differ
                logger.debug(f"Payload index for 'type' may already exist or creation skipped: {e}")

            try:
                from qdrant_client.http import models as rest

                # 'organization_id' is commonly used to partition tenant data
                self.client.create_payload_index(
                    collection_name=self.collection_name,
                    field_name="organization_id",
                    field_schema=rest.PayloadSchemaType.KEYWORD,
                )
                logger.info("✅ Created payload index on key 'organization_id'")
            except Exception as e:
                logger.debug(
                    f"Payload index for 'organization_id' may already exist or creation skipped: {e}"
                )

            # Initialize embeddings
            emb_cfg = self.settings.get_embedding_config()
            provider = emb_cfg["provider"]
            cfg = emb_cfg["config"]
            api_ver = cfg.get("api_version", "2024-02-15-preview")
            logger.info(
                f"🔧 Embedding config: provider={provider}, deployment_name={cfg.get('deployment_name')}, endpoint={cfg.get('endpoint')}, api_version={api_ver}"
            )
            if provider == "azure_openai":
                # Some parts of the app may set OPENAI/AZURE base URL env vars which conflict
                # with AzureOpenAIEmbeddings that expects azure_endpoint instead of base_url.
                # Temporarily clear them to avoid pydantic validation errors, then restore.
                conflicting_keys = [
                    "OPENAI_API_BASE",
                    "OPENAI_BASE_URL",
                    "AZURE_OPENAI_BASE",
                    "AZURE_API_BASE",
                ]
                saved_env: Dict[str, Optional[str]] = {}
                try:
                    for key in conflicting_keys:
                        if key in os.environ:
                            saved_env[key] = os.environ.pop(key)

                    self.embeddings = AzureOpenAIEmbeddings(
                        azure_endpoint=cfg["endpoint"],
                        api_key=cfg["api_key"],
                        azure_deployment=cfg["deployment_name"],
                        api_version=cfg.get("api_version", "2024-02-15-preview"),
                    )
                finally:
                    for key, value in saved_env.items():
                        if value is not None:
                            os.environ[key] = value
            elif provider == "openai":
                self.embeddings = OpenAIEmbeddings(
                    api_key=cfg["api_key"],
                    model=cfg.get("model", "text-embedding-3-large"),
                )
            else:
                # Minimal: allow only OpenAI/Azure here to ensure dim matches config
                raise ValueError(
                    f"Embedding provider '{provider}' not supported for Qdrant vector store"
                )

            logger.info("✅ QdrantVectorStore initialized")
        except Exception as e:
            logger.error(f"❌ Failed to initialize QdrantVectorStore: {e}")
            raise

    async def _embed_text(self, text: str) -> List[float]:
        if not self.embeddings:
            raise RuntimeError("Embeddings not initialized")
        vec = await self.embeddings.aembed_query(text)
        if len(vec) != self.dimensions:
            logger.warning(
                f"Embedding size {len(vec)} != configured {self.dimensions}. Proceeding, but check EMBEDDING_DIMENSIONS."
            )
        return vec

    async def upsert_resume_embedding(
        self, resume_id: str, text: str, metadata: Dict[str, Any]
    ) -> None:
        try:
            from qdrant_client.http.models import PointStruct

            vector = await self._embed_text(text)
            payload = {
                "type": "resume",
                "resume_id": resume_id,
                **metadata,
            }
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[
                    PointStruct(
                        id=resume_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.info(f"🔎 Upserted resume embedding to Qdrant: {resume_id}")
        except Exception as e:
            logger.error(f"❌ Failed to upsert resume embedding: {e}")
            raise

    async def delete_resume_embedding(self, resume_id: str) -> None:
        try:
            from qdrant_client.http.models import PointIdsList

            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(points=[resume_id]),
                wait=True,
            )
            logger.info(f"🗑️ Deleted resume embedding from Qdrant: {resume_id}")
        except Exception as e:
            logger.error(f"❌ Failed to delete resume embedding: {e}")
            raise

    async def search_resumes(
        self, query_text: str, top_k: int = 20, org_id: Optional[str] = None
    ) -> List[VectorSearchResult]:
        try:
            vector = await self._embed_text(query_text)
            query_filter = None
            if org_id:
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue

                query_filter = Filter(
                    must=[
                        FieldCondition(
                            key="type", match=MatchValue(value="resume")
                        ),
                        FieldCondition(
                            key="organization_id", match=MatchValue(value=org_id)
                        ),
                    ]
                )
            else:
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue

                query_filter = Filter(
                    must=[FieldCondition(key="type", match=MatchValue(value="resume"))]
                )

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=0.20,  # Lower threshold for better matching
                with_payload=True,
                with_vectors=False,
            )

            results: List[VectorSearchResult] = []
            for point in search_result:
                results.append(
                    VectorSearchResult(
                        id=str(point.id),
                        score=float(point.score),
                        payload=point.payload or {},
                    )
                )
            return results
        except Exception as e:
            logger.error(f"❌ Qdrant search failed: {e}")
            return []

    async def upsert_job_embedding(
        self,
        job_id: str,
        text: str,
        metadata: Dict[str, Any],
    ) -> None:
        """Store job embedding in Qdrant."""
        try:
            vector = await self._embed_text(text)
            
            # Add job-specific metadata
            payload = {
                **metadata,
                "type": "job",
                "job_id": job_id,
            }
            
            self.client.upsert(
                collection_name=self.collection_name,
                points=[
                    PointStruct(
                        id=job_id,
                        vector=vector,
                        payload=payload,
                    )
                ],
            )
            logger.info(f"✅ Job embedding stored: {job_id}")
        except Exception as e:
            logger.error(f"❌ Failed to store job embedding: {e}")
            raise

    async def delete_job_embedding(self, job_id: str) -> None:
        """Delete job embedding from Qdrant."""
        try:
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=PointIdsList(ids=[job_id]),
            )
            logger.info(f"✅ Job embedding deleted: {job_id}")
        except Exception as e:
            logger.error(f"❌ Failed to delete job embedding: {e}")
            raise

    async def search_jobs(
        self, query_text: str, top_k: int = 20, org_id: Optional[str] = None
    ) -> List[VectorSearchResult]:
        """Search for similar jobs using vector similarity."""
        try:
            vector = await self._embed_text(query_text)
            
            # Build filter for jobs only
            if org_id:
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue
                
                query_filter = Filter(
                    must=[
                        FieldCondition(key="type", match=MatchValue(value="job")),
                        FieldCondition(
                            key="organization_id", match=MatchValue(value=org_id)
                        ),
                    ]
                )
            else:
                from qdrant_client.http.models import Filter, FieldCondition, MatchValue

                query_filter = Filter(
                    must=[FieldCondition(key="type", match=MatchValue(value="job"))]
                )

            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=vector,
                query_filter=query_filter,
                limit=top_k,
                score_threshold=0.4,  # Higher threshold for job matching
                with_payload=True,
                with_vectors=False,
            )

            results: List[VectorSearchResult] = []
            for point in search_result:
                results.append(
                    VectorSearchResult(
                        id=str(point.id),
                        score=float(point.score),
                        payload=point.payload or {},
                    )
                )
            return results
        except Exception as e:
            logger.error(f"❌ Qdrant job search failed: {e}")
            return []


