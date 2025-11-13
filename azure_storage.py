"""
AI Recruit - Azure Storage Service
Comprehensive Azure Blob Storage integration for document management.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, BinaryIO, Tuple, Any
from uuid import uuid4
import mimetypes
import os

from azure.storage.blob.aio import BlobServiceClient, BlobClient
from azure.storage.blob import generate_blob_sas, BlobSasPermissions
from azure.core.exceptions import AzureError, ResourceNotFoundError
import aiofiles

from core.config import Settings

logger = logging.getLogger(__name__)


class AzureStorageService:
    """
    Azure Blob Storage service for document management.
    Handles upload, download, and management of recruitment documents.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.config = settings.get_azure_storage_config()
        self.blob_service_client = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize Azure Storage client."""
        try:
            self.blob_service_client = BlobServiceClient(
                account_url=f"https://{self.config['account_name']}.blob.core.windows.net",
                credential=self.config['account_key']
            )
            
            # Create containers if they don't exist
            await self._ensure_containers_exist()
            self._initialized = True
            logger.info("Azure Storage service initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Azure Storage: {str(e)}")
            raise
    
    async def _ensure_containers_exist(self):
        """Ensure all required containers exist."""
        containers = self.config['containers']
        
        for container_name in containers.values():
            try:
                container_client = self.blob_service_client.get_container_client(container_name)
                await container_client.create_container()
                logger.info(f"Created container: {container_name}")
            except Exception as e:
                if "ContainerAlreadyExists" in str(e):
                    logger.debug(f"Container already exists: {container_name}")
                else:
                    logger.error(f"Error creating container {container_name}: {str(e)}")
                    raise
    
    async def upload_resume(
        self,
        file_content: bytes,
        filename: str,
        user_id: str,
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Upload resume file to Azure Storage.
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            user_id: ID of the user uploading
            metadata: Additional metadata
            
        Returns:
            Dict containing upload information
        """
        if not self._initialized:
            await self.initialize()
        try:
            # Generate unique blob name
            file_ext = os.path.splitext(filename)[1]
            #blob_name = f"{user_id}/{uuid4()}{file_ext}"
            blob_name = f"{user_id}/{filename}"
            # Get container client
            container_name = self.config['containers']['resumes']
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            # Prepare metadata
            blob_metadata = {
                "original_filename": filename,
                "user_id": user_id,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
                "file_size": str(len(file_content)),
            }
            if metadata:
                # Ensure all metadata values are strings
                for key, value in metadata.items():
                    blob_metadata[key] = str(value) if value is not None else ""
            # Upload file
            await blob_client.upload_blob(
                file_content,
                metadata=blob_metadata,
                overwrite=True,
                content_type=blob_metadata["content_type"]
            )
            # Generate signed URL
            blob_url = await self._generate_signed_url(container_name, blob_name)
            logger.info(f"Successfully uploaded resume: {blob_name}")
            return {
                "blob_name": blob_name,
                "container": container_name,
                "url": blob_url,
                "original_filename": filename,
                "size": len(file_content),
                "content_type": blob_metadata["content_type"],
                "upload_timestamp": blob_metadata["upload_timestamp"]
            }  
        except Exception as e:
            logger.error(f"Failed to upload resume: {str(e)}")
            raise
    
    async def upload_document(
        self,
        file_content: bytes,
        filename: str,
        document_type: str = "general",
        metadata: Optional[Dict] = None
    ) -> Dict[str, str]:
        """
        Upload general document to Azure Storage.
        
        Args:
            file_content: Binary content of the file
            filename: Original filename
            document_type: Type of document (job_description, template, etc.)
            metadata: Additional metadata
            
        Returns:
            Dict containing upload information
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Generate unique blob name
            file_ext = os.path.splitext(filename)[1]
            blob_name = f"{document_type}/{uuid4()}{file_ext}"
            
            # Get container client
            container_name = self.config['containers']['documents']
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            # Prepare metadata
            blob_metadata = {
                "original_filename": filename,
                "document_type": document_type,
                "upload_timestamp": datetime.now(timezone.utc).isoformat(),
                "content_type": mimetypes.guess_type(filename)[0] or "application/octet-stream",
                "file_size": str(len(file_content)),
            }
            
            if metadata:
                # Ensure all metadata values are strings
                for key, value in metadata.items():
                    blob_metadata[key] = str(value) if value is not None else ""
            
            # Upload file
            await blob_client.upload_blob(
                file_content,
                metadata=blob_metadata,
                overwrite=True,
                content_type=blob_metadata["content_type"]
            )
            
            # Generate signed URL
            blob_url = await self._generate_signed_url(container_name, blob_name)
            
            logger.info(f"Successfully uploaded document: {blob_name}")
            
            return {
                "blob_name": blob_name,
                "container": container_name,
                "url": blob_url,
                "original_filename": filename,
                "document_type": document_type,
                "size": len(file_content),
                "content_type": blob_metadata["content_type"],
                "upload_timestamp": blob_metadata["upload_timestamp"]
            }
            
        except Exception as e:
            logger.error(f"Failed to upload document: {str(e)}")
            raise
    
    async def download_file(self, container_name: str, blob_name: str) -> bytes:
        """
        Download file from Azure Storage.
        
        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            
        Returns:
            Binary content of the file
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            download_stream = await blob_client.download_blob()
            content = await download_stream.readall()
            
            logger.info(f"Successfully downloaded file: {blob_name}")
            return content
            
        except ResourceNotFoundError:
            logger.error(f"File not found: {blob_name}")
            raise
        except Exception as e:
            logger.error(f"Failed to download file: {str(e)}")
            raise
    
    async def download_resume(self, blob_name: str, container_name: str = None) -> bytes:
        """
        Download resume file from Azure Storage.
        
        Args:
            blob_name: Name of the blob (required)
            container_name: Name of the container (optional, defaults to resumes container)
            
        Returns:
            Binary content of the resume file
        """
        if not blob_name or not blob_name.strip():
            raise ValueError("Please specify a container name and blob name.")
        
        # Use default resumes container if not specified
        if not container_name:
            container_name = self.config['containers']['resumes']
        
        if not container_name or not container_name.strip():
            raise ValueError("Please specify a container name and blob name.")
        
        logger.info(f"🔄 Downloading resume: {blob_name} from container: {container_name}")
        
        try:
            content = await self.download_file(container_name, blob_name)
            logger.info(f"✅ Successfully downloaded resume: {blob_name}")
            return content
            
        except ResourceNotFoundError:
            logger.error(f"❌ Resume not found: {blob_name} in container: {container_name}")
            raise
        except Exception as e:
            logger.error(f"❌ Failed to download resume: {str(e)}")
            raise
    
    async def delete_file(self, container_name: str, blob_name: str) -> bool:
        """
        Delete file from Azure Storage.
        
        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            
        Returns:
            True if deleted successfully
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            await blob_client.delete_blob()
            logger.info(f"Successfully deleted file: {blob_name}")
            return True
            
        except ResourceNotFoundError:
            logger.warning(f"File not found for deletion: {blob_name}")
            return False
        except Exception as e:
            logger.error(f"Failed to delete file: {str(e)}")
            raise
    
    async def get_file_metadata(self, container_name: str, blob_name: str) -> Dict:
        """
        Get file metadata from Azure Storage.
        
        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            
        Returns:
            Dictionary containing file metadata
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            blob_client = self.blob_service_client.get_blob_client(
                container=container_name,
                blob=blob_name
            )
            
            properties = await blob_client.get_blob_properties()
            
            return {
                "name": blob_name,
                "container": container_name,
                "size": properties.size,
                "content_type": properties.content_settings.content_type,
                "last_modified": properties.last_modified.isoformat(),
                "etag": properties.etag,
                "metadata": properties.metadata or {}
            }
            
        except Exception as e:
            logger.error(f"Failed to get file metadata: {str(e)}")
            raise
    
    async def list_files(
        self,
        container_name: str,
        prefix: Optional[str] = None,
        limit: Optional[int] = None
    ) -> List[Dict]:
        """
        List files in a container.
        
        Args:
            container_name: Name of the container
            prefix: Prefix to filter files
            limit: Maximum number of files to return
            
        Returns:
            List of file information dictionaries
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            container_client = self.blob_service_client.get_container_client(container_name)
            
            files = []
            async for blob in container_client.list_blobs(name_starts_with=prefix):
                if limit and len(files) >= limit:
                    break
                
                files.append({
                    "name": blob.name,
                    "size": blob.size,
                    "content_type": blob.content_settings.content_type if blob.content_settings else None,
                    "last_modified": blob.last_modified.isoformat() if blob.last_modified else None,
                    "metadata": blob.metadata or {}
                })
            
            logger.info(f"Listed {len(files)} files from container: {container_name}")
            return files
            
        except Exception as e:
            logger.error(f"Failed to list files: {str(e)}")
            raise
    
    async def _generate_signed_url(
        self,
        container_name: str,
        blob_name: str,
        permissions: str = "r",
        expiry_hours: Optional[int] = None
    ) -> str:
        """
        Generate signed URL for blob access.
        
        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            permissions: SAS permissions (r=read, w=write, d=delete)
            expiry_hours: Hours until expiry
            
        Returns:
            Signed URL string
        """
        try:
            expiry_hours = expiry_hours or self.config['url_expiry_hours']
            expiry_time = datetime.now(timezone.utc) + timedelta(hours=expiry_hours)
            
            # Generate SAS token
            sas_token = generate_blob_sas(
                account_name=self.config['account_name'],
                container_name=container_name,
                blob_name=blob_name,
                account_key=self.config['account_key'],
                permission=BlobSasPermissions(read="r" in permissions),
                expiry=expiry_time
            )
            
            # Construct URL
            blob_url = f"https://{self.config['account_name']}.blob.core.windows.net/{container_name}/{blob_name}?{sas_token}"
            
            return blob_url
            
        except Exception as e:
            logger.error(f"Failed to generate signed URL: {str(e)}")
            raise
    
    async def move_to_processed(
        self,
        source_container: str,
        source_blob: str,
        processed_metadata: Optional[Dict] = None
    ) -> str:
        """
        Move file to processed container after AI processing.
        
        Args:
            source_container: Source container name
            source_blob: Source blob name
            processed_metadata: Additional metadata from processing
            
        Returns:
            New blob name in processed container
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Download source file
            source_content = await self.download_file(source_container, source_blob)
            source_metadata = await self.get_file_metadata(source_container, source_blob)
            
            # Generate new blob name for processed container
            file_ext = os.path.splitext(source_blob)[1]
            new_blob_name = f"processed/{uuid4()}{file_ext}"
            
            # Prepare enhanced metadata
            enhanced_metadata = source_metadata.get("metadata", {})
            enhanced_metadata.update({
                "processed_timestamp": datetime.now(timezone.utc).isoformat(),
                "source_container": source_container,
                "source_blob": source_blob,
                "processing_status": "completed"
            })
            
            if processed_metadata:
                enhanced_metadata.update(processed_metadata)
            
            # Upload to processed container
            processed_container = self.config['containers']['processed']
            blob_client = self.blob_service_client.get_blob_client(
                container=processed_container,
                blob=new_blob_name
            )
            
            await blob_client.upload_blob(
                source_content,
                metadata=enhanced_metadata,
                overwrite=True,
                content_type=source_metadata.get("content_type", "application/octet-stream")
            )
            
            logger.info(f"Successfully moved file to processed: {new_blob_name}")
            
            return new_blob_name
            
        except Exception as e:
            logger.error(f"Failed to move file to processed: {str(e)}")
            raise
    
    async def cleanup_temp_files(self, max_age_hours: int = 24):
        """
        Cleanup temporary files older than specified age.
        
        Args:
            max_age_hours: Maximum age in hours for temp files
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            cutoff_time = datetime.now(timezone.utc) - timedelta(hours=max_age_hours)
            deleted_count = 0
            
            # Check all containers for old temp files
            for container_name in self.config['containers'].values():
                temp_files = await self.list_files(container_name, prefix="temp/")
                
                for file_info in temp_files:
                    if file_info.get("last_modified"):
                        last_modified = datetime.fromisoformat(file_info["last_modified"].replace("Z", "+00:00"))
                        
                        if last_modified < cutoff_time:
                            await self.delete_file(container_name, file_info["name"])
                            deleted_count += 1
            
            logger.info(f"Cleaned up {deleted_count} temporary files")
            
        except Exception as e:
            logger.error(f"Failed to cleanup temp files: {str(e)}")
            raise
    
    async def get_storage_stats(self) -> Dict[str, Any]:
        """
        Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            stats = {}
            
            for container_type, container_name in self.config['containers'].items():
                files = await self.list_files(container_name)
                total_size = sum(file_info.get("size", 0) for file_info in files)
                
                stats[container_type] = {
                    "file_count": len(files),
                    "total_size_bytes": total_size,
                    "total_size_mb": round(total_size / (1024 * 1024), 2)
                }
            
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get storage stats: {str(e)}")
            raise
    
    async def close(self):
        """Close Azure Storage client."""
        if self.blob_service_client:
            await self.blob_service_client.close()
            logger.info("Azure Storage client closed")