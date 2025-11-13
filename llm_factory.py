"""
AI Recruit - LLM Provider Factory
Dynamic LLM provider switching with fallback support.

NO MANUAL RULES - NO FALLBACKS - PURE AI INTELLIGENCE
"""

from typing import Dict, Any, Optional, List
import logging
import time
import random
from abc import ABC, abstractmethod

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatCohere, ChatOllama
from langchain_together import ChatTogether
from langchain_huggingface import ChatHuggingFace
from langchain_core.language_models import BaseChatModel

from core.config import Settings

logger = logging.getLogger(__name__)


def retry_llm_call(func, max_retries=3, base_delay=1.0, max_delay=60.0):
    """Retry LLM calls with exponential backoff for rate limiting."""
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            error_msg = str(e).lower()
            if "429" in error_msg or "rate limit" in error_msg or "too many requests" in error_msg:
                if attempt < max_retries - 1:
                    delay = min(base_delay * (2 ** attempt) + random.uniform(0, 1), max_delay)
                    logger.warning(f"Rate limit hit, retrying in {delay:.2f}s (attempt {attempt + 1}/{max_retries})")
                    time.sleep(delay)
                    continue
            elif "404" in error_msg or "not found" in error_msg:
                logger.error(f"Resource not found (404): {e}")
                raise
            else:
                logger.error(f"LLM call failed: {e}")
                raise
    
    raise Exception(f"LLM call failed after {max_retries} attempts")


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""
    
    @abstractmethod
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create a chat model instance."""
        pass
    
    @abstractmethod
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate provider configuration."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create OpenAI chat model."""
        return ChatOpenAI(
            api_key=config.get("api_key"),
            model=config.get("model", "gpt-4.1"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
            organization=config.get("organization"),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate OpenAI configuration."""
        return config.get("api_key") is not None


class AzureOpenAIProvider(LLMProvider):
    """Azure OpenAI LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create Azure OpenAI chat model."""
        deployment_name = config.get("deployment_name", "gpt-4.1")
        model_name = config.get("model", "gpt-4.1")
        
        logger.info(f"Creating Azure OpenAI chat model - deployment: {deployment_name}, model: {model_name}")
        
        return AzureChatOpenAI(
            api_key=config.get("api_key"),
            azure_endpoint=config.get("endpoint"),
            api_version=config.get("api_version", "2024-02-15-preview"),
            deployment_name=deployment_name,
            model=model_name,  # Use model name directly, not "azure/" prefix
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Azure OpenAI configuration."""
        required_fields = ["api_key", "endpoint", "deployment_name"]
        return all(config.get(field) for field in required_fields)


class AnthropicProvider(LLMProvider):
    """Anthropic Claude LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create Anthropic chat model."""
        return ChatAnthropic(
            api_key=config.get("api_key"),
            model=config.get("model", "claude-3-sonnet-20240229"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Anthropic configuration."""
        return config.get("api_key") is not None


class GoogleProvider(LLMProvider):
    """Google AI LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create Google AI chat model."""
        return ChatGoogleGenerativeAI(
            google_api_key=config.get("api_key"),
            model=config.get("model", "gemini-1.5-pro"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Google AI configuration."""
        return config.get("api_key") is not None


class CohereProvider(LLMProvider):
    """Cohere LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create Cohere chat model."""
        return ChatCohere(
            cohere_api_key=config.get("api_key"),
            model=config.get("model", "command-r-plus"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Cohere configuration."""
        return config.get("api_key") is not None


class TogetherProvider(LLMProvider):
    """Together AI LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create Together AI chat model."""
        return ChatTogether(
            api_key=config.get("api_key"),
            model=config.get("model", "meta-llama/Llama-2-70b-chat-hf"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Together AI configuration."""
        return config.get("api_key") is not None


class HuggingFaceProvider(LLMProvider):
    """Hugging Face LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create Hugging Face chat model."""
        return ChatHuggingFace(
            api_key=config.get("api_key"),
            model=config.get("model", "microsoft/DialoGPT-large"),
            temperature=config.get("temperature", 0.1),
            max_tokens=config.get("max_tokens", 4096),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Hugging Face configuration."""
        return config.get("api_key") is not None


class OllamaProvider(LLMProvider):
    """Ollama Local LLM Provider."""
    
    def create_chat_model(self, config: Dict[str, Any]) -> BaseChatModel:
        """Create Ollama chat model."""
        return ChatOllama(
            base_url=config.get("base_url", "http://localhost:11434"),
            model=config.get("model", "llama3:8b"),
            temperature=config.get("temperature", 0.1),
            num_predict=config.get("max_tokens", 4096),
        )
    
    def validate_config(self, config: Dict[str, Any]) -> bool:
        """Validate Ollama configuration."""
        return config.get("base_url") is not None


class LLMFactory:
    """
    Factory class for creating LLM providers with dynamic switching.
    Supports multiple providers with automatic fallback.
    """
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.providers = {
            "openai": OpenAIProvider(),
            "azure_openai": AzureOpenAIProvider(),
            "anthropic": AnthropicProvider(),
            "google": GoogleProvider(),
            "cohere": CohereProvider(),
            "together": TogetherProvider(),
            "huggingface": HuggingFaceProvider(),
            "ollama": OllamaProvider(),
        }
        self._cache: Dict[str, BaseChatModel] = {}
    
    def create_llm(self, provider_name: Optional[str] = None) -> BaseChatModel:
        """
        Create LLM instance with optional provider override.
        
        Args:
            provider_name: Override the default provider
            
        Returns:
            BaseChatModel: Configured LLM instance
            
        Raises:
            ValueError: If provider is invalid or configuration is missing
        """
        provider_name = provider_name or self.settings.LLM_PROVIDER
        
        # Check cache first
        if provider_name in self._cache:
            return self._cache[provider_name]
        
        # Get provider
        provider = self.providers.get(provider_name)
        if not provider:
            raise ValueError(f"Unsupported LLM provider: {provider_name}")
        
        # Get configuration
        llm_config = self.settings.get_llm_config()
        config = llm_config["config"]
        
        # Validate configuration
        if not provider.validate_config(config):
            raise ValueError(f"Invalid configuration for provider: {provider_name}")
        
        # Create model
        try:
            model = provider.create_chat_model(config)
            self._cache[provider_name] = model
            logger.info(f"Successfully created LLM: {provider_name}")
            return model
        except Exception as e:
            logger.error(f"Failed to create LLM {provider_name}: {str(e)}")
            raise
    
    def create_llm_with_fallback(self) -> BaseChatModel:
        """
        Create LLM with automatic fallback to alternative providers.
        
        Returns:
            BaseChatModel: Configured LLM instance
            
        Raises:
            RuntimeError: If all providers fail
        """
        # Try primary provider first
        try:
            return self.create_llm()
        except Exception as e:
            logger.warning(f"Primary LLM provider failed: {str(e)}")
        
        # Try fallback providers
        fallback_providers = self.settings.LLM_FALLBACK_PROVIDERS
        for fallback_provider in fallback_providers:
            try:
                logger.info(f"Attempting fallback provider: {fallback_provider}")
                return self.create_llm(fallback_provider)
            except Exception as e:
                logger.warning(f"Fallback provider {fallback_provider} failed: {str(e)}")
                continue
        
        raise RuntimeError("All LLM providers failed. Check your configuration.")
    
    def get_available_providers(self) -> List[str]:
        """Get list of available LLM providers."""
        return list(self.providers.keys())
    
    def validate_provider_config(self, provider_name: str) -> bool:
        """
        Validate configuration for a specific provider.
        
        Args:
            provider_name: Name of the provider to validate
            
        Returns:
            bool: True if configuration is valid
        """
        provider = self.providers.get(provider_name)
        if not provider:
            return False
        
        llm_config = self.settings.get_llm_config()
        config = llm_config["config"]
        
        return provider.validate_config(config)
    
    def clear_cache(self):
        """Clear the LLM cache."""
        self._cache.clear()
        logger.info("LLM cache cleared")
    
    def get_provider_info(self) -> Dict[str, Any]:
        """Get information about current provider configuration."""
        llm_config = self.settings.get_llm_config()
        
        return {
            "current_provider": llm_config["provider"],
            "fallback_providers": llm_config["fallback_providers"],
            "available_providers": self.get_available_providers(),
            "provider_status": {
                provider: self.validate_provider_config(provider)
                for provider in self.get_available_providers()
            }
        }