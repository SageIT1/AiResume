"""
Rate Limiting Configuration for AI Recruit
Handles Azure OpenAI rate limiting and retry logic
"""

import asyncio
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class RateLimitConfig:
    """Configuration for rate limiting and retries."""
    max_retries: int = 3
    base_delay: int = 5  # Base delay in seconds
    max_delay: int = 300  # Maximum delay in seconds (5 minutes)
    exponential_base: int = 2  # Exponential backoff base
    jitter: bool = True  # Add random jitter to prevent thundering herd

class RateLimiter:
    """Rate limiter for AI API calls."""
    
    def __init__(self, config: Optional[RateLimitConfig] = None):
        self.config = config or RateLimitConfig()
        self._call_times = []  # Track recent call times
    
    async def wait_if_needed(self, min_interval: int = 2) -> None:
        """Wait if we've made calls too recently."""
        import time
        current_time = time.time()
        
        # Remove calls older than 60 seconds
        self._call_times = [t for t in self._call_times if current_time - t < 60]
        
        if self._call_times:
            time_since_last = current_time - self._call_times[-1]
            if time_since_last < min_interval:
                wait_time = min_interval - time_since_last
                logger.info(f"⏳ Rate limiting: waiting {wait_time:.1f} seconds")
                await asyncio.sleep(wait_time)
        
        self._call_times.append(time.time())
    
    async def execute_with_retry(self, func, *args, **kwargs) -> Any:
        """Execute a function with retry logic for rate limiting."""
        last_exception = None
        
        for attempt in range(self.config.max_retries):
            try:
                # Wait before each attempt (except first)
                if attempt > 0:
                    delay = min(
                        self.config.base_delay * (self.config.exponential_base ** attempt),
                        self.config.max_delay
                    )
                    
                    # Add jitter to prevent thundering herd
                    if self.config.jitter:
                        import random
                        jitter = random.uniform(0.1, 0.5) * delay
                        delay += jitter
                    
                    logger.info(f"⏳ Retry attempt {attempt + 1}/{self.config.max_retries} after {delay:.1f}s")
                    await asyncio.sleep(delay)
                
                # Execute the function
                return await func(*args, **kwargs)
                
            except Exception as e:
                last_exception = e
                error_str = str(e).lower()
                
                # Check if it's a rate limiting error
                if any(keyword in error_str for keyword in ["429", "rate limit", "too many requests", "quota"]):
                    logger.warning(f"⚠️ Rate limit hit on attempt {attempt + 1}: {e}")
                    if attempt < self.config.max_retries - 1:
                        continue  # Try again
                    else:
                        logger.error(f"❌ Max retries reached for rate limiting")
                        raise e
                else:
                    # Not a rate limiting error, don't retry
                    logger.error(f"❌ Non-rate-limit error: {e}")
                    raise e
        
        # If we get here, all retries failed
        if last_exception:
            raise last_exception
        else:
            raise Exception("All retry attempts failed")

# Global rate limiter instance
rate_limiter = RateLimiter()

# Convenience functions
async def wait_before_ai_call() -> None:
    """Wait before making an AI call to respect rate limits."""
    await rate_limiter.wait_if_needed(min_interval=3)

async def execute_ai_call_with_retry(func, *args, **kwargs) -> Any:
    """Execute an AI call with retry logic."""
    return await rate_limiter.execute_with_retry(func, *args, **kwargs)
