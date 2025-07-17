"""
Redis caching layer for sentiment analysis results.
"""

import hashlib
import json
import logging
from datetime import datetime
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import RedisError

logger = logging.getLogger(__name__)


class SentimentCache:
    """
    Redis-based cache for sentiment analysis results.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379", ttl: int = 3600):
        """
        Args:
            ttl: Time to live in seconds
        """
        self.redis_url = redis_url
        self.ttl = ttl
        self.redis: Redis | None = None
        self.connected = False

    async def connect(self) -> bool:
        """Connect to Redis.

        Returns:
            bool: True if connection succeeded, False otherwise
        """
        try:
            logger.debug(f"Attempting to connect to Redis at: {self.redis_url}")
            self.redis = Redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_connect_timeout=5,
                socket_timeout=5,
                retry_on_error=[],
            )
            logger.debug("Redis client created successfully")
            logger.debug("About to ping Redis")
            ping_result = await self.redis.ping()
            logger.debug(f"Redis ping result: {ping_result}")
            logger.debug(f"Redis ping result type: {type(ping_result)}")

            self.connected = True
            logger.info("Successfully connected to Redis")
            return True

        except RedisError as e:
            logger.error(f"Failed to connect to Redis: {str(e)}")
            logger.error(f"Redis error type: {type(e)}")
            self.connected = False
            return False
        except Exception as e:
            logger.error(f"Unexpected error connecting to Redis: {str(e)}")
            logger.error(f"Error type: {type(e)}")
            import traceback

            logger.error(f"Traceback: {traceback.format_exc()}")
            self.connected = False
            return False

    async def disconnect(self) -> None:
        """Disconnect from Redis."""
        if self.redis and self.connected:
            try:
                await self.redis.close()
                self.connected = False
                logger.info("Disconnected from Redis")
            except RedisError as e:
                logger.error(f"Error disconnecting from Redis: {str(e)}")

    def _generate_cache_key(self, text: str) -> str:
        """Generate a consistent cache key for the given text."""
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return f"sentiment:{text_hash}"

    async def get(self, text: str) -> dict[str, Any] | None:
        """
        Get cached sentiment result for text.

        Args:
            text: Input text to check cache for

        Returns:
            Cached sentiment result or None if not found
        """
        if not self.connected or not self.redis:
            return None

        try:
            cache_key = self._generate_cache_key(text)
            cached_result = await self.redis.get(cache_key)

            if not cached_result:
                logger.debug(f"Cache miss for key: {cache_key}")
                return None

            result = json.loads(cached_result)
            logger.debug(f"Cache hit for key: {cache_key}")
            return result

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in cache for key {cache_key}: {str(e)}")
            await self.delete(text)  # Clean up invalid entry
            return None
        except RedisError as e:
            logger.error(f"Redis error getting cache: {str(e)}")
            return None

    async def set(self, text: str, result: dict[str, Any]) -> bool:
        """
        Cache sentiment result for text.

        Args:
            text: Input text
            result: Sentiment analysis result to cache

        Returns:
            True if successfully cached, False otherwise
        """
        if not self.connected or not self.redis:
            return False

        try:
            cache_key = self._generate_cache_key(text)
            cached_result = {
                **result,
                "cached_at": datetime.utcnow().isoformat(),
                "text": text,
            }

            await self.redis.setex(
                cache_key, self.ttl, json.dumps(cached_result, ensure_ascii=False)
            )
            logger.debug(f"Cached result for key: {cache_key}")
            return True

        except (RedisError, TypeError) as e:
            logger.error(f"Error setting cache: {str(e)}")
            return False

    async def get_batch(self, texts: list[str]) -> dict[str, dict[str, Any] | None]:
        """
        Get cached sentiment results for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            Dictionary mapping text to cached result (or None if not cached)
        """
        if not self.connected or not self.redis:
            return dict.fromkeys(texts)

        try:
            cache_keys = [self._generate_cache_key(text) for text in texts]
            cached_values = await self.redis.mget(cache_keys)

            results = {}
            for text, cached_value in zip(texts, cached_values, strict=False):
                if cached_value is None:
                    results[text] = None
                    continue

                try:
                    results[text] = json.loads(cached_value)
                except json.JSONDecodeError:
                    results[text] = None
                    logger.warning(f"Invalid JSON in cache for text: {text[:50]}...")

            return results

        except RedisError as e:
            logger.error(f"Redis error getting batch cache: {str(e)}")
            return dict.fromkeys(texts)

    async def set_batch(self, text_results: dict[str, dict[str, Any]]) -> int:
        """
        Cache multiple sentiment results.
        Args:
            text_results: Dictionary mapping text to sentiment result
        Returns:
            Number of successfully cached results
        """
        if not self.connected or not self.redis:
            return 0
        try:
            pipe = self.redis.pipeline()
            now = datetime.now(datetime.timezone.utc)
            for text, result in text_results.items():
                key = self._generate_cache_key(text)
                payload = {**result, "cached_at": now, "text": text}
                pipe.setex(key, self.ttl, json.dumps(payload, ensure_ascii=False))
            await pipe.execute()
            return len(text_results)
        except RedisError as e:
            logger.error(f"Redis error setting batch cache: {str(e)}")
            return 0

    async def health_check(self) -> dict[str, Any]:
        """
        Check if cache is healthy.
        Returns:
            Dictionary with health status and details
        """
        status = {"connected": False, "error": "Not initialized"}

        if not self.redis:
            return status

        try:
            await self.redis.ping()
            status.update({"connected": True, "error": None, "status": "ok"})
        except RedisError as e:
            status.update({"error": str(e), "status": "unavailable"})
            logger.error(f"Redis health check failed: {str(e)}")

        return status

    async def get_stats(self) -> dict[str, Any]:
        """
        Get cache statistics.

        Returns:
            Dictionary with cache statistics
        """
        if not self.connected or not self.redis:
            return {"connected": False}

        try:
            info = await self.redis.info("memory")
            keys = await self.redis.keys("sentiment:*")

            return {
                "connected": True,
                "keys": len(keys),
                "memory_used": info.get("used_memory", 0),
                "memory_used_human": info.get("used_memory_human", "0B"),
                "ttl_seconds": self.ttl,
            }
        except RedisError as e:
            logger.error(f"Error getting Redis stats: {str(e)}")
            return {"connected": False, "error": str(e)}

    async def delete(self, text: str) -> bool:
        """
        Delete a cached entry.

        Args:
            text: Text to delete from cache

        Returns:
            True if deleted, False otherwise
        """
        if not self.connected or not self.redis:
            return False

        try:
            cache_key = self._generate_cache_key(text)
            return bool(await self.redis.delete(cache_key))
        except RedisError as e:
            logger.error(f"Error deleting cache key: {str(e)}")
            return False

    async def clear_namespace(self, namespace: str = "sentiment:*") -> int:
        """
        Clear all keys in the given namespace.

        Args:
            namespace: Redis key pattern to clear

        Returns:
            Number of keys deleted
        """
        if not self.connected or not self.redis:
            return 0

        try:
            keys = await self.redis.keys(namespace)
            if keys:
                count = await self.redis.delete(*keys)
                logger.info(f"Cleared {count} keys from namespace {namespace}")
                return count
            return 0
        except RedisError as e:
            logger.error(f"Error clearing namespace {namespace}: {str(e)}")
            return 0
