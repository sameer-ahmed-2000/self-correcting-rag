"""
Redis query cache — optional, degrades gracefully if Redis is unavailable.
Caches identical (question, sources) pairs so the LLM is never called twice
for the same request, significantly reducing cost and latency.
"""
import os
import json
import hashlib
import logging
from typing import Optional

logger = logging.getLogger("rag.cache")

REDIS_URL = os.getenv("REDIS_URL", "")  # empty = cache disabled
CACHE_TTL = int(os.getenv("CACHE_TTL_SECONDS", "3600"))  # 1 hour default

_redis_client = None
_redis_available = False


def _get_client():
    global _redis_client, _redis_available
    if _redis_client is not None:
        return _redis_client if _redis_available else None
    if not REDIS_URL:
        return None
    try:
        import redis
        _redis_client = redis.from_url(REDIS_URL, socket_connect_timeout=1, socket_timeout=1)
        _redis_client.ping()
        _redis_available = True
        logger.info(f"Redis cache connected: {REDIS_URL}")
    except Exception as e:
        _redis_available = False
        logger.warning(f"Redis unavailable ({e}) — cache disabled, falling back to live LLM")
    return _redis_client if _redis_available else None


def _cache_key(question: str, history_len: int) -> str:
    raw = f"{question.strip().lower()}|h={history_len}"
    return "rag:query:" + hashlib.sha256(raw.encode()).hexdigest()[:24]


def get_cached(question: str, history_len: int) -> Optional[dict]:
    """Return cached result dict or None."""
    client = _get_client()
    if not client:
        return None
    try:
        key = _cache_key(question, history_len)
        value = client.get(key)
        if value:
            logger.info(f"[CACHE] HIT — key={key}")
            return json.loads(value)
    except Exception as e:
        logger.warning(f"[CACHE] GET error: {e}")
    return None


def set_cached(question: str, history_len: int, result: dict) -> None:
    """Store result dict in cache with TTL."""
    client = _get_client()
    if not client:
        return
    try:
        key = _cache_key(question, history_len)
        client.setex(key, CACHE_TTL, json.dumps(result))
        logger.info(f"[CACHE] SET — key={key} ttl={CACHE_TTL}s")
    except Exception as e:
        logger.warning(f"[CACHE] SET error: {e}")


def invalidate(question: str, history_len: int) -> None:
    """Manually invalidate a cached entry."""
    client = _get_client()
    if not client:
        return
    try:
        client.delete(_cache_key(question, history_len))
    except Exception:
        pass
