"""Shared KV-cache lifecycle operations."""

CORE_CACHE_KEYS = {"k", "v", "global_end_index", "local_end_index"}


def reset_kv_cache(cache):
    """Reset indices and discard method-specific metadata between videos."""
    cache["global_end_index"].zero_()
    cache["local_end_index"].zero_()
    for key in tuple(cache):
        if key not in CORE_CACHE_KEYS:
            del cache[key]
