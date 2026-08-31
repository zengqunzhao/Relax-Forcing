"""Lifecycle helpers for chunk-level Relaxed KV history selection."""

SELECTION_KEY = "selected_hist_frame_idx"
CHUNK_START_KEY = "selected_hist_chunk_start"


def begin_history_chunk(kv_cache, current_start):
    """Invalidate layer selections when autoregressive generation advances."""
    if not kv_cache:
        return

    chunk_start = int(current_start)
    if kv_cache[0].get(CHUNK_START_KEY) == chunk_start:
        return

    for layer_cache in kv_cache:
        layer_cache.pop(SELECTION_KEY, None)
        layer_cache[CHUNK_START_KEY] = chunk_start


def share_history_selection(kv_cache):
    """Share the history indices selected by layer 0 with every layer."""
    if not kv_cache:
        return

    selected = kv_cache[0].get(SELECTION_KEY)
    if selected is None:
        return

    for layer_cache in kv_cache[1:]:
        layer_cache[SELECTION_KEY] = selected
