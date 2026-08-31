import unittest

from utils.history_selection import (
    CHUNK_START_KEY,
    SELECTION_KEY,
    begin_history_chunk,
    share_history_selection,
)


class HistorySelectionLifecycleTest(unittest.TestCase):
    def test_reuses_selection_within_chunk(self):
        selected = object()
        caches = [
            {CHUNK_START_KEY: 12, SELECTION_KEY: selected},
            {CHUNK_START_KEY: 12, SELECTION_KEY: selected},
        ]
        begin_history_chunk(caches, current_start=12)
        self.assertTrue(all(cache[SELECTION_KEY] is selected for cache in caches))

    def test_invalidates_selection_for_next_chunk(self):
        caches = [
            {CHUNK_START_KEY: 12, SELECTION_KEY: object()},
            {CHUNK_START_KEY: 12, SELECTION_KEY: object()},
        ]
        begin_history_chunk(caches, current_start=15)
        for cache in caches:
            self.assertNotIn(SELECTION_KEY, cache)
            self.assertEqual(cache[CHUNK_START_KEY], 15)

    def test_shares_layer_zero_selection(self):
        selected = object()
        caches = [{SELECTION_KEY: selected}, {}, {}]
        share_history_selection(caches)
        self.assertTrue(all(cache[SELECTION_KEY] is selected for cache in caches))


if __name__ == "__main__":
    unittest.main()
