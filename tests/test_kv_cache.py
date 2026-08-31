import unittest

from utils.kv_cache import reset_kv_cache


class FakeIndex:
    def __init__(self, value):
        self.value = value

    def zero_(self):
        self.value = 0


class KVCacheLifecycleTest(unittest.TestCase):
    def test_reset_preserves_storage_and_removes_method_metadata(self):
        key_storage = object()
        value_storage = object()
        global_index = FakeIndex(42)
        local_index = FakeIndex(21)
        cache = {
            "k": key_storage,
            "v": value_storage,
            "global_end_index": global_index,
            "local_end_index": local_index,
            "selected_hist_frame_idx": object(),
            "win_q": object(),
            "abs_frame_idx": object(),
        }

        reset_kv_cache(cache)

        self.assertIs(cache["k"], key_storage)
        self.assertIs(cache["v"], value_storage)
        self.assertEqual(global_index.value, 0)
        self.assertEqual(local_index.value, 0)
        self.assertEqual(
            set(cache), {"k", "v", "global_end_index", "local_end_index"}
        )


if __name__ == "__main__":
    unittest.main()
