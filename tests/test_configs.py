import unittest
from pathlib import Path
from unittest.mock import patch

from relax_forcing.config import load_config
from relax_forcing.methods import METHODS, resolve_config
from relax_forcing.runtime import prepare_config, validate_config
from relax_forcing.runtime import _safe_stem


ROOT = Path(__file__).resolve().parents[1]


class MethodConfigTest(unittest.TestCase):
    def test_every_method_has_a_valid_config(self):
        for name, spec in METHODS.items():
            with self.subTest(method=name):
                path = resolve_config(ROOT, name)
                self.assertTrue(path.is_file(), spec.config)
                config = prepare_config(load_config(path))
                self.assertEqual(config.method, name)
                self.assertEqual(config.checkpoint_key, spec.checkpoint_key)
                validate_config(config, num_output_frames=240)

    def test_installed_config_fallback(self):
        config_path = ROOT / METHODS["relax_forcing"].config

        class FakeDistribution:
            files = [
                Path("../../../share/relax-forcing")
                / METHODS["relax_forcing"].config
            ]

            @staticmethod
            def locate_file(_package_file):
                return config_path

        with patch("relax_forcing.methods.distribution", return_value=FakeDistribution()):
            resolved = resolve_config(Path("/missing/source/root"), "relax_forcing")

        self.assertEqual(resolved, config_path.resolve())

    def test_paper_defaults(self):
        config = prepare_config(load_config(resolve_config(ROOT, "relax_forcing")))
        self.assertEqual(config.kv_cache_sink_frames, 2)
        self.assertEqual(config.kv_cache_hist_frames, 1)
        self.assertEqual(config.kv_cache_tail_frames, 1)
        self.assertEqual(config.kv_cache_num_hist_candidates, 4)
        self.assertEqual(config.lambda_redundancy, 2.0)

    def test_cli_override_is_applied(self):
        config = prepare_config(
            load_config(resolve_config(ROOT, "relax_forcing")),
            {"lambda_redundancy": 4.0, "kv_cache_size": 96},
        )
        self.assertEqual(config.lambda_redundancy, 4.0)
        self.assertEqual(config.model_kwargs["local_attn_size"], 96)

    def test_invalid_frame_count_is_rejected(self):
        config = prepare_config(load_config(resolve_config(ROOT, "self_forcing")))
        with self.assertRaisesRegex(ValueError, "multiple"):
            validate_config(config, num_output_frames=241)

    def test_negative_memory_size_is_rejected(self):
        config = prepare_config(
            load_config(resolve_config(ROOT, "relax_forcing")),
            {"sink_frames": -1},
        )
        with self.assertRaisesRegex(ValueError, "non-negative"):
            validate_config(config, num_output_frames=240)

    def test_vbench_filename_keeps_prompt_and_sample_suffix(self):
        stem = _safe_stem("A scene/with:unsafe characters", index=4, sample=2)
        self.assertEqual(stem, "A scene_with_unsafe characters-2")


if __name__ == "__main__":
    unittest.main()
