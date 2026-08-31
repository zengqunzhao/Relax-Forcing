import unittest
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]


class ReleaseMetadataTest(unittest.TestCase):
    def test_preferred_citation_is_bmvc_2026(self):
        with (ROOT / "CITATION.cff").open("r", encoding="utf-8") as handle:
            citation = yaml.safe_load(handle)

        preferred = citation["preferred-citation"]
        self.assertEqual(preferred["type"], "conference-paper")
        self.assertEqual(preferred["year"], 2026)
        self.assertEqual(preferred["collection-type"], "proceedings")
        self.assertIn("BMVC", preferred["conference"]["name"])

    def test_readme_announces_acceptance_and_conference_citation(self):
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("**Accepted at BMVC 2026**", readme)
        self.assertIn("@inproceedings{zhao2026relaxforcing", readme)


if __name__ == "__main__":
    unittest.main()
