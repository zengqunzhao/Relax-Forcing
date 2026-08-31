from __future__ import annotations

import argparse
import sys
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from relax_forcing.methods import METHODS


BASE_REPO = "Wan-AI/Wan2.1-T2V-1.3B"
BASE_DIR = ROOT / "wan_models" / "Wan2.1-T2V-1.3B"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download base and method weights")
    parser.add_argument("--method", choices=["all", *sorted(METHODS)], default="relax_forcing")
    parser.add_argument("--skip-base", action="store_true")
    args = parser.parse_args()

    if not args.skip_base:
        print(f"Downloading {BASE_REPO} to {BASE_DIR}")
        snapshot_download(repo_id=BASE_REPO, local_dir=BASE_DIR)

    names = METHODS if args.method == "all" else {args.method: METHODS[args.method]}
    downloads = {(spec.checkpoint_repo, spec.checkpoint_file) for spec in names.values()}
    for repo_id, filename in sorted(downloads):
        print(f"Downloading {repo_id}/{filename}")
        path = hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            local_dir=ROOT,
        )
        print(f"Saved to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
