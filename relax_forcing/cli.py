from __future__ import annotations

import argparse
import json
from pathlib import Path

from yaml import YAMLError

from .config import load_config
from .methods import METHODS, resolve_config
from .runtime import prepare_config, read_prompts, run_inference, validate_config


ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="relax-forcing",
        description="Unified long-video inference for Relax Forcing and supported baselines.",
    )
    parser.add_argument("--method", choices=sorted(METHODS), default="relax_forcing")
    parser.add_argument("--list-methods", action="store_true")
    parser.add_argument("--config", help="Override the method's YAML config")
    parser.add_argument("--checkpoint-path")
    parser.add_argument("--checkpoint-key")

    prompts = parser.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file")
    parser.add_argument("--start-prompt-index", type=int, default=0)
    parser.add_argument("--num-prompts", type=int)
    parser.add_argument("--num-samples", type=int, default=1)

    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--num-output-frames", type=int, default=240)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--low-memory", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--low-memory-threshold", type=float, default=40.0)
    parser.add_argument("--profile", action="store_true")

    parser.add_argument("--kv-cache-size", type=int)
    parser.add_argument("--sink-frames", type=int)
    parser.add_argument("--hist-frames", type=int)
    parser.add_argument("--tail-frames", type=int)
    parser.add_argument("--hist-position-idx", type=int)
    parser.add_argument("--num-hist-candidates", type=int)
    parser.add_argument("--lambda-redundancy", type=float)
    parser.add_argument(
        "--contiguous-rope",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the resolved configuration without loading CUDA.",
    )
    return parser


def _memory_overrides(args) -> dict:
    return {
        "kv_cache_size": args.kv_cache_size,
        "sink_frames": args.sink_frames,
        "hist_frames": args.hist_frames,
        "tail_frames": args.tail_frames,
        "hist_position_idx": args.hist_position_idx,
        "num_hist_candidates": args.num_hist_candidates,
        "lambda_redundancy": args.lambda_redundancy,
        "contiguous_rope": args.contiguous_rope,
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.list_methods:
        for name, spec in METHODS.items():
            print(f"{name:18} {spec.description}")
        return 0

    try:
        config_path = resolve_config(ROOT, args.method, args.config)
        config = prepare_config(load_config(config_path), _memory_overrides(args))
        validate_config(config, args.num_output_frames)
    except (OSError, TypeError, ValueError, YAMLError) as exc:
        parser.error(str(exc))

    if args.dry_run:
        print(json.dumps({"method": args.method, "config": config}, indent=2))
        return 0

    if not args.checkpoint_path:
        parser.error("--checkpoint-path is required unless --dry-run is used")
    if not Path(args.checkpoint_path).is_file():
        parser.error(f"checkpoint does not exist: {args.checkpoint_path}")
    if args.start_prompt_index < 0:
        parser.error("--start-prompt-index must be non-negative")
    if args.num_prompts is not None and args.num_prompts <= 0:
        parser.error("--num-prompts must be positive")
    if args.num_samples <= 0:
        parser.error("--num-samples must be positive")
    if args.fps <= 0:
        parser.error("--fps must be positive")
    try:
        prompts = read_prompts(args.prompt, args.prompt_file)
    except ValueError as exc:
        parser.error(str(exc))
    if args.start_prompt_index >= len(prompts):
        parser.error(
            f"--start-prompt-index {args.start_prompt_index} exceeds "
            f"the {len(prompts)} available prompts"
        )

    run_inference(args, config, prompts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
