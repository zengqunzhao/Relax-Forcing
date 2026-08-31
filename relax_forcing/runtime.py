from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from .config import Config


MEMORY_FIELDS = (
    "sink_frames",
    "hist_frames",
    "tail_frames",
    "hist_position_idx",
    "num_hist_candidates",
    "lambda_redundancy",
    "contiguous_rope",
)


def prepare_config(config: Config, cli_overrides: dict | None = None) -> Config:
    """Promote method memory settings to the legacy pipeline interface."""
    overrides = cli_overrides or {}
    memory = dict(config.get("memory", {}))
    for field in MEMORY_FIELDS:
        value = overrides.get(field)
        if value is not None:
            memory[field] = value

    model_kwargs = dict(config.get("model_kwargs", {}))
    if overrides.get("kv_cache_size") is not None:
        model_kwargs["local_attn_size"] = overrides["kv_cache_size"]

    config.model_kwargs = model_kwargs
    config.memory = memory
    config.kv_cache_sink_frames = int(memory.get("sink_frames", 0))
    config.kv_cache_hist_frames = int(memory.get("hist_frames", 0))
    config.kv_cache_tail_frames = int(memory.get("tail_frames", 0))
    config.kv_cache_hist_position_idx = int(memory.get("hist_position_idx", -1))
    config.kv_cache_num_hist_candidates = int(memory.get("num_hist_candidates", 0))
    config.lambda_redundancy = float(memory.get("lambda_redundancy", 0.0))
    config.contiguous_rope = bool(memory.get("contiguous_rope", False))
    return config


def validate_config(config: Config, num_output_frames: int) -> None:
    block = int(config.get("num_frame_per_block", 1))
    if num_output_frames <= 0 or num_output_frames % block:
        raise ValueError(
            f"num_output_frames must be a positive multiple of the {block}-frame block size"
        )

    cache_size = int(config.model_kwargs.get("local_attn_size", -1))
    if cache_size == 0 or cache_size < -1:
        raise ValueError("KV cache size must be -1 (unbounded) or a positive integer")

    memory_fields = {
        "sink_frames": config.kv_cache_sink_frames,
        "hist_frames": config.kv_cache_hist_frames,
        "tail_frames": config.kv_cache_tail_frames,
        "num_hist_candidates": config.kv_cache_num_hist_candidates,
    }
    invalid = [name for name, value in memory_fields.items() if value < 0]
    if invalid:
        raise ValueError(f"memory sizes must be non-negative: {', '.join(invalid)}")
    memory_size = (
        config.kv_cache_sink_frames
        + config.kv_cache_hist_frames
        + config.kv_cache_tail_frames
        + block
    )
    if cache_size != -1 and memory_size > cache_size:
        raise ValueError(
            f"selected memory plus current block ({memory_size} frames) exceeds "
            f"the {cache_size}-frame KV cache"
        )

    if config.kv_cache_hist_frames and config.kv_cache_num_hist_candidates < config.kv_cache_hist_frames:
        raise ValueError("num_hist_candidates must be at least hist_frames")


def read_prompts(prompt: str | None, prompt_file: str | None) -> list[str]:
    if prompt is not None:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("--prompt cannot be empty")
        return [prompt]
    if not prompt_file:
        raise ValueError("provide --prompt or --prompt-file")

    with Path(prompt_file).open("r", encoding="utf-8") as handle:
        prompts = [line.strip() for line in handle if line.strip()]
    if not prompts:
        raise ValueError(f"no prompts found in {prompt_file}")
    return prompts


def _safe_stem(prompt: str, index: int, sample: int) -> str:
    text = re.sub(r"[/\\:\r\n\t]+", "_", prompt).strip(" .")
    text = text.encode("utf-8")[:200].decode("utf-8", errors="ignore")
    return f"{text or f'prompt_{index:04d}'}-{sample}"


def _load_state_dict(torch, checkpoint_path: str, checkpoint_key: str):
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict):
        raise ValueError("checkpoint must contain a state-dict mapping")
    if checkpoint_key not in payload:
        available = ", ".join(sorted(payload))
        raise KeyError(
            f"checkpoint key {checkpoint_key!r} was not found; available keys: {available}"
        )
    return payload[checkpoint_key]


def run_inference(args, config: Config, prompts: list[str]) -> None:
    """Run distributed or single-GPU inference after CLI/config validation."""
    import torch
    import torch.distributed as dist
    from einops import rearrange
    from torchvision.io import write_video
    from utils.misc import set_seed

    if not torch.cuda.is_available():
        raise RuntimeError("Relax Forcing inference requires a CUDA-capable GPU")

    if "LOCAL_RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        local_rank = rank = 0
        world_size = 1
        device = torch.device("cuda")

    # The low-memory helper binds its default device during import.
    import demo_utils.memory as memory

    memory.gpu = device
    from pipeline.causal_inference import CausalInferencePipeline

    pipeline_name = config.get("pipeline", "causal")
    if pipeline_name == "causal":
        pipeline = CausalInferencePipeline(config, device=device)
    elif pipeline_name == "rolling":
        from pipeline.rolling_forcing_inference import RollingForcingInferencePipeline

        pipeline = RollingForcingInferencePipeline(config, device=device)
    else:
        raise ValueError(f"unsupported pipeline: {pipeline_name}")

    checkpoint_key = args.checkpoint_key or config.get("checkpoint_key")
    state_dict = _load_state_dict(torch, args.checkpoint_path, checkpoint_key)
    pipeline.generator.load_state_dict(state_dict, strict=True)

    pipeline = pipeline.to(dtype=torch.bfloat16)
    free_vram = memory.get_cuda_free_memory_gb(device)
    low_memory = args.low_memory == "on" or (
        args.low_memory == "auto" and free_vram < args.low_memory_threshold
    )
    if low_memory:
        memory.DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
    else:
        pipeline.text_encoder.to(device=device)
    pipeline.generator.to(device=device)
    pipeline.vae.to(device=device)

    selected = prompts[args.start_prompt_index :]
    if args.num_prompts is not None:
        selected = selected[: args.num_prompts]
    indexed_prompts = list(enumerate(selected, start=args.start_prompt_index))
    local_items = indexed_prompts[rank::world_size]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    if dist.is_initialized():
        dist.barrier()

    shape = list(config.image_or_video_shape)
    channels, height, width = map(int, shape[-3:])
    mapping = {}
    started = time.time()
    torch.set_grad_enabled(False)

    for global_index, text_prompt in local_items:
        for sample_index in range(args.num_samples):
            seed = args.seed + global_index * args.num_samples + sample_index
            set_seed(seed)
            noise = torch.randn(
                [1, args.num_output_frames, channels, height, width],
                device=device,
                dtype=torch.bfloat16,
            )
            video, _ = pipeline.inference(
                noise=noise,
                text_prompts=[text_prompt],
                return_latents=True,
                initial_latent=None,
                low_memory=low_memory,
                profile=args.profile,
            )
            frames = (
                255.0 * rearrange(video, "b t c h w -> b t h w c").cpu()
            ).round().to(torch.uint8)
            filename = _safe_stem(text_prompt, global_index, sample_index) + ".mp4"
            write_video(str(output_dir / filename), frames[0], fps=args.fps)
            mapping[filename] = {"prompt": text_prompt, "seed": seed}
            pipeline.vae.model.clear_cache()

    if dist.is_initialized():
        gathered = [None] * world_size
        dist.all_gather_object(gathered, mapping)
        mapping = {key: value for part in gathered for key, value in part.items()}
        dist.barrier()

    if rank == 0:
        prompt_mapping = {
            filename: details["prompt"] for filename, details in mapping.items()
        }
        metadata = {
            "method": args.method,
            "checkpoint": args.checkpoint_path,
            "checkpoint_key": checkpoint_key,
            "elapsed_seconds": time.time() - started,
            "videos": mapping,
        }
        with (output_dir / "metadata.json").open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, ensure_ascii=False)
        with (output_dir / "prompt_mapping.json").open(
            "w", encoding="utf-8"
        ) as handle:
            json.dump(prompt_mapping, handle, indent=2, ensure_ascii=False)

    if dist.is_initialized():
        dist.destroy_process_group()
