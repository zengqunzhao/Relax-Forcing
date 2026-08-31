from __future__ import annotations

from dataclasses import dataclass
from importlib.metadata import PackageNotFoundError, distribution
from pathlib import Path


@dataclass(frozen=True)
class MethodSpec:
    name: str
    config: str
    checkpoint_repo: str
    checkpoint_file: str
    checkpoint_key: str
    description: str


METHODS = {
    "relax_forcing": MethodSpec(
        name="relax_forcing",
        config="configs/methods/relax_forcing.yaml",
        checkpoint_repo="gdhe17/Self-Forcing",
        checkpoint_file="checkpoints/self_forcing_dmd.pt",
        checkpoint_key="generator_ema",
        description="Relaxed Sink/History/Tail KV memory (ours)",
    ),
    "self_forcing": MethodSpec(
        name="self_forcing",
        config="configs/methods/self_forcing.yaml",
        checkpoint_repo="gdhe17/Self-Forcing",
        checkpoint_file="checkpoints/self_forcing_dmd.pt",
        checkpoint_key="generator_ema",
        description="Dense sliding-window Self-Forcing baseline",
    ),
    "attention_sink": MethodSpec(
        name="attention_sink",
        config="configs/methods/attention_sink.yaml",
        checkpoint_repo="gdhe17/Self-Forcing",
        checkpoint_file="checkpoints/self_forcing_dmd.pt",
        checkpoint_key="generator_ema",
        description="Self-Forcing with persistent initial-frame anchors",
    ),
    "causvid": MethodSpec(
        name="causvid",
        config="configs/methods/causvid.yaml",
        checkpoint_repo="tianweiy/CausVid",
        checkpoint_file="autoregressive_checkpoint/model.pt",
        checkpoint_key="generator",
        description="CausVid checkpoint and distilled timestep schedule",
    ),
    "rolling_forcing": MethodSpec(
        name="rolling_forcing",
        config="configs/methods/rolling_forcing.yaml",
        checkpoint_repo="TencentARC/RollingForcing",
        checkpoint_file="checkpoints/rolling_forcing_dmd.pt",
        checkpoint_key="generator_ema",
        description="Rolling-window denoising baseline",
    ),
    "deep_forcing": MethodSpec(
        name="deep_forcing",
        config="configs/methods/deep_forcing.yaml",
        checkpoint_repo="gdhe17/Self-Forcing",
        checkpoint_file="checkpoints/self_forcing_dmd.pt",
        checkpoint_key="generator_ema",
        description="Deep Sink with Participative Compression baseline",
    ),
}


def resolve_config(root: Path, method: str, override: str | None = None) -> Path:
    if method not in METHODS:
        raise ValueError(f"Unknown method: {method}")
    if override:
        return Path(override).resolve()

    relative_path = Path(METHODS[method].config)
    source_path = root / relative_path
    if source_path.is_file():
        return source_path

    try:
        installed_distribution = distribution("relax-forcing")
        package_files = installed_distribution.files or ()
    except PackageNotFoundError:
        return source_path

    installed_suffix = Path("share") / "relax-forcing" / relative_path
    suffix = installed_suffix.as_posix()
    for package_file in package_files:
        if Path(str(package_file)).as_posix().endswith(suffix):
            installed_path = Path(
                installed_distribution.locate_file(package_file)
            ).resolve()
            if installed_path.is_file():
                return installed_path
    return source_path
