# Contributing

## Development Setup

```bash
conda create -n relax-forcing-dev python=3.10 -y
conda activate relax-forcing-dev
pip install -r requirements-dev.txt
pip install flash-attn --no-build-isolation
pip install -e .
```

Editable installation is recommended for development. The release CI also
tests a regular `pip install .` to verify that packaged configuration files and
the console entry point work outside the source tree.

Run the CPU contract suite before opening a pull request:

```bash
python -m unittest discover -s tests -v
python -m compileall -q relax_forcing pipeline scripts utils wan
```

Changes to a model adapter should also be checked on a CUDA GPU using the
corresponding upstream checkpoint. Include the exact command and hardware in
the pull request.

## Repository Rules

- Do not commit model weights, generated videos, caches, or experiment logs.
- Keep method defaults in `configs/methods/`; avoid method-specific CLI forks.
- Preserve upstream attribution and license notices when adapting baseline code.
- Add a CPU-level contract test for configuration or cache-policy changes.
- Keep unrelated formatting and refactoring out of focused pull requests.
