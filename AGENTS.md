# Repository Guidelines

## Project Structure & Module Organization
- Core library lives in `src/openpi`, with subpackages `models`, `policies`, `training`, `shared`, and PyTorch builds in `models_pytorch`.
- CLI utilities and automation scripts reside in `scripts/`; reproducible workflows and robot-specific tutorials live under `examples/`.
- Reference docs, setup notes, and architecture guides are in `docs/`; pinned vendor code sits in `third_party/`.
- The `packages/openpi-client` directory contains the lightweight client used when talking to remote policy servers.

## Build, Test, and Development Commands
- `uv sync && uv pip install -e .` installs dependencies and links the repo for editable development.
- `uv run pytest` executes the Python suite; append `-m "not manual"` to skip long-running robot integration cases.
- `uv run scripts/compute_norm_stats.py --config-name <config>` prepares dataset statistics before training.
- `uv run scripts/train.py <config> --exp-name=<run>` launches a training job; results land in `checkpoints/<config>/<run>/`.
- `uv run scripts/serve_policy.py policy:checkpoint --policy.config=<name> --policy.dir=<path>` starts the inference server used by evaluation clients.

## Coding Style & Naming Conventions
- Python modules follow PEP 8 with 4-space indentation and type hints; files and configs stay snake_case (e.g., `pi05_libero`).
- Format code with `ruff format .` and fix lint issues via `ruff check . --fix` before committing.
- Keep public APIs documented in docstrings; prefer explicit exports in `__init__.py` files when exposing new entry points.

## Testing Guidelines
- Unit tests co-locate with implementation (e.g., `src/openpi/transforms_test.py`); mirror module names with a `_test.py` suffix.
- Tag hardware-dependent or slow scenarios with `@pytest.mark.manual` so `-m "not manual"` remains fast for CI.
- Extend fixtures in `src/openpi/conftest.py` for shared utilities, and mock external downloads via `src/openpi/shared/download_test.py` patterns.
- Aim to maintain coverage on touched modules; document any untestable behaviors directly in the PR.

## Commit & Pull Request Guidelines
- Write concise, imperative commits; conventional prefixes like `feat:`, `fix:`, and `docs:` keep history skimmable (`git log` uses them today).
- Ensure `pre-commit run --all-files`, `ruff check`, and `pytest` complete cleanly before requesting review.
- Summarize PR intent, note affected robot platforms or configs, and link issues/discussions plus relevant logs or screenshots.
- Avoid committing large checkpoints; instead, reference external storage paths or add download instructions in `docs/`.

## 用中文和我交流