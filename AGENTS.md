# Repository Guidelines

## Project Structure & Module Organization

This repository currently contains extracted MELD data under `MELD.Raw/`; application code and tooling are not implemented yet. Keep raw data unchanged. Training labels are in `MELD.Raw/train/train_sent_emo.csv`; development and test labels are directly under `MELD.Raw/`.

Use this structure as implementation proceeds:

- `src/sentient_signals/`: preprocessing, encoders, fusion, response policy, and streaming service.
- `app/static/`: browser interface and SVG/CSS desk-lamp character.
- `scripts/`: preparation, feature caching, training, evaluation, and benchmarking entry points.
- `tests/`: unit and integration tests; `configs/`: reproducible settings.
- `data/processed/`, `checkpoints/`, and `reports/`: generated audio/manifests, model artifacts, and measured results.

## Build, Test, and Development Commands

Python 3.12 is available. Create an isolated environment with `py -3.12 -m venv .venv`; activate it in PowerShell using `.\.venv\Scripts\Activate.ps1`.

No dependency manifest or runnable application commands exist yet. When scaffolding, define development dependencies in `pyproject.toml` and document exact preparation, training, evaluation, and serving commands. Adopt these tooling commands once configured:

- `python -m pytest`: run tests.
- `python -m ruff check .`: lint Python.
- `python -m ruff format .`: format Python.

## Coding Style & Naming Conventions

Use four-space Python indentation, type hints for public functions, `snake_case` functions/modules, and `PascalCase` classes. Use two-space indentation for browser assets. Validate API payloads with Pydantic. Keep preprocessing shared between training and inference; separate model predictions, response decisions, and lamp animation.

## Testing Guidelines

Use pytest and `test_*.py` filenames. Prioritize split-aware clip matching, resampling, feature consistency, probability validity, response grounding, and cancellation of stale turns. Include one real-clip integration test and manually verify microphone and reduced-motion behavior. No numerical coverage threshold is established.

## Commit & Pull Request Guidelines

There is no Git history yet. Use concise imperative commits, such as `Add resumable audio preparation`. PRs should explain behavior, validation, and limitations. Include screenshots for lamp/interface changes and measured comparisons for model changes.

## ML Constraints & Data Handling

Keep inference local and total learned parameters below six billion, including frozen encoders and ASR. Preserve official MELD splits; never tune on test labels. Log missing clips, including development `dia110_utt7.mp4`. Exclude raw media, caches, environments, and downloaded weights from Git. Report measured latency and resource usage; distinguish targets from results. Keep the three-day scope focused on Text + Audio, grounded templates, and the lightweight lamp.
