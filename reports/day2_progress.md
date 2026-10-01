# Day 2 implementation progress

Started 2026-09-26. This is an implementation status report, not a final evaluation.

## Implemented

- Added versioned client, status, error, and result event schemas.
- Added a bounded mono PCM16 turn buffer with a hard ten-second limit.
- Runs inference off the event loop and permits cancellation while inference is active.
- Suppresses stale results by turn generation and prevents replacement work until the active
  CPU job finishes.
- Added microphone capture, prepared-WAV replay, supplied/ASR transcript selection, and a
  prediction inspection panel.
- Added an original CSS desk lamp with listening, processing, response, error, and
  reduced-motion states.
- Added deterministic whole-dialogue development/calibration partitioning, scalar temperature
  fitting, and probability-fusion helpers.
- Generalized training to text-only, audio-only, and feature-fusion heads, with unweighted and
  inverse-square-root class-weighted losses and configurable seeds.
- Added `scripts/day2_experiment.py` to cache all three splits resumably, run all 18 heads,
  compare fusion weights on development-selection dialogues, and serialize the calibrated
  seed-42 model. Runtime supports both calibrated feature and probability fusion.
- Pinned both frozen encoder revisions and propagated offline loading to them.
- Replaced the disconnected CSS lamp with a connected SVG rig: fixed base, articulated arms,
  independent head, interruptible gesture sequences, automatic return to rest, and static
  reduced-motion poses. Added a gesture preview panel.

## Validation of this update

- 24 Python tests passed, including calibrated runtime/offline probability agreement.
- Node animation tests passed for articulation, smooth interruption, return to idle, and
  reduced motion. Ruff passed.
- Rendered and visually inspected all eight SVG poses in `reports/lamp/pose-sheet.png`.
- Connected browser inventory was empty. The pose renders do not verify browser layout,
  microphone permissions, or the perceived timing of animation.
- Full preparation and the data completeness audit passed for 13,706 clips: 9,988 train,
  1,108 development, 2,610 test. Training `dia125_utt3.mp4` exists but fails decoding with
  PyAV `InvalidDataError`; development `dia110_utt7.mp4` is missing. No other exclusions.
- `scripts/check_day2.py` checks exact official IDs/labels, unique clips, WAV format/duration,
  feature alignment, all training artifacts, calibration separation, and deployed inference.

## Still pending for Day 2

- Complete-split encoder caching and the planned three-seed, two-loss, three-head training run.
- Development comparison of feature fusion against probability fusion at the planned alphas.
- Fit and persist calibration only after the deployed model is selected.
- Browser integration and live-latency review are user-owned; feedback will be recorded separately.

No final test labels have been opened for model selection.
