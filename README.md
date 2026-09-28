# SentientSignals

SentientSignals is a local, turn-level Text + Audio emotion prototype trained on the
official MELD splits. Day 1 established the feasibility path. Day 2 adds the bounded
WebSocket protocol, microphone/WAV replay interface, cancellable stale-result handling,
calibration utilities, and the first animated lamp. See `plan.md` for the three-day scope.

## Day 1 setup (PowerShell)

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[encoders,dev]"
```

Prepare manifests and WAV audio (raw files are never modified):

```powershell
sentient-prepare --raw-root MELD.Raw --output-root data/processed
```

Run the representative 100-clip feasibility benchmark at a chosen CPU thread count:

```powershell
sentient-benchmark --manifest data/processed/dev.jsonl --limit 100 --threads 4
```

Cache Day 1 features, train a subset-labelled fusion head, and produce one JSON result:

```powershell
sentient-cache --manifest data/processed/train.jsonl --output data/features/day1_encoder_train_clean.npz --limit 300
sentient-cache --manifest data/processed/dev.jsonl --output data/features/day1_encoder_dev_clean.npz --limit 100
sentient-train --train-features data/features/day1_encoder_train_clean.npz --dev-features data/features/day1_encoder_dev_clean.npz --output checkpoints/day1_encoder_subset.pt
sentient-predict --checkpoint checkpoints/day1_encoder_subset.pt --text "You tell them to wait!" --audio data/processed/dev/dia42_utt1.wav
```

Omit `--text` to run the complete local speech-to-response path:

```powershell
sentient-predict --checkpoint checkpoints/day1_encoder_subset.pt --audio data/processed/dev/dia42_utt1.wav --threads 8 --offline
```

`--offline` works after the three selected model checkpoints have been downloaded once.
You can provide several paths after `--audio`; the process keeps every model loaded and
prints one result per file, so only the first result pays the startup cost.

`sentient-cache` uses frozen MiniLM and wav2vec2 by default. Model downloads are cached
outside Git. Add `--feature-mode feasibility` only for a fast smoke test; those compact,
deterministic features are not the planned final encoders and artifacts retain that label.

The current checkpoint is deliberately a wiring/feasibility artifact trained on 300 clips;
it is not the final model. Measured Day 1 results are in `reports/day1_summary.md`.

Run validation with:

```powershell
python -m pytest
python -m ruff check .
```

## Day 2 live interface

Install the serving dependencies and launch the local application:

```powershell
python -m pip install -e ".[encoders,asr,serve,dev]"
sentient-serve --checkpoint checkpoints/day1_encoder_subset.pt
```

Open `http://127.0.0.1:8000`. Hold the button to record, or choose a mono 16 kHz PCM16
WAV for replay. A replay may use an explicitly labelled supplied transcript; otherwise it
uses local ASR. The current checkpoint remains the Day 1 subset feasibility artifact until
the full Day 2 comparison is trained and selected.

Train an individual Day 2 comparison head while reserving 20% of whole development
dialogues for calibration:

```powershell
sentient-train --train-features data/features/train.npz --dev-features data/features/dev.npz --modality text --class-weighting inverse_sqrt --seed 42 --output checkpoints/text_weighted_seed42.pt
sentient-train --train-features data/features/train.npz --dev-features data/features/dev.npz --modality audio --class-weighting none --seed 42 --output checkpoints/audio_plain_seed42.pt
sentient-train --train-features data/features/train.npz --dev-features data/features/dev.npz --modality fusion --class-weighting inverse_sqrt --seed 42 --output checkpoints/fusion_weighted_seed42.pt
```

Repeat selection runs with seeds `13`, `42`, and `2026`. Do not pass
`--include-calibration` for Day 2 selection; that flag only reproduces the earlier feasibility
behavior.

### Full Day 2 experiment

Full-run data is separate from the Day 1 subset. Run these commands from the repository root:

```powershell
sentient-prepare --output-root data/day2/processed --splits train dev test
python scripts/day2_experiment.py --threads 8
```

Feature shards resume automatically and include audio content, transcript, encoder revisions,
and preprocessing in their cache identity. The runner trains 18 heads (three modalities, two
losses, three seeds), compares probability fusion weights on selection dialogues, and fits
temperature on the reserved calibration dialogues. Test representations are cached after model
selection; test metrics remain reserved for Day 3.

Progress is written to `reports/day2_live_status.json`, comparisons to
`reports/day2_experiment.json`, and the calibrated deployment artifact to
`checkpoints/day2/deployed.pt`. Once the run finishes, launch it explicitly:

```powershell
sentient-serve --checkpoint checkpoints/day2/deployed.pt
```

Run the completeness check with `python scripts/check_day2.py`. It validates official clip
identities, WAVs, feature alignment, all 18 heads, training-only normalization, the calibration
split, and real-clip offline inference. Results go to `reports/day2_completeness.json`.
The full prepared set contains 9,988 train / 1,108 dev / 2,610 test clips: training
`dia125_utt3.mp4` is corrupt (PyAV `InvalidDataError`), and development `dia110_utt7.mp4`
is missing. Both exclusions are recorded in `reports/data_preparation.json`.

### Lamp verification

Reload the page and expand **Preview lamp gestures** to try each animation without a microphone
or inference. The SVG uses connected arm joints, a stationary base, and an independently
controlled shade. New states interrupt smoothly; response gestures settle back to rest.
Reduced-motion preferences use static poses.

```powershell
npm ci
npm run test:lamp
npm run render:lamp
```

`reports/lamp/pose-sheet.png` shows eight rendered poses from the actual rig. These are SVG
renders, not browser screenshots. Desktop/mobile layout and motion timing still require a
browser inspection; the animation tests check interpolation, interruption, and reduced motion.

## Emotion-trained audio comparison

Compare the Day 2 speech representation with an IEMOCAP-trained SUPERB wav2vec2 emotion
representation. The experiment reuses the existing text cache, extracts resumable audio
features, and trains audio-only and feature-fusion heads across both losses and all three seeds:

```powershell
.\.venv\Scripts\Activate.ps1
python scripts/audio_encoder_experiment.py --threads 8
```

Results are written to `reports/audio_encoder_experiment.json`. Re-run only the heads with
`python scripts/audio_encoder_experiment.py --threads 8 --compare-only`. The experiment keeps
`checkpoints/day2/deployed.pt` unchanged until the new representation proves better.
After the heads exist, re-run only the probability-fusion comparison with
`python scripts/audio_encoder_experiment.py --threads 8 --probability-only`.

## Day 3 accuracy selection

Search finer text/audio fusion weights on the development selection dialogues only:

```powershell
python scripts/day3_accuracy_experiment.py
```

Review `reports/day3_accuracy_experiment.json`, then freeze the recommended seed-42 artifact
without opening test labels:

```powershell
python scripts/day3_accuracy_experiment.py --apply
```

This also reports a three-seed ensemble comparison, but does not deploy it because it loads six
heads for a smaller system-level tradeoff. Test evaluation remains a separate final step.

### Attentive audio pooling

Preserve 32 temporal audio bins and learn which parts of each utterance carry emotion:

```powershell
python scripts/attentive_audio_experiment.py --threads 8
```

Feature extraction is resumable. After the cache exists, retrain only the small attention heads
with `python scripts/attentive_audio_experiment.py --threads 8 --compare-only`. Results are
written to `reports/attentive_audio_experiment.json`; the Day 3 checkpoint is not replaced.

### Explicit prosody experiment

Append dependency-free pitch, energy, rhythm, voicing, and spectral features to the existing
wav2vec cache, then train audio and feature-fusion heads:

```powershell
python scripts/prosody_experiment.py --threads 8
```

Prosody caching resumes per clip. Retrain without recaching using
`python scripts/prosody_experiment.py --threads 8 --compare-only`. Results are written to
`reports/prosody_experiment.json`; no deployment artifact is replaced automatically.

Train a text-led dynamic gate using the cached prosody features:

```powershell
python scripts/gated_fusion_experiment.py --threads 8
```

Results are written to `reports/gated_fusion_experiment.json`; the current deployment remains
unchanged until the three-seed development mean beats the Day 3 baseline.

### Final test evaluation

After freezing the Day 3 checkpoint, run the one-shot cached MELD test evaluation:

```powershell
python scripts/day3_evaluate.py
```

This writes `reports/day3_final_evaluation.json` and `reports/day3_final_summary.md`.
