# Day 1 feasibility report

Measured on 2026-09-26 on Windows 11 with Python 3.12.6. These are local
measurements, not targets or results borrowed from another system.

## Completed

- Created an isolated `.venv` and exact-pinned core, encoder, and development dependencies.
- Revalidated the official labels against the media paths: train has 9,989 matches;
  development has 1,108 matches and is missing `dia110_utt7.mp4` as expected.
- Prepared a deterministic label-stratified subset of 300 train and 300 development clips
  as mono 16 kHz PCM16 WAV. No selected clip failed to decode.
- Cached frozen MiniLM (384 dimensions) and wav2vec2 mean/std features (1,536 dimensions)
  for 300 train and 100 development examples.
- Trained a 246,791-parameter fusion head with training-only normalization. The artifact is
  labelled `day1_subset_encoders_feasibility`.
- Added faster-whisper 1.2.1 with `tiny.en` using CPU INT8 inference.
- Ran real MELD WAV files through local transcription, both emotion encoders, the head,
  uncertainty logic, grounded response policy, and the validated seven-probability JSON schema.
- Added a reusable inference pipeline that keeps all models resident between turns.
- Normalized legacy Windows punctuation in MELD text before feature extraction and WER scoring.

## Measurements

The complete Day 1 path contains 155,091,975 learned parameters:

| Component | Learned parameters |
|---|---:|
| faster-whisper `tiny.en` | 37,760,256 |
| MiniLM text encoder | 22,713,216 |
| wav2vec2 audio encoder | 94,371,712 |
| Fusion head | 246,791 |
| **Total** | **155,091,975** |

Frozen and quantized weights still count in this ledger. The total is well below six billion.

The same representative 100-development-clip set was tested sequentially at each thread count:

| CPU threads | Wall time | Per-clip p50 | Per-clip p95 |
|---:|---:|---:|---:|
| 2 | 73.3 s | 536 ms | 1,296 ms |
| 4 | 58.3 s | 413 ms | 963 ms |
| 8 | 41.1 s | 268 ms | 652 ms |

An encoder-only repeat at eight threads measured 11.34 s cold model load plus first inference,
327 ms mean warm feature latency, 275 ms p50, 712 ms p95, and 1.06 GB peak process RSS.
This measurement excludes local ASR and browser presentation.

The final persistent end-to-end benchmark used 30 deterministic, label-stratified development
clips. It includes WAV reading, local transcription, MiniLM, wav2vec2, the fusion head, and the
response policy:

| Measurement | Result |
|---|---:|
| Cold startup plus first complete result | 20.62 s |
| Warm complete-result mean | 1.08 s |
| Warm complete-result p50 | 1.03 s |
| Warm complete-result p95 | 1.43 s |
| Mean transcription portion | 571 ms |
| Mean emotion-feature portion | 506 ms |
| Peak process RSS | 1.21 GB |
| Mean word error rate | 46.9% |

The high WER is a measured result, not hidden: several short MELD clips contain overlapping
dialogue or context beyond the labelled utterance. A first audio-only example transcribed
`You tell them to wait!` as `You tell him to wait.`. Day 2 evaluation will measure how this
transcription noise changes emotion accuracy against supplied transcripts on the same examples.

The punctuation-normalized 300/100 subset head reached development macro-F1 0.207. This small balanced subset was
chosen to prove the pipeline, so that number is not a model-selection or final-quality claim.

## Next work: Day 2

- Cache the complete official splits (the measured extraction rate is safely within the
  eight-hour budget), then run the planned seeds, losses, unimodal heads, and fusion comparison.
- Add calibration, a persistent FastAPI/WebSocket service, cancellation, replay, microphone
  input, and the first lamp interface. Final test evaluation remains reserved for Day 3.
