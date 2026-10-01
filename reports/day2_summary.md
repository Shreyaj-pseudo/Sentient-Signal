# Day 2 full training experiment

The 18 planned heads completed training on 9,988 training clips. Model selection used
912 development utterances; 196 utterances in separate development dialogues were reserved
for calibration. Dialogue partition seed: 42. Model seeds: 13, 42, 2026.

## Development selection results

Macro-F1 mean and sample standard deviation across three seeds, with supplied transcripts:

| Model | Loss | Mean | Standard deviation |
|---|---|---:|---:|
| Text | Unweighted | 0.3894 | 0.0068 |
| Text | Inverse square root | 0.4226 | 0.0051 |
| Audio | Unweighted | 0.2336 | 0.0132 |
| Audio | Inverse square root | 0.2424 | 0.0062 |
| Feature fusion | Unweighted | 0.3677 | 0.0011 |
| Feature fusion | Inverse square root | 0.3865 | 0.0085 |
| Probability fusion, 25% text | Best unimodal losses | 0.3119 | 0.0245 |
| Probability fusion, 50% text | Best unimodal losses | 0.4014 | 0.0112 |
| Probability fusion, 75% text | Best unimodal losses | **0.4230** | 0.0147 |

Selected probability fusion (75% text / 25% audio) exceeds the best feature-fusion mean by
0.0365, above the planned 0.005 preference threshold. Its advantage over text alone is only
0.0004; this experiment does not establish a meaningful audio benefit. These selection-set
results are not final test estimates.

## Deployment and calibration

- Artifact: `checkpoints/day2/deployed.pt`.
- Uses the seed-42 text and audio heads, both trained with inverse-square-root class weighting.
- Fits one temperature on the log of the combined probabilities using calibration dialogues only.
- Temperature: 0.9757875. Calibration negative log likelihood: 1.428042 before, 1.427801 after.
- Calibration-set accuracy: 0.5102. Calibration data was used to fit the temperature and this
  is not an independent final evaluation.
- Head training, comparison, and calibration together took 146.05 seconds after features
  were available. Feature extraction is a separate, longer stage.

## Data completeness

The prepared-data audit verified unique official IDs, transcripts, labels, mono PCM16 at
16 kHz, and non-empty WAVs with matching durations for all 13,706 clips.

| Split | Official labels | Prepared | Exclusion |
|---|---:|---:|---|
| Train | 9,989 | 9,988 | `dia125_utt3.mp4`: repeated PyAV InvalidDataError |
| Development | 1,109 | 1,108 | `dia110_utt7.mp4`: missing media |
| Test | 2,610 | 2,610 | None |

Raw media was left unchanged. Separate Day 2 manifests, WAVs, and cache shards are under
`data/day2/`. Cache identities include WAV content, normalized transcript, encoder revision,
and preprocessing. All 9,988 training and 1,108 development feature vectors are cached.
Test caching and the final artifact completeness check are in progress.

## Lamp and validation

The disconnected CSS pieces were replaced with an original connected SVG rig. The base
stays fixed; both arms and the shade move independently. Nod, celebration, reassurance, and
clarification gestures interrupt from the current pose and settle to idle. Reduced motion
uses static poses. A preview panel exercises the gestures without recording.

The actual SVG was rendered in eight poses and visually inspected in
`reports/lamp/pose-sheet.png`. This checks geometry and pose appearance; it is not a browser
screenshot or proof of live animation timing. No browser was connected for desktop/mobile QA.

24 Python tests and the Node lamp-controller tests passed. Ruff passed.

Full test metrics, ASR impact evaluation, and the final response review remain Day 3 evidence
work. Browser latency and microphone/reduced-motion review are user-owned. The response policy
now includes the planned transcript-cue and negation rules.
