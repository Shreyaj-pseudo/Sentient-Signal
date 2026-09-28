# SentientSignals: GitHub-informed implementation plan

Status: Day 1 feasibility implementation completed on 2026-09-26, including local
ASR-to-response timing. Day 2 implementation has started with comparison-head training,
calibration utilities, a bounded live runtime service, replay, and the first lamp interface.
All 18 full-training comparison heads and deployed calibration are complete; probability
fusion with 75% text / 25% audio was selected on development data. Final feature-cache and
artifact checks are in progress. Live latency measurement and final evaluation remain pending.

## 1. Goal, constraints, and research findings

Build a local Text + Audio emotion system represented by a simple expressive desk lamp.

The visitor holds a button, speaks, and releases it. The system transcribes their speech locally, combines text and audio, predicts a MELD emotion, and produces a short grounded response with a coordinated lamp gesture.

The target machine remains your Intel Core Ultra 5 laptop with approximately 16 GB RAM and integrated graphics. Complete the prototype within three days.

### What the GitHub research tells us

Repository documentation and selected implementation files from five relevant projects were inspected. This was an architecture and code review, not a reproduction of their reported results.

| Reference | Architecture observed | Decision for our project |
|---|---|---|
| [Official MELD baseline](https://github.com/declare-lab/MELD/blob/master/baseline/baseline.py) | Separate text, audio, and bimodal paths; cached features; dialogue-level bidirectional LSTMs; validation-based checkpointing | Preserve separate modality baselines, explicit data splits, and reproducible feature extraction |
| [DialogueRNN](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/DialogueRNN/model.py) | Speaker/global/emotion state tracking; a bidirectional model processes dialogue forward and backward | Preserve dialogue IDs for analysis, but use only the completed current utterance in our first version |
| [Multimodal Emotion Recognition](https://github.com/mmakiuchi/multimodal_emotion_recognition/blob/70eaa160b1b33b8b3b03925a8e6c939ac2ab95a6/analysis/multi_ua_wa_helpers.py) | Separate speech/text models followed by a weighted combination of their probabilities | Add probability-level fusion as a cheap comparison against our learned feature-fusion classifier |
| [Real-Time Voice and Emotion Processing](https://github.com/ErdoganPeker/Real-Time-Voice-and-Emotion-Processing/blob/main/app/main.py) | PCM audio over WebSockets, buffered local Whisper inference, worker threads, hand-written acoustic emotion scoring | Adopt bounded audio transport and off-thread inference; keep our emotion model trained on MELD |
| [Human Computer Lab's lamp runtime](https://github.com/humancomputerlab/lelamp_runtime/blob/ee23699075fc3d3154da86a71c38a024545b6785/lelamp/service/motors/animation_service.py) | Named recordings, event-driven animation, interruption, interpolation from the current pose | Build a small gesture library with smooth interruption, separate from perception and response selection |

Important differences:

- The multimodal research project uses IEMOCAP and dataset-specific processing. Its results cannot be transferred directly to MELD.
- Bidirectional processing across a full dialogue can use future utterances unavailable during live interaction. Bidirectional processing within an already completed utterance is compatible with our turn-level design.
- The live demo's acoustic scores are hand-written rules with different categories; they are not evidence of MELD emotion accuracy.
- Human Computer Lab's [sample voice agent](https://github.com/humancomputerlab/lelamp_runtime/blob/ee23699075fc3d3154da86a71c38a024545b6785/main.py) uses a remote realtime model. Our core will remain entirely local.
- The inspected [DialogueRNN training script](https://github.com/declare-lab/conv-emotion/blob/6128ca20e9c736605cce7e99d5d95db0356c35f5/DialogueRNN/train_MELD.py) tracks the best test F1 during training. Our model selection will use development data only.

These findings strengthen the original plan without requiring a larger model or physical robot.

## 2. Product and architecture

### Interaction

Support one English-speaking user and push-to-talk turns of approximately 1-10 seconds.

The browser shows:

- The original desk-lamp character.
- A hold-to-talk button and recording indicator.
- The latest transcript and response.
- An expandable inspection panel.
- A MELD replay mode.

Replay sends audio at its recorded rate through the same backend. It supports supplied transcripts and local transcription, with the mode clearly labelled. Ground-truth emotions are revealed after inference and never enter the model's input.

### Processing pipeline

```text
Microphone or paced MELD replay
                |
        Buffered audio chunks
                |
        +-------+---------+
        v                 v
Local transcription   Audio encoder
        |                 |
   Text encoder           |
        +--------+--------+
                 v
       Multimodal classifier
                 |
          Emotion estimate
                 |
          Response policy
                 |
        JSON + text + gesture
```

Keep four responsibilities separate:

1. **Perception:** transcription and representations.
2. **Prediction:** emotion probabilities and uncertainty.
3. **Response policy:** reply and gesture selection.
4. **Presentation:** browser feedback and lamp animation.

This separation makes failures easier to diagnose and leaves a clear future boundary for physical hardware integration.

### Local models

| Component | Selected implementation |
|---|---|
| ASR | Whisper `tiny.en`, faster-whisper CPU INT8 |
| Text encoder | Frozen `sentence-transformers/all-MiniLM-L6-v2` |
| Audio encoder | Frozen `facebook/wav2vec2-base` backbone |
| Classifiers | Small PyTorch MLPs |
| Responses | Deterministic transcript-aware templates |
| Lamp | SVG, CSS, and JavaScript |

Expected total is roughly 160M learned parameters, subject to exact checkpoint accounting.

Frozen encoders still count toward the six-billion limit. INT8 reduces memory use, not parameter count.

### Features and classifier

- Text: pooled 384-dimensional MiniLM representation.
- Audio: mean and population standard deviation over valid wav2vec2 frame representations.
- Fit feature standardization on training data only.
- Feature-fusion classifier: concatenate both representations, then a 128-unit hidden layer, ReLU, dropout `0.2`, and seven output logits.
- Use the same feature extraction functions for training, replay, and live inference.

Cache representations with keys that include input identity, preprocessing configuration, and model revision. Run encoders in evaluation/inference mode.

The first version has no cross-turn context model. Dialogue IDs remain available for grouping, replay, and evaluation.

### Four comparisons, not four large models

Train the existing three small heads:

1. Text only.
2. Audio only.
3. Feature fusion.

Add a fourth result using the existing unimodal outputs:

```text
combined probability =
    alpha * text probability + (1 - alpha) * audio probability
```

Evaluate `alpha` in `{0.25, 0.50, 0.75}` on development-selection data. This adds no encoder and negligible runtime cost.

Select feature fusion or probability fusion using mean development macro-F1 across seeds. Break differences below 0.005 in favor of feature fusion to avoid changing architecture for a negligible observed gain.

Unimodal models remain baselines, even if one outperforms fusion. Report that outcome honestly; the submitted core must still combine both modalities.

## 3. Lamp behavior and runtime contract

### Original lamp design

Create a round base, two articulated arms, visible joints, and a rounded shade. Express character through pose and light rather than facial features.

Use authored gestures:

| Event or response intent | Gesture |
|---|---|
| Idle | Relaxed pose |
| Listening | Gentle forward lean |
| Processing | Small tilt and slow light pulse |
| Celebration | Restrained upward bounce |
| Acknowledgment | Small nod |
| Reassurance | Slow forward inclination |
| Clarification | Sideways tilt |
| Error | Neutral pose with retry text |

The lamp responds to the user rather than mirroring their emotion. Anger should lead to a calm acknowledgment.

### Animation controller

Represent gestures as short keyframe sequences containing arm angles, shade angle, and light intensity.

- Interpolate smoothly from the current pose.
- A new listening turn interrupts a previous response gesture.
- Cancellation invalidates old gestures and results.
- Return to idle after a gesture completes.
- Respect reduced-motion preferences with static poses.
- Keep technical error feedback readable without relying on motion or color.

Borrow the event-driven design from the lamp runtime. Author our own SVG and movement sequences; do not copy its identity, prompts, or recorded motor trajectories.

### Backend and events

Use FastAPI, Pydantic, WebSockets, and a static frontend.

The browser sends 16 kHz mono PCM16 in approximately 100 ms chunks. The server runs CPU inference in a worker so it can continue receiving cancellation and connection events.

Client events:

- `start_turn`
- Binary audio chunks
- `end_turn`
- `cancel_turn`

Server events:

- `status`
- `transcript`
- `result`
- `error`

Include session/turn identifiers and a schema version. Successful results contain:

- Transcript and its source.
- One MELD emotion.
- All seven probabilities.
- Uncertainty flag.
- Response text and policy identifier.
- Gesture identifier.
- Component timings.

Do not include ground-truth labels in inference requests.

Bound audio buffers and allow one active inference job. Cancellation suppresses output but does not assume a running CPU call has stopped. Do not launch unlimited replacement jobs while a cancelled job finishes.

Automatically finish live recording at ten seconds. Reject malformed audio and handle empty transcription with a retry response.

### Latency definition

Call the system **streaming input with turn-level inference**.

Targets on the actual laptop:

- Listening feedback within 100 ms.
- Median release-to-visible-result latency at most two seconds.
- p95 at most three seconds for 1-5-second turns.
- Backend memory below 8 GB.

Measure these rather than promising them. Report longer turns and cold startup separately.

Display complete response text as soon as ready. Do not introduce artificial streaming delays.

### Grounded response policy

Use transcript cues, the fused prediction, and uncertainty. Keep rules small and documented; handle explicit negation and escape transcript text.

An uncertain result receives clarification rather than a confident emotion assertion. Return an exact transcript reference where useful, without inventing a topic.

Reply and gesture must come from the same policy decision. Low confidence cannot produce a clarification sentence alongside an emphatic celebration.

## 4. Data, training, and evaluation

### Data preparation

Retain the official MELD train/development/test splits.

Previously verified matches:

- Train: 9,989 of 9,989.
- Development: 1,108 of 1,109.
- Test: 2,610 of 2,610.

Record missing development clip `dia110_utt7.mp4`. Revalidate during preparation.

Full Day 2 decoding also identified corrupt training clip `dia125_utt3.mp4` (repeated PyAV
`InvalidDataError`). Exclude and report it: 9,988 decoded train, 1,108 development, 2,610 test.

Extract mono 16 kHz WAV files with PyAV and generate JSONL manifests containing relative audio paths, transcripts, labels, duration, and dialogue/utterance IDs.

Ignore Apple metadata and unreferenced media. Record every failed decode or exclusion.

For recordings over ten seconds, encode consecutive windows and combine frame statistics using frame counts and second moments. Do not average standard deviations directly. Preserve the entire recording for offline evaluation.

Store labels by name and persist the explicit index mapping. Do not assume another repository uses our alphabetical label order.

### Training defaults

- AdamW, learning rate `1e-3`, weight decay `1e-4`.
- Batch size 64.
- Maximum 30 epochs, early stopping patience five.
- Seeds 13, 42, and 2026.
- Compare unweighted cross-entropy and inverse-square-root class weighting.
- Select using development macro-F1.
- Use seed 42 for the deployed configuration.

Partition development dialogues deterministically, using seed 42:

- 80% for model/loss/fusion selection.
- 20% for final calibration.

Fit a scalar temperature to the selected deployed model using calibration data only. For probability fusion, calibrate the final combined distribution through its log probabilities.

Mark predictions uncertain when maximum probability is below `0.55` or the top-two margin is below `0.15`. These are documented policy defaults, not universally reliable confidence thresholds.

### Evaluation

Freeze configuration before opening final test results.

Report:

- Macro-F1, weighted F1, accuracy, per-class metrics.
- Majority-class baseline and confusion matrix.
- Mean and standard deviation across seeds.
- Dialogue-bootstrap interval for the selected fused-versus-text comparison.
- Uncertainty coverage and accuracy.
- Exclusion counts.

Show audio contribution with development audio shuffling and paired recordings of identical words with different delivery.

Evaluate ASR impact on a fixed 100-example test sample. Compare emotion results using supplied and ASR transcripts on the same examples and report word error rate.

Review 20 development responses for grounding, appropriate uncertainty, brevity, and gesture consistency. Label this an author review, not a user study.

### Tests

Cover:

- Split-aware matching, label mapping, resampling, and missing inputs.
- Cached/fresh feature agreement and long-audio pooling.
- Training-only normalization.
- Finite, normalized seven-class output.
- Chunked/buffered input equivalence.
- Prediction independence from later dialogue turns.
- Cancellation, stale results, bounded work, and disconnects.
- Reply/gesture consistency and reduced motion.
- A real-clip integration test.
- Offline inference after all assets are installed.

## 5. Delivery, timebox, and saved plan

### Day 1: feasibility and first trained path

Create the isolated environment, pin dependencies, and benchmark 100 representative clips.

Measure extraction cost, memory, and complete inference latency. Test CPU thread counts 2, 4, and 8.

Prepare data, cache features, and train a small initial classifier so one real input can reach valid JSON and a reply. Any subset-trained checkpoint must be labelled as such.

Budget at most eight hours of elapsed feature extraction. If projected full extraction exceeds that budget, retain complete development/test processing and choose a deterministic training-dialogue subset that fits the remaining budget. Publish the subset manifest. If even full evaluation features exceed the budget, report that feasibility problem before changing the evaluation scope.

### Day 2: model comparison and live interaction

Finish the planned classifiers and fusion comparison. Add calibration and response rules. Connect microphone streaming, replay, cancellation, and a basic lamp.

### Day 3: final evidence and handoff

Freeze configuration, run evaluation, measure resources, finish interruptible lamp gestures, test offline operation, and record the demo.

Reduce decorative polish before reducing core validation. Omit vision, RL, TTS, physical robot control, encoder fine-tuning, and open-ended generation.

### Repository and commands

Follow `AGENTS.md`: core Python package, static browser assets, scripts, tests, configurations, and generated reports.

Expose a consistent CLI for preparation, feature caching, training, evaluation, benchmarking, and serving. Document exact working commands after implementation; none exist yet.

Keep raw media, model downloads, caches, and environments out of Git. Provide the trained small classifier artifacts and instructions to acquire encoder weights so reviewers can run without retraining.

### Required reports

- Exact learned-parameter ledger, including ASR, frozen encoders, all loaded heads, and calibration.
- Assertion that the deployed path contains at most 6B learned parameters.
- Hardware, OS, dependency revisions, thread count, peak RAM, disk requirements, and training/extraction time.
- Warm p50/p95 latency over at least 30 representative turns, plus cold startup.
- Dataset exclusions, test metrics, limitations, and completed/omitted checklist.
- Attribution for datasets, models, dependencies, research influences, and AI-generated components.

Store the inspected GitHub references and commit identifiers with the architecture rationale. Verify licenses before reusing code; the default is an original implementation informed by the observed designs.

### Demonstration

Show live speech, the listening lamp, the resulting prediction/reply, and the inspection panel. Then show a MELD replay, paired delivery examples, one failure or uncertainty case, and the measured resource results.

The submission must distinguish model evidence from authored animation choices and qualitative demonstrations.

### Plan persistence

This file is the complete replacement plan, saved as `plan.md` in the repository root. Keep it updated when agreed implementation decisions change, including their reasons and supporting evidence.
