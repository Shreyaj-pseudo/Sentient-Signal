"""Check complete prepared splits, cached features, and all experiment artifacts.

No test predictions or test metrics are computed. Run from the repository root.
"""

from __future__ import annotations

import json
import wave
from pathlib import Path

import numpy as np
import torch

from sentient_signals.asr import WHISPER_TINY_EN_PARAMETERS
from sentient_signals.calibration import dialogue_calibration_mask
from sentient_signals.constants import EMOTIONS
from sentient_signals.data import load_manifest, read_split
from sentient_signals.model import InferencePipeline, _feature_matrix


def main() -> None:
    root = Path("data/day2")
    report: dict = {"passed": [], "pending": [], "failures": []}
    for split, expected in (("train", 9988), ("dev", 1108), ("test", 2610)):
        try:
            items = load_manifest(root / "processed" / f"{split}.jsonl")
            raw, missing = read_split(Path("MELD.Raw"), split, root / "processed")
            excluded = {"dia125_utt3"} if split == "train" else set()
            expected_missing = ["dev/dev_splits_complete/dia110_utt7.mp4"] if split == "dev" else []
            assert missing == expected_missing, f"Unexpected missing source clips: {missing}"
            official = {item.clip_id: item for item in raw if item.clip_id not in excluded}
            assert len(items) == expected, f"Expected {expected}, got {len(items)}"
            assert len({item.clip_id for item in items}) == len(items), "Duplicate clip IDs"
            assert {item.clip_id for item in items} == set(official), "Clip identity mismatch"
            for item in items:
                original = official[item.clip_id]
                assert item.split == split and item.text == original.text
                assert item.label == original.label and item.emotion == EMOTIONS[item.label]
                with wave.open(item.audio_path, "rb") as audio:
                    assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (
                        1,
                        2,
                        16000,
                    ), item.audio_path
                    assert audio.getnframes() > 0, item.audio_path
                    assert abs(audio.getnframes() / 16000 - item.duration_seconds) < 1e-6
            report["passed"].append(f"{split}: {expected} unique official clips and valid WAVs")
            cache = root / "features" / f"{split}.npz"
            if not cache.exists():
                report["pending"].append(f"{split} feature archive")
                continue
            with np.load(cache, allow_pickle=False) as data:
                assert data["clip_ids"].tolist() == [item.clip_id for item in items]
                assert data["labels"].tolist() == [item.label for item in items]
                assert str(data["feature_mode"].item()) == "encoders"
                for modality, dimension in (("text", 384), ("audio", 1536)):
                    assert data[modality].shape == (expected, dimension)
                    assert np.isfinite(data[modality]).all()
            report["passed"].append(f"{split}: features finite, correctly shaped and aligned")
        except (AssertionError, OSError, ValueError) as error:
            report["failures"].append(f"{split}: {error}")

    result_path = Path("reports/day2_experiment.json")
    if not result_path.exists():
        report["pending"].append("18-head comparison and calibrated deployment")
    else:
        try:
            result = json.loads(result_path.read_text())
            with (
                np.load(root / "features/train.npz", allow_pickle=False) as train,
                np.load(root / "features/dev.npz", allow_pickle=False) as dev,
            ):
                mask = dialogue_calibration_mask(dev["clip_ids"])
                assert result["calibration_clip_ids"] == dev["clip_ids"][mask].tolist()
                assert result["selection_clip_ids"] == dev["clip_ids"][~mask].tolist()
                select_dialogues = {clip.split("_")[0] for clip in result["selection_clip_ids"]}
                calibrate_dialogues = {
                    clip.split("_")[0] for clip in result["calibration_clip_ids"]
                }
                assert select_dialogues.isdisjoint(calibrate_dialogues)
                for modality in ("text", "audio", "fusion"):
                    matrix = _feature_matrix(train, modality)
                    mean, scale = matrix.mean(0), matrix.std(0)
                    scale[scale < 1e-6] = 1
                    for loss in ("none", "inverse_sqrt"):
                        for seed in (13, 42, 2026):
                            path = Path(f"checkpoints/day2/{modality}_{loss}_{seed}.pt")
                            head = torch.load(path, weights_only=True, map_location="cpu")
                            assert head["train_examples"] == len(matrix)
                            assert head["calibration_clip_ids"] == result["calibration_clip_ids"]
                            assert head["labels"] == list(EMOTIONS)
                            assert np.array_equal(head["normalization_mean"].numpy(), mean)
                            assert np.array_equal(head["normalization_scale"].numpy(), scale)
                deployed = InferencePipeline(
                    Path("checkpoints/day2/deployed.pt"), local_files_only=True
                )
                assert deployed.artifact["seed"] == 42
                assert deployed.artifact["temperature"] == result["temperature"] > 0
                probabilities = deployed.predict_features(dev["text"][0], dev["audio"][0])
                assert probabilities.shape == (7,) and np.isfinite(probabilities).all()
                assert np.all(probabilities >= 0) and np.isclose(probabilities.sum(), 1)
                clip = load_manifest(root / "processed/dev.jsonl")[0]
                fresh = deployed.predict(Path(clip.audio_path), text=clip.text)
                assert np.allclose(list(fresh.probabilities.values()), probabilities, atol=1e-5)
                transcribed = deployed.predict(Path(clip.audio_path))
                assert transcribed.transcript_source == "asr"
                ledger = {
                    "asr": WHISPER_TINY_EN_PARAMETERS,
                    "text_encoder": sum(
                        p.numel() for p in deployed.extractor._text_model.parameters()
                    ),
                    "audio_encoder": sum(
                        p.numel() for p in deployed.extractor._audio_model.parameters()
                    ),
                    "loaded_heads": sum(p.numel() for p in deployed.model.parameters()),
                    "calibration_temperature": 1,
                }
                assert sum(ledger.values()) < 6_000_000_000
                report["learned_parameters"] = ledger
                report["total_learned_parameters"] = sum(ledger.values())
                report["real_clip"] = {
                    "clip_id": clip.clip_id,
                    "supplied_emotion": fresh.emotion,
                    "asr_emotion": transcribed.emotion,
                    "asr_transcript": transcribed.transcript,
                    "offline": True,
                    "cached_fresh_agreement": True,
                }
            assert len(result["comparisons"]) == 9
            report["passed"].append(
                "All 18 heads; training-only normalization; disjoint calibration dialogues; "
                "seed-42 calibrated deployment with normalized probabilities"
            )
        except (AssertionError, OSError, KeyError, ValueError) as error:
            report["failures"].append(f"Experiment: {error}")
    report["complete"] = not report["pending"] and not report["failures"]
    Path("reports/day2_completeness.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
