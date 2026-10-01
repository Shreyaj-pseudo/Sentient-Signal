"""Resumable full-split feature caching and development-only model selection.

Run from the repository root with .venv/Scripts/python scripts/day2_experiment.py.
Preparation must have completed first. Test features are cached only after selection.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from sklearn.metrics import accuracy_score, f1_score, log_loss

from sentient_signals.calibration import (
    apply_temperature,
    dialogue_calibration_mask,
    fit_temperature,
    fuse_probabilities,
)
from sentient_signals.constants import EMOTIONS
from sentient_signals.data import load_manifest
from sentient_signals.features import AUDIO_REVISION, TEXT_REVISION, FeatureExtractor
from sentient_signals.model import FusionHead, _feature_matrix, fit_head

ROOT = Path("data/day2")
REPORT = Path("reports/day2_experiment.json")
CHECKPOINTS = Path("checkpoints/day2")
SEEDS = (13, 42, 2026)
LOSSES = ("none", "inverse_sqrt")
MODALITIES = ("text", "audio", "fusion")
FEATURE_CONFIG = {
    "text_revision": TEXT_REVISION,
    "audio_revision": AUDIO_REVISION,
    "preprocessing": "pcm16_16000_mono_equal_windows_160000_mean_population_std_pad400_v1",
    "normalized_text_embeddings": True,
}


def save_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def cache_split(split: str, extractor: FeatureExtractor) -> Path:
    manifest = ROOT / "processed" / f"{split}.jsonl"
    deadline = time.monotonic() + 7200
    while not manifest.exists():
        if time.monotonic() > deadline:
            raise TimeoutError(f"Preparation did not produce {manifest} within two hours")
        print(f"Waiting for preparation: {manifest}", flush=True)
        time.sleep(30)
    items = load_manifest(manifest)
    # The raw train clip exists but PyAV cannot open it (reproduced independently).
    # Development dia110_utt7 is absent from the distributed media.
    expected = {"train": 9988, "dev": 1108, "test": 2610}[split]
    if len(items) != expected or any(item.split != split for item in items):
        raise ValueError(f"Expected complete {split} manifest with {expected} matched clips")
    if split == "train" and any(item.clip_id == "dia125_utt3" for item in items):
        raise ValueError("Unexpected manifest: documented corrupt training clip is present")
    directory = ROOT / "features" / split
    directory.mkdir(parents=True, exist_ok=True)
    texts, audios = [], []
    started = time.perf_counter()
    for index, item in enumerate(items):
        wav = Path(item.audio_path)
        signature = hashlib.sha256(
            wav.read_bytes()
            + item.text.encode()
            + json.dumps(FEATURE_CONFIG, sort_keys=True).encode()
        ).hexdigest()
        shard = directory / f"{item.clip_id}_{signature}.npz"
        if shard.exists():
            with np.load(shard, allow_pickle=False) as cached:
                text, audio = cached["text"].copy(), cached["audio"].copy()
        else:
            text, audio, _ = extractor.encode(item.text, wav)
            # Stop on extraction failure so exclusions cannot silently alter the experiment.
            with shard.with_suffix(".tmp").open("wb") as handle:
                np.savez_compressed(handle, text=text, audio=audio)
            shard.with_suffix(".tmp").replace(shard)
        if text.shape != (384,) or audio.shape != (1536,):
            raise ValueError(f"Invalid feature dimensions: {split}/{item.clip_id}")
        if not np.isfinite(text).all() or not np.isfinite(audio).all():
            raise ValueError(f"Non-finite features: {split}/{item.clip_id}")
        texts.append(text)
        audios.append(audio)
        if (index + 1) % 100 == 0 or index + 1 == len(items):
            progress = {
                "stage": "cache",
                "split": split,
                "done": index + 1,
                "total": len(items),
                "elapsed_seconds": time.perf_counter() - started,
                "rss_bytes": psutil.Process().memory_info().rss,
            }
            save_json(Path("reports/day2_live_status.json"), progress)
            print(json.dumps(progress), flush=True)
    output = ROOT / "features" / f"{split}.npz"
    np.savez_compressed(
        output,
        text=np.stack(texts),
        audio=np.stack(audios),
        labels=np.asarray([item.label for item in items]),
        clip_ids=np.asarray([item.clip_id for item in items]),
        feature_mode=np.asarray("encoders"),
        feature_config=np.asarray(json.dumps(FEATURE_CONFIG, sort_keys=True)),
    )
    return output


def logits(artifact: dict, features) -> np.ndarray:
    matrix = _feature_matrix(features, artifact["modality"])
    matrix = (matrix - artifact["normalization_mean"].numpy()) / artifact[
        "normalization_scale"
    ].numpy()
    model = FusionHead(matrix.shape[1])
    model.load_state_dict(artifact["model_state"])
    model.eval()
    with torch.inference_mode():
        return model(torch.from_numpy(matrix)).numpy()


def macro(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            probabilities.argmax(axis=1),
            labels=list(range(7)),
            average="macro",
            zero_division=0,
        )
    )


def compare(train: Path, dev: Path) -> dict:
    CHECKPOINTS.mkdir(parents=True, exist_ok=True)
    with np.load(dev, allow_pickle=False) as data:
        features = {key: data[key].copy() for key in data.files}
    calibration = dialogue_calibration_mask(features["clip_ids"])
    selection = ~calibration
    labels = features["labels"]
    scores, predictions, artifacts = {}, {}, {}
    training_started = time.perf_counter()
    for modality in MODALITIES:
        for loss in LOSSES:
            key = f"{modality}_{loss}"
            scores[key] = []
            for seed in SEEDS:
                name = f"{key}_{seed}"
                path = CHECKPOINTS / f"{name}.pt"
                print(f"Training {name}", flush=True)
                fit_head(train, dev, path, seed=seed, modality=modality, class_weighting=loss)
                artifact = torch.load(path, map_location="cpu", weights_only=True)
                artifact["artifact_kind"] = f"day2_full_{modality}"
                artifact["labels"] = list(EMOTIONS)
                artifact["feature_config"] = FEATURE_CONFIG
                torch.save(artifact, path)
                artifacts[name] = artifact
                predictions[name] = apply_temperature(logits(artifact, features), 1.0)
                scores[key].append(macro(labels[selection], predictions[name][selection]))
                save_json(
                    Path("reports/day2_live_status.json"),
                    {
                        "stage": "training",
                        "completed": len(artifacts),
                        "total": 18,
                        "last_model": name,
                        "selection_macro_f1": scores[key][-1],
                    },
                )
    best_losses = {
        modality: max(LOSSES, key=lambda loss: np.mean(scores[f"{modality}_{loss}"]))
        for modality in MODALITIES
    }
    for alpha in (0.25, 0.5, 0.75):
        key = f"probability_{alpha}"
        scores[key] = []
        for seed in SEEDS:
            combined = fuse_probabilities(
                predictions[f"text_{best_losses['text']}_{seed}"],
                predictions[f"audio_{best_losses['audio']}_{seed}"],
                alpha,
            )
            predictions[f"{key}_{seed}"] = combined
            scores[key].append(macro(labels[selection], combined[selection]))
    feature_key = f"fusion_{best_losses['fusion']}"
    probability_key = max(
        (f"probability_{alpha}" for alpha in (0.25, 0.5, 0.75)),
        key=lambda key: np.mean(scores[key]),
    )
    selected = (
        probability_key
        if np.mean(scores[probability_key]) - np.mean(scores[feature_key]) >= 0.005
        else feature_key
    )
    if selected.startswith("probability"):
        deploy = {
            "artifact_kind": "day2_full_probability_fusion",
            "modality": "probability",
            "feature_mode": "encoders",
            "text_dimension": 384,
            "audio_dimension": 1536,
            "alpha": float(selected.split("_")[1]),
            "text_head": artifacts[f"text_{best_losses['text']}_42"],
            "audio_head": artifacts[f"audio_{best_losses['audio']}_42"],
        }
        raw = np.log(np.maximum(predictions[f"{selected}_42"], 1e-12))
    else:
        deploy = artifacts[f"{selected}_42"].copy()
        raw = logits(deploy, features)
    temperature = fit_temperature(raw[calibration], labels[calibration])
    calibrated = apply_temperature(raw[calibration], temperature)
    deploy.update(
        {
            "temperature": temperature,
            "labels": list(EMOTIONS),
            "feature_config": FEATURE_CONFIG,
            "seed": 42,
            "calibration_clip_ids": features["clip_ids"][calibration].tolist(),
        }
    )
    torch.save(deploy, CHECKPOINTS / "deployed.pt")
    report = {
        "seeds": list(SEEDS),
        "feature_config": FEATURE_CONFIG,
        "train_examples": 9988,
        "exclusions": [
            {"split": "train", "clip": "dia125_utt3", "reason": "PyAV InvalidDataError"},
            {"split": "dev", "clip": "dia110_utt7", "reason": "Missing source media"},
        ],
        "selection_examples": int(selection.sum()),
        "calibration_examples": int(calibration.sum()),
        "selection_clip_ids": features["clip_ids"][selection].tolist(),
        "calibration_clip_ids": features["clip_ids"][calibration].tolist(),
        "comparisons": {
            key: {
                "per_seed": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for key, values in scores.items()
        },
        "best_losses": best_losses,
        "selected": selected,
        "tie_rule": "Prefer feature fusion for probability-fusion improvement below 0.005",
        "temperature": temperature,
        "calibration_nll_before": float(
            log_loss(
                labels[calibration], apply_temperature(raw[calibration], 1), labels=list(range(7))
            )
        ),
        "calibration_nll_after": float(
            log_loss(labels[calibration], calibrated, labels=list(range(7)))
        ),
        "calibration_accuracy": float(accuracy_score(labels[calibration], calibrated.argmax(1))),
        "training_seconds": time.perf_counter() - training_started,
        "test_metrics": "Not evaluated; reserved for Day 3",
    }
    save_json(REPORT, report)
    print(f"Selected {selected}; temperature={temperature:.4f}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    extractor = FeatureExtractor(local_files_only=True)
    train, dev = ROOT / "features/train.npz", ROOT / "features/dev.npz"
    if not args.compare_only:
        train = cache_split("train", extractor)
        dev = cache_split("dev", extractor)
    report = compare(train, dev)
    if not args.compare_only:
        cache_split("test", extractor)
    report.update(
        {
            "wall_seconds": time.perf_counter() - started,
            "platform": platform.platform(),
            "threads": args.threads,
            "rss_bytes_at_end": psutil.Process().memory_info().rss,
        }
    )
    save_json(REPORT, report)
    save_json(Path("reports/day2_live_status.json"), {"stage": "complete"})


if __name__ == "__main__":
    main()
