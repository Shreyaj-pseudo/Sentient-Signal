"""Compare an IEMOCAP-trained audio representation with the Day 2 baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from sentient_signals.audio import read_pcm16_wav
from sentient_signals.calibration import (
    apply_temperature,
    dialogue_calibration_mask,
    fuse_probabilities,
)
from sentient_signals.data import load_manifest
from sentient_signals.features import FeatureExtractor
from sentient_signals.model import FusionHead, _feature_matrix, fit_head

SEEDS = (13, 42, 2026)
LOSSES = ("none", "inverse_sqrt")
MODALITIES = ("audio", "fusion")
ALPHAS = (0.25, 0.5, 0.75)


def cache_split(split: str, extractor: FeatureExtractor) -> Path:
    items = load_manifest(Path("data/day2/processed") / f"{split}.jsonl")
    baseline_path = Path("data/day2/features") / f"{split}.npz"
    with np.load(baseline_path, allow_pickle=False) as baseline:
        text = baseline["text"].copy()
        clip_ids = baseline["clip_ids"].astype(str)
    if clip_ids.tolist() != [item.clip_id for item in items]:
        raise ValueError(f"{split} manifest does not match the Day 2 text cache")

    shard_dir = (
        Path("data/audio_encoder/shards")
        / extractor.audio_config["audio_revision"][:12]
        / split
    )
    shard_dir.mkdir(parents=True, exist_ok=True)
    audio = []
    for index, item in enumerate(items, 1):
        shard = shard_dir / f"{item.clip_id}.npy"
        if shard.exists():
            vector = np.load(shard, allow_pickle=False)
        else:
            vector = extractor.encode_audio(read_pcm16_wav(Path(item.audio_path)))
            with shard.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, vector)
            shard.with_suffix(".tmp").replace(shard)
        if vector.shape != (extractor.audio_dimension,) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid emotion-audio feature: {split}/{item.clip_id}")
        audio.append(vector)
        if index % 100 == 0 or index == len(items):
            print(f"{split}: cached {index}/{len(items)}", flush=True)

    output = Path("data/audio_encoder") / f"{split}.npz"
    np.savez_compressed(
        output,
        text=text,
        audio=np.stack(audio),
        labels=np.asarray([item.label for item in items]),
        clip_ids=clip_ids,
        feature_mode=np.asarray("encoders"),
        feature_config=np.asarray(json.dumps(extractor.audio_config, sort_keys=True)),
    )
    return output


def predict(checkpoint: Path, features: dict[str, np.ndarray]) -> np.ndarray:
    artifact = torch.load(checkpoint, map_location="cpu", weights_only=True)
    matrix = _feature_matrix(features, artifact["modality"])
    matrix = (matrix - artifact["normalization_mean"].numpy()) / artifact[
        "normalization_scale"
    ].numpy()
    model = FusionHead(matrix.shape[1])
    model.load_state_dict(artifact["model_state"])
    model.eval()
    with torch.inference_mode():
        return apply_temperature(model(torch.from_numpy(matrix)).numpy(), 1.0)


def macro_f1(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            probabilities.argmax(axis=1),
            labels=list(range(7)),
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--compare-only", action="store_true")
    mode.add_argument("--probability-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    extractor = FeatureExtractor(audio_encoder="emotion")
    train = Path("data/audio_encoder/train.npz")
    dev = Path("data/audio_encoder/dev.npz")
    if not args.compare_only and not args.probability_only:
        train = cache_split("train", extractor)
        dev = cache_split("dev", extractor)
    if not train.exists() or not dev.exists():
        raise FileNotFoundError("Run without --compare-only to build the feature caches first")

    checkpoint_dir = Path("checkpoints/audio_encoder")
    output = Path("reports/audio_encoder_experiment.json")
    scores: dict[str, list[float]] = {}
    if args.probability_only:
        previous = json.loads(output.read_text(encoding="utf-8"))
        scores = {
            name: values["per_seed"]
            for name, values in previous["scores"].items()
            if not name.startswith("probability_")
        }
    else:
        for modality in MODALITIES:
            for loss in LOSSES:
                name = f"{modality}_{loss}"
                scores[name] = []
                for seed in SEEDS:
                    result = fit_head(
                        train,
                        dev,
                        checkpoint_dir / f"{name}_{seed}.pt",
                        seed=seed,
                        modality=modality,
                        class_weighting=loss,
                    )
                    scores[name].append(float(result["dev_macro_f1"]))

    baseline = json.loads(Path("reports/day2_experiment.json").read_text(encoding="utf-8"))
    best_audio_loss = max(LOSSES, key=lambda loss: np.mean(scores[f"audio_{loss}"]))
    text_loss = baseline["best_losses"]["text"]
    with np.load(dev, allow_pickle=False) as cached:
        features = {name: cached[name].copy() for name in cached.files}
    selection = ~dialogue_calibration_mask(features["clip_ids"])
    labels = features["labels"][selection]
    for alpha in ALPHAS:
        scores[f"probability_{alpha}"] = []
    for seed in SEEDS:
        text = predict(Path("checkpoints/day2") / f"text_{text_loss}_{seed}.pt", features)
        audio = predict(
            checkpoint_dir / f"audio_{best_audio_loss}_{seed}.pt",
            features,
        )
        for alpha in ALPHAS:
            fused = fuse_probabilities(text, audio, alpha)
            scores[f"probability_{alpha}"].append(macro_f1(labels, fused[selection]))

    report = {
        "audio_config": extractor.audio_config,
        "scores": {
            name: {
                "per_seed": values,
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)),
            }
            for name, values in scores.items()
        },
        "day2_text_baseline_macro_f1": baseline["comparisons"]["text_inverse_sqrt"]["mean"],
        "day2_audio_baseline_macro_f1": baseline["comparisons"]["audio_inverse_sqrt"]["mean"],
        "day2_selected_fusion_macro_f1": baseline["comparisons"]["probability_0.75"]["mean"],
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
