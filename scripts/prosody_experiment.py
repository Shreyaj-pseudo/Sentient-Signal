"""Add explicit prosody to cached wav2vec features and compare on development data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from sentient_signals.audio import read_pcm16_wav
from sentient_signals.calibration import dialogue_calibration_mask, fuse_probabilities
from sentient_signals.constants import EMOTIONS
from sentient_signals.data import load_manifest
from sentient_signals.features import PROSODY_FEATURES, prosody_features
from sentient_signals.model import FusionHead, _feature_matrix, fit_head

SEEDS = (13, 42, 2026)
MODALITIES = ("audio", "fusion")
WEIGHTS = (0.75, 0.8, 0.85)


def cache_split(split: str) -> Path:
    items = load_manifest(Path("data/day2/processed") / f"{split}.jsonl")
    with np.load(Path("data/day2/features") / f"{split}.npz", allow_pickle=False) as data:
        baseline = {key: data[key].copy() for key in data.files}
    if baseline["clip_ids"].astype(str).tolist() != [item.clip_id for item in items]:
        raise ValueError(f"{split} manifest does not match the Day 2 feature cache")

    shard_dir = Path("data/prosody/shards") / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    vectors = []
    for index, item in enumerate(items, 1):
        shard = shard_dir / f"{item.clip_id}.npy"
        if shard.exists():
            vector = np.load(shard, allow_pickle=False)
        else:
            vector = prosody_features(read_pcm16_wav(Path(item.audio_path)))
            with shard.with_suffix(".tmp").open("wb") as handle:
                np.save(handle, vector)
            shard.with_suffix(".tmp").replace(shard)
        if vector.shape != (len(PROSODY_FEATURES),) or not np.isfinite(vector).all():
            raise ValueError(f"Invalid prosody features: {split}/{item.clip_id}")
        vectors.append(vector)
        if index % 500 == 0 or index == len(items):
            print(f"{split}: cached {index}/{len(items)}", flush=True)

    output = Path("data/prosody") / f"{split}.npz"
    np.savez_compressed(
        output,
        text=baseline["text"],
        audio=np.concatenate([baseline["audio"], np.stack(vectors)], axis=1),
        labels=baseline["labels"],
        clip_ids=baseline["clip_ids"],
        feature_mode=np.asarray("encoders"),
        feature_config=np.asarray(
            json.dumps(
                {
                    "base": "day2_wav2vec2_mean_population_std",
                    "prosody_features": list(PROSODY_FEATURES),
                },
                sort_keys=True,
            )
        ),
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
        return torch.softmax(model(torch.from_numpy(matrix)), dim=1).numpy()


def macro(labels: np.ndarray, probabilities: np.ndarray) -> float:
    return float(
        f1_score(
            labels,
            probabilities.argmax(1),
            labels=list(range(len(EMOTIONS))),
            average="macro",
            zero_division=0,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    train_path = Path("data/prosody/train.npz")
    dev_path = Path("data/prosody/dev.npz")
    if not args.compare_only:
        train_path, dev_path = cache_split("train"), cache_split("dev")
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("Run without --compare-only to cache prosody first")

    checkpoint_dir = Path("checkpoints/prosody")
    scores: dict[str, list[float]] = {modality: [] for modality in MODALITIES}
    for modality in MODALITIES:
        for seed in SEEDS:
            result = fit_head(
                train_path,
                dev_path,
                checkpoint_dir / f"{modality}_inverse_sqrt_{seed}.pt",
                seed=seed,
                modality=modality,
                class_weighting="inverse_sqrt",
            )
            scores[modality].append(float(result["dev_macro_f1"]))

    with np.load(dev_path, allow_pickle=False) as data:
        features = {key: data[key].copy() for key in data.files}
    selection = ~dialogue_calibration_mask(features["clip_ids"])
    labels = features["labels"][selection]
    fusion_scores = {weight: [] for weight in WEIGHTS}
    for seed in SEEDS:
        text = predict(Path("checkpoints/day2") / f"text_inverse_sqrt_{seed}.pt", features)
        audio = predict(checkpoint_dir / f"audio_inverse_sqrt_{seed}.pt", features)
        for weight in WEIGHTS:
            fusion_scores[weight].append(
                macro(labels, fuse_probabilities(text, audio, weight)[selection])
            )

    report = {
        "prosody_features": list(PROSODY_FEATURES),
        "seeds": list(SEEDS),
        "scores": {
            modality: {"per_seed": values, "mean": float(np.mean(values))}
            for modality, values in scores.items()
        },
        "probability_fusion": {
            str(weight): {"per_seed": values, "mean": float(np.mean(values))}
            for weight, values in fusion_scores.items()
        },
        "audio_baseline": 0.24243250441399014,
        "day3_fusion_baseline": 0.42644645224791217,
        "test_metrics": "Not evaluated",
    }
    output = Path("reports/prosody_experiment.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
