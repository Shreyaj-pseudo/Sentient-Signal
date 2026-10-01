"""Train temporal attention over frozen wav2vec2 features on development data only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from sentient_signals.audio import read_pcm16_wav
from sentient_signals.calibration import dialogue_calibration_mask, fuse_probabilities
from sentient_signals.constants import EMOTIONS
from sentient_signals.data import load_manifest
from sentient_signals.features import FeatureExtractor, temporal_audio_bins
from sentient_signals.model import AttentiveAudioHead, FusionHead

BINS = 32
SEEDS = (13, 42, 2026)


def cache_split(split: str, extractor: FeatureExtractor) -> Path:
    items = load_manifest(Path("data/day2/processed") / f"{split}.jsonl")
    shard_dir = Path("data/attentive_audio/shards") / split
    shard_dir.mkdir(parents=True, exist_ok=True)
    sequences, masks = [], []
    for index, item in enumerate(items, 1):
        shard = shard_dir / f"{item.clip_id}.npz"
        if shard.exists():
            with np.load(shard, allow_pickle=False) as cached:
                sequence, mask = cached["sequence"], cached["mask"]
        else:
            frames = extractor.encode_audio_frames(read_pcm16_wav(Path(item.audio_path)))
            sequence, mask = temporal_audio_bins(frames, BINS)
            with shard.with_suffix(".tmp").open("wb") as handle:
                np.savez_compressed(handle, sequence=sequence.astype(np.float16), mask=mask)
            shard.with_suffix(".tmp").replace(shard)
        if sequence.shape != (BINS, extractor.audio_frame_dimension) or mask.shape != (BINS,):
            raise ValueError(f"Invalid temporal features: {split}/{item.clip_id}")
        sequences.append(sequence)
        masks.append(mask)
        if index % 100 == 0 or index == len(items):
            print(f"{split}: cached {index}/{len(items)}", flush=True)

    output = Path("data/attentive_audio") / f"{split}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        audio=np.stack(sequences).astype(np.float16),
        mask=np.stack(masks),
        labels=np.asarray([item.label for item in items]),
        clip_ids=np.asarray([item.clip_id for item in items]),
    )
    return output


def normalization(frames: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = int(mask.sum())
    mean = np.sum(frames, axis=(0, 1), where=mask[..., None], dtype=np.float64) / count
    square_mean = (
        np.sum(np.square(frames, dtype=np.float32), axis=(0, 1), where=mask[..., None]) / count
    )
    scale = np.sqrt(np.maximum(square_mean - mean**2, 0)).astype(np.float32)
    scale[scale < 1e-6] = 1
    return mean.astype(np.float32), scale


def predict(
    model: AttentiveAudioHead,
    frames: np.ndarray,
    mask: np.ndarray,
    mean: torch.Tensor,
    scale: torch.Tensor,
) -> np.ndarray:
    outputs = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(frames), 128):
            batch = torch.from_numpy(frames[start : start + 128].astype(np.float32))
            batch = (batch - mean) / scale
            valid = torch.from_numpy(mask[start : start + 128])
            outputs.append(torch.softmax(model(batch, valid), dim=1).numpy())
    return np.concatenate(outputs)


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


def train(train_path: Path, dev_path: Path, seed: int) -> tuple[dict, np.ndarray, np.ndarray]:
    with np.load(train_path, allow_pickle=False) as data:
        train_frames = data["audio"].copy()
        train_mask = data["mask"].copy()
        train_labels = data["labels"].astype(np.int64)
    with np.load(dev_path, allow_pickle=False) as data:
        dev_frames = data["audio"].copy()
        dev_mask = data["mask"].copy()
        dev_labels = data["labels"].astype(np.int64)
        selection = ~dialogue_calibration_mask(data["clip_ids"])

    torch.manual_seed(seed)
    mean_array, scale_array = normalization(train_frames, train_mask)
    mean, scale = torch.from_numpy(mean_array), torch.from_numpy(scale_array)
    model = AttentiveAudioHead(train_frames.shape[2])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train_labels, minlength=len(EMOTIONS)).astype(np.float32)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(weights / weights.mean()))
    rng = np.random.default_rng(seed)
    best_score, best_state, stale = -1.0, None, 0

    for _epoch in range(30):
        model.train()
        order = rng.permutation(len(train_labels))
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            frames = torch.from_numpy(train_frames[indices].astype(np.float32))
            frames = (frames - mean) / scale
            mask = torch.from_numpy(train_mask[indices])
            labels = torch.from_numpy(train_labels[indices])
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(frames, mask), labels)
            loss.backward()
            optimizer.step()
        probabilities = predict(model, dev_frames, dev_mask, mean, scale)
        score = macro(dev_labels[selection], probabilities[selection])
        if score > best_score:
            best_score = score
            best_state = {key: value.cpu().clone() for key, value in model.state_dict().items()}
            stale = 0
        else:
            stale += 1
            if stale == 5:
                break

    model.load_state_dict(best_state)
    probabilities = predict(model, dev_frames, dev_mask, mean, scale)
    artifact = {
        "artifact_kind": "attentive_audio_experiment",
        "frame_dimension": train_frames.shape[2],
        "bins": BINS,
        "normalization_mean": mean,
        "normalization_scale": scale,
        "model_state": best_state,
        "seed": seed,
        "dev_macro_f1": best_score,
    }
    checkpoint = Path("checkpoints/attentive_audio") / f"audio_inverse_sqrt_{seed}.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, checkpoint)
    return artifact, probabilities, selection


def text_probabilities(seed: int, features: dict[str, np.ndarray]) -> np.ndarray:
    artifact = torch.load(
        Path("checkpoints/day2") / f"text_inverse_sqrt_{seed}.pt",
        map_location="cpu",
        weights_only=True,
    )
    values = features["text"].astype(np.float32)
    values = (values - artifact["normalization_mean"].numpy()) / artifact[
        "normalization_scale"
    ].numpy()
    model = FusionHead(values.shape[1])
    model.load_state_dict(artifact["model_state"])
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(torch.from_numpy(values)), dim=1).numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)

    train_path = Path("data/attentive_audio/train.npz")
    dev_path = Path("data/attentive_audio/dev.npz")
    if not args.compare_only:
        extractor = FeatureExtractor(local_files_only=True)
        train_path = cache_split("train", extractor)
        dev_path = cache_split("dev", extractor)
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError("Run without --compare-only to cache temporal features first")

    with np.load("data/day2/features/dev.npz", allow_pickle=False) as data:
        day2_dev = {key: data[key].copy() for key in data.files}
    with np.load(dev_path, allow_pickle=False) as data:
        if data["clip_ids"].astype(str).tolist() != day2_dev["clip_ids"].astype(str).tolist():
            raise ValueError("Attentive and Day 2 development caches do not align")
    labels = day2_dev["labels"]
    audio_scores, fusion_scores = [], {weight: [] for weight in (0.75, 0.8, 0.85)}
    for seed in SEEDS:
        _artifact, audio, selection = train(train_path, dev_path, seed)
        text = text_probabilities(seed, day2_dev)
        audio_scores.append(macro(labels[selection], audio[selection]))
        for weight in fusion_scores:
            fused = fuse_probabilities(text, audio, weight)
            fusion_scores[weight].append(macro(labels[selection], fused[selection]))

    baseline = json.loads(Path("reports/day2_experiment.json").read_text(encoding="utf-8"))
    report = {
        "pooling": "single_head_attentive_mean_population_std_over_32_temporal_bins",
        "encoder": "facebook/wav2vec2-base",
        "seeds": list(SEEDS),
        "audio_macro_f1": {"per_seed": audio_scores, "mean": float(np.mean(audio_scores))},
        "fusion_macro_f1": {
            str(weight): {"per_seed": scores, "mean": float(np.mean(scores))}
            for weight, scores in fusion_scores.items()
        },
        "mean_std_audio_baseline": baseline["comparisons"]["audio_inverse_sqrt"]["mean"],
        "day3_fusion_baseline": 0.42644645224791217,
        "test_metrics": "Not evaluated",
    }
    output = Path("reports/attentive_audio_experiment.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
