"""Train modality-aware fusion on cached text and wav2vec-plus-prosody features."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn

from sentient_signals.calibration import dialogue_calibration_mask
from sentient_signals.constants import EMOTIONS
from sentient_signals.model import GatedFusionHead

SEEDS = (13, 42, 2026)


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


def predict(
    model: GatedFusionHead, text: torch.Tensor, audio: torch.Tensor
) -> tuple[np.ndarray, np.ndarray]:
    probabilities, gates = [], []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(text), 128):
            log_probabilities, _text, _audio, gate = model(
                text[start : start + 128], audio[start : start + 128]
            )
            probabilities.append(log_probabilities.exp().numpy())
            gates.append(gate[:, 0].numpy())
    return np.concatenate(probabilities), np.concatenate(gates)


def train(seed: int) -> tuple[float, np.ndarray]:
    with np.load("data/prosody/train.npz", allow_pickle=False) as data:
        train_text = data["text"].astype(np.float32)
        train_audio = data["audio"].astype(np.float32)
        train_labels = data["labels"].astype(np.int64)
    with np.load("data/prosody/dev.npz", allow_pickle=False) as data:
        dev_text = data["text"].astype(np.float32)
        dev_audio = data["audio"].astype(np.float32)
        dev_labels = data["labels"].astype(np.int64)
        selection = ~dialogue_calibration_mask(data["clip_ids"])

    text_mean, text_scale = train_text.mean(0), train_text.std(0)
    audio_mean, audio_scale = train_audio.mean(0), train_audio.std(0)
    text_scale[text_scale < 1e-6] = 1
    audio_scale[audio_scale < 1e-6] = 1
    train_text = torch.from_numpy((train_text - text_mean) / text_scale)
    train_audio = torch.from_numpy((train_audio - audio_mean) / audio_scale)
    dev_text = torch.from_numpy((dev_text - text_mean) / text_scale)
    dev_audio = torch.from_numpy((dev_audio - audio_mean) / audio_scale)
    labels = torch.from_numpy(train_labels)

    torch.manual_seed(seed)
    model = GatedFusionHead(train_text.shape[1], train_audio.shape[1])
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    counts = np.bincount(train_labels, minlength=len(EMOTIONS)).astype(np.float32)
    weights = 1 / np.sqrt(np.maximum(counts, 1))
    weights = torch.from_numpy(weights / weights.mean())
    fused_loss = nn.NLLLoss(weight=weights)
    branch_loss = nn.CrossEntropyLoss(weight=weights)
    rng = np.random.default_rng(seed)
    best_score, best_state, stale = -1.0, None, 0

    for _epoch in range(30):
        model.train()
        order = rng.permutation(len(labels))
        for start in range(0, len(order), 64):
            indices = order[start : start + 64]
            log_probabilities, text_logits, audio_logits, _gate = model(
                train_text[indices], train_audio[indices]
            )
            loss = fused_loss(log_probabilities, labels[indices]) + 0.2 * (
                branch_loss(text_logits, labels[indices])
                + branch_loss(audio_logits, labels[indices])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

        probabilities, _gates = predict(model, dev_text, dev_audio)
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
    _probabilities, gates = predict(model, dev_text, dev_audio)
    artifact = {
        "artifact_kind": "prosody_gated_fusion_experiment",
        "text_dimension": train_text.shape[1],
        "audio_dimension": train_audio.shape[1],
        "text_mean": torch.from_numpy(text_mean),
        "text_scale": torch.from_numpy(text_scale),
        "audio_mean": torch.from_numpy(audio_mean),
        "audio_scale": torch.from_numpy(audio_scale),
        "model_state": best_state,
        "seed": seed,
        "dev_macro_f1": best_score,
    }
    output = Path("checkpoints/gated_fusion") / f"seed_{seed}.pt"
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(artifact, output)
    return best_score, gates[selection]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    if not Path("data/prosody/train.npz").exists() or not Path("data/prosody/dev.npz").exists():
        raise FileNotFoundError("Run scripts/prosody_experiment.py first")

    scores, gates = [], []
    for seed in SEEDS:
        score, seed_gates = train(seed)
        scores.append(score)
        gates.append(seed_gates)
        print(f"seed={seed} dev_macro_f1={score:.4f}", flush=True)

    all_gates = np.concatenate(gates)
    report = {
        "model": "text_led_dynamic_probability_gate_with_prosody_audio",
        "seeds": list(SEEDS),
        "macro_f1": {
            "per_seed": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
        },
        "text_weight": {
            "mean": float(all_gates.mean()),
            "p10": float(np.percentile(all_gates, 10)),
            "p90": float(np.percentile(all_gates, 90)),
        },
        "day3_baseline": 0.42644645224791217,
        "test_metrics": "Not evaluated",
    }
    output = Path("reports/gated_fusion_experiment.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
