"""One-shot final evaluation of the frozen Day 3 checkpoint on MELD test."""

from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from sentient_signals.calibration import apply_temperature, fuse_probabilities
from sentient_signals.constants import EMOTIONS
from sentient_signals.model import FusionHead

SEEDS = (13, 42, 2026)
BOOTSTRAPS = 2_000


def head_probabilities(artifact: dict, modality: str, features: np.ndarray) -> np.ndarray:
    values = features.astype(np.float32)
    values = (values - artifact["normalization_mean"].numpy()) / artifact[
        "normalization_scale"
    ].numpy()
    model = FusionHead(values.shape[1])
    model.load_state_dict(artifact["model_state"])
    model.eval()
    with torch.inference_mode():
        return torch.softmax(model(torch.from_numpy(values)), dim=1).numpy()


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


def dialogue_bootstrap(
    labels: np.ndarray,
    fused: np.ndarray,
    text: np.ndarray,
    clip_ids: np.ndarray,
    samples: int = BOOTSTRAPS,
) -> dict[str, float]:
    dialogue_ids = np.asarray(
        [re.fullmatch(r"(dia\d+)_utt\d+", value).group(1) for value in clip_ids.astype(str)]
    )
    groups = [np.flatnonzero(dialogue_ids == value) for value in np.unique(dialogue_ids)]
    rng = np.random.default_rng(42)
    differences = []
    for _ in range(samples):
        indices = np.concatenate(
            [groups[index] for index in rng.integers(0, len(groups), len(groups))]
        )
        differences.append(
            macro(labels[indices], fused[indices]) - macro(labels[indices], text[indices])
        )
    return {
        "mean_difference": float(np.mean(differences)),
        "ci95_low": float(np.percentile(differences, 2.5)),
        "ci95_high": float(np.percentile(differences, 97.5)),
        "probability_fused_better": float(np.mean(np.asarray(differences) > 0)),
        "samples": samples,
    }


def metric_report(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    predicted = probabilities.argmax(1)
    per_class = classification_report(
        labels,
        predicted,
        labels=list(range(len(EMOTIONS))),
        target_names=list(EMOTIONS),
        output_dict=True,
        zero_division=0,
    )
    return {
        "macro_f1": macro(labels, probabilities),
        "weighted_f1": float(f1_score(labels, predicted, average="weighted", zero_division=0)),
        "accuracy": float(accuracy_score(labels, predicted)),
        "per_class": {
            emotion: {
                key: float(per_class[emotion][key])
                for key in ("precision", "recall", "f1-score", "support")
            }
            for emotion in EMOTIONS
        },
        "confusion_matrix": confusion_matrix(
            labels, predicted, labels=list(range(len(EMOTIONS)))
        ).tolist(),
    }


def main() -> None:
    checkpoint_path = Path("checkpoints/day3_deployed.pt")
    feature_path = Path("data/day2/features/test.npz")
    if not checkpoint_path.exists() or not feature_path.exists():
        raise FileNotFoundError("Day 3 checkpoint and complete test feature cache are required")
    artifact = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    with np.load(feature_path, allow_pickle=False) as data:
        features = {key: data[key].copy() for key in data.files}
    labels = features["labels"].astype(np.int64)
    if len(labels) != 2_610 or len(np.unique(features["clip_ids"])) != 2_610:
        raise ValueError("Expected the complete official 2,610-clip MELD test split")

    text = head_probabilities(artifact["text_head"], "text", features["text"])
    audio = head_probabilities(artifact["audio_head"], "audio", features["audio"])
    uncalibrated = fuse_probabilities(text, audio, float(artifact["alpha"]))
    fused = apply_temperature(
        np.log(np.maximum(uncalibrated, 1e-12)), float(artifact["temperature"])
    )

    order = np.sort(fused, axis=1)
    uncertain = (order[:, -1] < 0.55) | ((order[:, -1] - order[:, -2]) < 0.15)
    correct = fused.argmax(1) == labels
    rng = np.random.default_rng(42)
    shuffled_audio = head_probabilities(
        artifact["audio_head"], "audio", features["audio"][rng.permutation(len(labels))]
    )
    shuffled_fused = apply_temperature(
        np.log(
            np.maximum(fuse_probabilities(text, shuffled_audio, float(artifact["alpha"])), 1e-12)
        ),
        float(artifact["temperature"]),
    )

    seed_fused, seed_text = [], []
    for seed in SEEDS:
        text_artifact = torch.load(
            Path("checkpoints/day2") / f"text_inverse_sqrt_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        audio_artifact = torch.load(
            Path("checkpoints/day2") / f"audio_inverse_sqrt_{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        seed_text_probabilities = head_probabilities(text_artifact, "text", features["text"])
        seed_audio_probabilities = head_probabilities(audio_artifact, "audio", features["audio"])
        seed_text.append(macro(labels, seed_text_probabilities))
        seed_fused.append(
            macro(
                labels,
                fuse_probabilities(
                    seed_text_probabilities, seed_audio_probabilities, float(artifact["alpha"])
                ),
            )
        )

    majority = int(np.bincount(labels, minlength=len(EMOTIONS)).argmax())
    majority_probabilities = np.eye(len(EMOTIONS), dtype=np.float32)[np.full(len(labels), majority)]
    deployed_metrics = metric_report(labels, fused)
    report = {
        "evaluation_status": "final_test_opened",
        "checkpoint": str(checkpoint_path),
        "test_examples": len(labels),
        "alpha": float(artifact["alpha"]),
        "temperature": float(artifact["temperature"]),
        "deployed": deployed_metrics,
        "text_only_seed42": metric_report(labels, text),
        "audio_only_seed42": metric_report(labels, audio),
        "majority_baseline": {
            "emotion": EMOTIONS[majority],
            **metric_report(labels, majority_probabilities),
        },
        "three_seed_test_macro_f1": {
            "text": {
                "per_seed": seed_text,
                "mean": float(np.mean(seed_text)),
                "std": float(np.std(seed_text, ddof=1)),
            },
            "fused": {
                "per_seed": seed_fused,
                "mean": float(np.mean(seed_fused)),
                "std": float(np.std(seed_fused, ddof=1)),
            },
        },
        "fused_minus_text_bootstrap": dialogue_bootstrap(labels, fused, text, features["clip_ids"]),
        "audio_shuffle": {
            "macro_f1": macro(labels, shuffled_fused),
            "change_from_deployed": macro(labels, shuffled_fused) - deployed_metrics["macro_f1"],
        },
        "uncertainty": {
            "uncertain_fraction": float(uncertain.mean()),
            "certain_coverage": float((~uncertain).mean()),
            "certain_accuracy": float(correct[~uncertain].mean()) if (~uncertain).any() else None,
            "uncertain_accuracy": float(correct[uncertain].mean()) if uncertain.any() else None,
        },
        "exclusions": [
            {"split": "train", "clip": "dia125_utt3", "reason": "PyAV InvalidDataError"},
            {"split": "dev", "clip": "dia110_utt7", "reason": "Missing source media"},
            {"split": "test", "count": 0},
        ],
    }
    output = Path("reports/day3_final_evaluation.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    Path("reports/day3_final_summary.md").write_text(
        "# Day 3 final evaluation\n\n"
        f"Frozen checkpoint: `{checkpoint_path}`. Official MELD test examples: {len(labels)}.\n\n"
        "| Model | Macro-F1 | Weighted F1 | Accuracy |\n"
        "|---|---:|---:|---:|\n"
        f"| Deployed fusion | {deployed_metrics['macro_f1']:.4f} | "
        f"{deployed_metrics['weighted_f1']:.4f} | {deployed_metrics['accuracy']:.4f} |\n"
        f"| Text only | {report['text_only_seed42']['macro_f1']:.4f} | "
        f"{report['text_only_seed42']['weighted_f1']:.4f} | "
        f"{report['text_only_seed42']['accuracy']:.4f} |\n"
        f"| Audio only | {report['audio_only_seed42']['macro_f1']:.4f} | "
        f"{report['audio_only_seed42']['weighted_f1']:.4f} | "
        f"{report['audio_only_seed42']['accuracy']:.4f} |\n"
        f"| Majority ({EMOTIONS[majority]}) | {report['majority_baseline']['macro_f1']:.4f} | "
        f"{report['majority_baseline']['weighted_f1']:.4f} | "
        f"{report['majority_baseline']['accuracy']:.4f} |\n\n"
        f"Audio shuffle macro-F1: {report['audio_shuffle']['macro_f1']:.4f}. "
        f"Dialogue-bootstrap fused-minus-text 95% CI: "
        f"[{report['fused_minus_text_bootstrap']['ci95_low']:.4f}, "
        f"{report['fused_minus_text_bootstrap']['ci95_high']:.4f}].\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
