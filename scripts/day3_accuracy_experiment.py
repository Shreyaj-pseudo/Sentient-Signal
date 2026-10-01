"""Tune cheap Day 3 fusion variants without touching the test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score

from sentient_signals.calibration import (
    apply_temperature,
    dialogue_calibration_mask,
    fit_temperature,
    fuse_probabilities,
)
from sentient_signals.model import FusionHead, _feature_matrix

SEEDS = (13, 42, 2026)
WEIGHTS = tuple(round(float(weight), 3) for weight in np.arange(0.5, 0.951, 0.025))


def logits(artifact: dict, features: dict[str, np.ndarray]) -> np.ndarray:
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


def choose_weight(scores: dict[float, list[float]]) -> float:
    """Choose by mean across seeds; prefer more text on an exact tie."""
    return max(scores, key=lambda weight: (np.mean(scores[weight]), weight))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--apply",
        action="store_true",
        help="replace the deployed artifact after selection and calibration",
    )
    args = parser.parse_args()

    dev_path = Path("data/day2/features/dev.npz")
    if not dev_path.exists():
        raise FileNotFoundError("Run the Day 2 experiment first")
    with np.load(dev_path, allow_pickle=False) as data:
        features = {key: data[key].copy() for key in data.files}
    labels = features["labels"]
    calibration = dialogue_calibration_mask(features["clip_ids"])
    selection = ~calibration

    probabilities: dict[str, list[np.ndarray]] = {"text": [], "audio": []}
    artifacts: dict[str, list[dict]] = {"text": [], "audio": []}
    for modality in probabilities:
        for seed in SEEDS:
            path = Path("checkpoints/day2") / f"{modality}_inverse_sqrt_{seed}.pt"
            artifact = torch.load(path, map_location="cpu", weights_only=True)
            artifacts[modality].append(artifact)
            probabilities[modality].append(apply_temperature(logits(artifact, features), 1))

    scores = {
        float(weight): [
            macro(
                labels[selection],
                fuse_probabilities(
                    probabilities["text"][index],
                    probabilities["audio"][index],
                    float(weight),
                )[selection],
            )
            for index in range(len(SEEDS))
        ]
        for weight in WEIGHTS
    }
    selected_weight = choose_weight(scores)
    text_ensemble = np.mean(probabilities["text"], axis=0)
    audio_ensemble = np.mean(probabilities["audio"], axis=0)
    ensemble_scores = {
        float(weight): macro(
            labels[selection],
            fuse_probabilities(text_ensemble, audio_ensemble, float(weight))[selection],
        )
        for weight in WEIGHTS
    }

    baseline = float(np.mean(scores[0.75]))
    selected_mean = float(np.mean(scores[selected_weight]))
    report = {
        "selection_only": True,
        "seeds": list(SEEDS),
        "baseline_weight": 0.75,
        "baseline_macro_f1": baseline,
        "weights": {
            str(weight): {"per_seed": values, "mean": float(np.mean(values))}
            for weight, values in scores.items()
        },
        "selected_weight": selected_weight,
        "selected_macro_f1": selected_mean,
        "mean_improvement": selected_mean - baseline,
        "three_seed_ensemble": {
            "best_weight": max(ensemble_scores, key=ensemble_scores.get),
            "best_macro_f1": max(ensemble_scores.values()),
            "deployment_cost": "three text heads and three audio heads",
        },
        "recommendation": "fine_weight" if selected_mean > baseline else "keep_day2",
        "test_metrics": "Not evaluated; configuration is not frozen until --apply",
    }
    output = Path("reports/day3_accuracy_experiment.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if args.apply:
        index = SEEDS.index(42)
        fused = fuse_probabilities(
            probabilities["text"][index], probabilities["audio"][index], selected_weight
        )
        temperature = fit_temperature(
            np.log(np.maximum(fused[calibration], 1e-12)), labels[calibration]
        )
        deployed = {
            "artifact_kind": "day3_probability_fusion",
            "modality": "probability",
            "feature_mode": "encoders",
            "text_dimension": 384,
            "audio_dimension": 1536,
            "alpha": selected_weight,
            "text_head": artifacts["text"][index],
            "audio_head": artifacts["audio"][index],
            "temperature": temperature,
            "labels": artifacts["text"][index]["labels"],
            "feature_config": artifacts["text"][index]["feature_config"],
            "seed": 42,
            "calibration_clip_ids": features["clip_ids"][calibration].tolist(),
        }
        torch.save(deployed, "checkpoints/day3_deployed.pt")
        report["applied_checkpoint"] = "checkpoints/day3_deployed.pt"
        report["temperature"] = temperature
        report["test_metrics"] = "Not evaluated"
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
