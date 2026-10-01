"""Compare current-turn text with three-turn causal dialogue context."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from sentient_signals.data import ManifestItem, load_manifest
from sentient_signals.features import TEXT_MODEL, TEXT_REVISION
from sentient_signals.model import fit_head

SEEDS = (13, 42, 2026)
WINDOW = 3


def contextualize(items: list[ManifestItem], window: int = WINDOW) -> list[str]:
    if window < 1:
        raise ValueError("window must be positive")
    contexts = []
    for index, current in enumerate(items):
        history = []
        for previous in items[max(0, index - window) : index]:
            if previous.dialogue_id == current.dialogue_id:
                role = "SELF" if previous.speaker == current.speaker else "OTHER"
                history.append(f"{role}: {previous.text}")
        contexts.append("\n".join([*history, f"CURRENT: {current.text}"]))
    return contexts


def cache_split(split: str) -> Path:
    from sentence_transformers import SentenceTransformer

    items = load_manifest(Path("data/day2/processed") / f"{split}.jsonl")
    baseline_path = Path("data/day2/features") / f"{split}.npz"
    with np.load(baseline_path, allow_pickle=False) as baseline:
        audio = baseline["audio"].copy()
        labels = baseline["labels"].copy()
        clip_ids = baseline["clip_ids"].astype(str)
    if clip_ids.tolist() != [item.clip_id for item in items]:
        raise ValueError(f"{split} manifest does not match the Day 2 feature cache")

    model = SentenceTransformer(TEXT_MODEL, revision=TEXT_REVISION, local_files_only=True)
    text = model.encode(
        contextualize(items),
        batch_size=128,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    output = Path("data/context_text") / f"{split}.npz"
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        text=text,
        audio=audio,
        labels=labels,
        clip_ids=clip_ids,
        feature_mode=np.asarray("encoders"),
        feature_config=np.asarray(
            json.dumps(
                {
                    "text_model": TEXT_MODEL,
                    "text_revision": TEXT_REVISION,
                    "context": "previous_3_turns_self_other_current",
                },
                sort_keys=True,
            )
        ),
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    import torch

    torch.set_num_threads(args.threads)
    train = Path("data/context_text/train.npz")
    dev = Path("data/context_text/dev.npz")
    if not args.compare_only:
        train, dev = cache_split("train"), cache_split("dev")
    if not train.exists() or not dev.exists():
        raise FileNotFoundError("Run without --compare-only to create contextual features")

    scores = []
    for seed in SEEDS:
        result = fit_head(
            train,
            dev,
            Path("checkpoints/context_text") / f"text_inverse_sqrt_{seed}.pt",
            seed=seed,
            modality="text",
            class_weighting="inverse_sqrt",
        )
        scores.append(float(result["dev_macro_f1"]))
        print(f"seed={seed} dev_macro_f1={scores[-1]:.4f}", flush=True)

    baseline = json.loads(Path("reports/day2_experiment.json").read_text(encoding="utf-8"))[
        "comparisons"
    ]["text_inverse_sqrt"]
    report = {
        "context": "previous_3_turns_self_other_current",
        "seeds": list(SEEDS),
        "context_macro_f1": {
            "per_seed": scores,
            "mean": float(np.mean(scores)),
            "std": float(np.std(scores, ddof=1)),
        },
        "current_turn_baseline_macro_f1": baseline,
        "mean_improvement": float(np.mean(scores) - baseline["mean"]),
        "test_metrics": "Not evaluated",
    }
    output = Path("reports/context_text_experiment.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
