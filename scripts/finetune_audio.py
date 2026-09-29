"""Partially fine-tune the MELD audio encoder without changing deployment.

Run from the repository root:
    python scripts/finetune_audio.py
"""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

from sentient_signals.audio import read_pcm16_wav
from sentient_signals.calibration import dialogue_calibration_mask
from sentient_signals.constants import EMOTIONS, SAMPLE_RATE
from sentient_signals.data import ManifestItem, load_manifest
from sentient_signals.features import EMOTION_AUDIO_MODEL, EMOTION_AUDIO_REVISION


class AudioDataset(Dataset):
    def __init__(self, items: list[ManifestItem], root: Path = Path("data/day2/processed")) -> None:
        self.items = items
        self.root = root

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
        item = self.items[index]
        path = self.root / item.split / f"{item.clip_id}.wav"
        return read_pcm16_wav(path)[: SAMPLE_RATE * 10], item.label


def unfreeze_last_layers(model: nn.Module, count: int) -> list[str]:
    if count < 1:
        raise ValueError("count must be positive")
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.wav2vec2.encoder.layers
    if count > len(layers):
        raise ValueError(f"Cannot unfreeze {count} of {len(layers)} encoder layers")
    for module in [*layers[-count:], model.projector, model.classifier]:
        for parameter in module.parameters():
            parameter.requires_grad = True
    return [name for name, parameter in model.named_parameters() if parameter.requires_grad]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--projector-learning-rate", type=float, default=1e-4)
    parser.add_argument("--classifier-learning-rate", type=float, default=1e-3)
    parser.add_argument("--unfreeze-layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/audio_finetuned.pt"))
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size) < 1:
        parser.error("epochs, patience, and batch-size must be positive")

    torch.manual_seed(args.seed)
    train_items = load_manifest(Path("data/day2/processed/train.jsonl"))
    dev_items = load_manifest(Path("data/day2/processed/dev.jsonl"))
    calibration = dialogue_calibration_mask(np.asarray([item.clip_id for item in dev_items]))
    dev_items = [
        item for item, reserved in zip(dev_items, calibration, strict=True) if not reserved
    ]

    processor = AutoFeatureExtractor.from_pretrained(
        EMOTION_AUDIO_MODEL, revision=EMOTION_AUDIO_REVISION
    )

    def collate(
        batch: list[tuple[np.ndarray, int]],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        audio, labels = zip(*batch, strict=True)
        inputs = processor(
            list(audio),
            sampling_rate=SAMPLE_RATE,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return dict(inputs), torch.tensor(labels, dtype=torch.long)

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        AudioDataset(train_items),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate,
        generator=generator,
    )
    dev_loader = DataLoader(AudioDataset(dev_items), batch_size=args.batch_size, collate_fn=collate)

    model = AutoModelForAudioClassification.from_pretrained(
        EMOTION_AUDIO_MODEL, revision=EMOTION_AUDIO_REVISION
    )
    # LayerDrop can return fewer hidden states than this checkpoint's 13 learned weights.
    # Fine-tuning the tail uses the final hidden state directly and avoids that mismatch.
    model.config.use_weighted_layer_sum = False
    model.classifier = nn.Linear(model.classifier.in_features, len(EMOTIONS))
    trainable = unfreeze_last_layers(model, args.unfreeze_layers)
    model.to(args.device)
    trainable_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    print(
        f"device={args.device} train={len(train_items)} dev={len(dev_items)} "
        f"trainable_parameters={trainable_count:,}",
        flush=True,
    )

    counts = np.bincount([item.label for item in train_items], minlength=len(EMOTIONS))
    weights = 1 / np.sqrt(counts)
    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(weights / weights.mean(), dtype=torch.float32, device=args.device)
    )
    optimizer = torch.optim.AdamW(
        [
            {
                "params": model.wav2vec2.encoder.layers[-args.unfreeze_layers :].parameters(),
                "lr": args.encoder_learning_rate,
            },
            {"params": model.projector.parameters(), "lr": args.projector_learning_rate},
            {"params": model.classifier.parameters(), "lr": args.classifier_learning_rate},
        ],
        weight_decay=1e-4,
    )

    best_score, best_state, stale = -1.0, None, 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        for batch_number, (inputs, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            logits = model(**{key: value.to(args.device) for key, value in inputs.items()}).logits
            loss = criterion(logits, labels.to(args.device))
            loss.backward()
            optimizer.step()
            if batch_number == 1 or batch_number % 25 == 0:
                print(
                    f"epoch={epoch} batch={batch_number}/{len(train_loader)} "
                    f"loss={loss.item():.4f}",
                    flush=True,
                )

        model.eval()
        predicted, expected = [], []
        with torch.inference_mode():
            for inputs, labels in dev_loader:
                logits = model(
                    **{key: value.to(args.device) for key, value in inputs.items()}
                ).logits
                predicted.extend(logits.argmax(1).cpu().tolist())
                expected.extend(labels.tolist())
        score = f1_score(
            expected, predicted, labels=range(len(EMOTIONS)), average="macro", zero_division=0
        )
        counts = np.bincount(predicted, minlength=len(EMOTIONS))
        distribution = ", ".join(
            f"{emotion}={count}" for emotion, count in zip(EMOTIONS, counts, strict=True)
        )
        print(
            f"epoch={epoch} dev_macro_f1={score:.4f} predictions=[{distribution}]",
            flush=True,
        )
        if score > best_score:
            best_score = float(score)
            best_state = copy.deepcopy(model.state_dict())
            torch.save(
                {
                    "artifact_kind": "meld_audio_partial_finetune",
                    "model": EMOTION_AUDIO_MODEL,
                    "revision": EMOTION_AUDIO_REVISION,
                    "labels": list(EMOTIONS),
                    "unfrozen_encoder_layers": args.unfreeze_layers,
                    "trainable_parameters": trainable,
                    "model_state": best_state,
                    "dev_macro_f1": best_score,
                    "epoch": epoch,
                    "seed": args.seed,
                },
                args.output,
            )
            print(f"saved={args.output}", flush=True)
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    print(f"saved={args.output} best_dev_macro_f1={best_score:.4f}")


if __name__ == "__main__":
    main()
