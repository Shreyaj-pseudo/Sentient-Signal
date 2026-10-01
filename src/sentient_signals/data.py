"""MELD split discovery and manifest handling."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from sentient_signals.constants import LABEL_TO_INDEX


@dataclass(frozen=True)
class ManifestItem:
    split: str
    dialogue_id: int
    utterance_id: int
    speaker: str
    text: str
    emotion: str
    label: int
    source_path: str
    audio_path: str
    duration_seconds: float | None = None

    @property
    def clip_id(self) -> str:
        return f"dia{self.dialogue_id}_utt{self.utterance_id}"


SPLIT_LAYOUT = {
    "train": ("train/train_sent_emo.csv", "train/train_splits"),
    "dev": ("dev_sent_emo.csv", "dev/dev_splits_complete"),
    "test": ("test_sent_emo.csv", "test/output_repeated_splits_test"),
}

MELD_PUNCTUATION = str.maketrans(
    {
        "\u0085": "...",
        "\u0091": "'",
        "\u0092": "'",
        "\u0093": '"',
        "\u0094": '"',
        "\u0096": "-",
        "\u0097": "-",
    }
)


def normalize_meld_text(value: str) -> str:
    """Repair legacy Windows punctuation found in the supplied MELD CSV files."""
    return value.translate(MELD_PUNCTUATION).replace("Â", "").strip()


def read_split(
    raw_root: Path, split: str, output_root: Path
) -> tuple[list[ManifestItem], list[str]]:
    """Match CSV rows to media by split and report all missing inputs."""
    if split not in SPLIT_LAYOUT:
        raise ValueError(f"Unknown split: {split}")
    csv_rel, media_rel = SPLIT_LAYOUT[split]
    csv_path = raw_root / csv_rel
    media_dir = raw_root / media_rel
    items: list[ManifestItem] = []
    missing: list[str] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            dialogue_id = int(row["Dialogue_ID"])
            utterance_id = int(row["Utterance_ID"])
            clip_id = f"dia{dialogue_id}_utt{utterance_id}"
            source = media_dir / f"{clip_id}.mp4"
            if not source.is_file() or source.name.startswith("._"):
                missing.append(source.relative_to(raw_root).as_posix())
                continue
            emotion = row["Emotion"].strip().lower()
            if emotion not in LABEL_TO_INDEX:
                raise ValueError(f"Unsupported label {emotion!r} in {csv_path}")
            audio = output_root / split / f"{clip_id}.wav"
            items.append(
                ManifestItem(
                    split=split,
                    dialogue_id=dialogue_id,
                    utterance_id=utterance_id,
                    speaker=normalize_meld_text(row["Speaker"]),
                    text=normalize_meld_text(row["Utterance"]),
                    emotion=emotion,
                    label=LABEL_TO_INDEX[emotion],
                    source_path=source.resolve().as_posix(),
                    audio_path=audio.resolve().as_posix(),
                )
            )
    return items, missing


def write_manifest(path: Path, items: Iterable[ManifestItem]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")


def load_manifest(path: Path) -> list[ManifestItem]:
    with path.open("r", encoding="utf-8") as handle:
        items = []
        for line in handle:
            if line.strip():
                value = json.loads(line)
                value["text"] = normalize_meld_text(value["text"])
                value["speaker"] = normalize_meld_text(value["speaker"])
                items.append(ManifestItem(**value))
        return items


def representative_subset(
    items: list[ManifestItem], limit: int, seed: int = 42
) -> list[ManifestItem]:
    """Choose a deterministic, label-stratified subset without dialogue leakage assumptions."""
    if limit >= len(items):
        return list(items)
    import random

    groups: dict[str, list[ManifestItem]] = {}
    for item in items:
        groups.setdefault(item.emotion, []).append(item)
    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)
    chosen: list[ManifestItem] = []
    labels = sorted(groups)
    while len(chosen) < limit:
        progressed = False
        for label in labels:
            if groups[label] and len(chosen) < limit:
                chosen.append(groups[label].pop())
                progressed = True
        if not progressed:
            break
    return chosen
