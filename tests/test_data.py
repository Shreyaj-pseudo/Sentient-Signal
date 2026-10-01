import csv
from pathlib import Path

from sentient_signals.data import normalize_meld_text, read_split, representative_subset


def test_split_matching_and_missing_clip(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    media = raw / "dev" / "dev_splits_complete"
    media.mkdir(parents=True)
    (media / "dia1_utt2.mp4").touch()
    rows = [
        {
            "Sr No.": "1",
            "Utterance": "hello",
            "Speaker": "A",
            "Emotion": "joy",
            "Sentiment": "positive",
            "Dialogue_ID": "1",
            "Utterance_ID": "2",
            "Season": "1",
            "Episode": "1",
            "StartTime": "0",
            "EndTime": "1",
        },
        {
            "Sr No.": "2",
            "Utterance": "missing",
            "Speaker": "B",
            "Emotion": "fear",
            "Sentiment": "negative",
            "Dialogue_ID": "1",
            "Utterance_ID": "3",
            "Season": "1",
            "Episode": "1",
            "StartTime": "1",
            "EndTime": "2",
        },
    ]
    with (raw / "dev_sent_emo.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)

    items, missing = read_split(raw, "dev", tmp_path / "processed")

    assert [item.clip_id for item in items] == ["dia1_utt2"]
    assert items[0].label == 3
    assert missing == ["dev/dev_splits_complete/dia1_utt3.mp4"]


def test_representative_subset_is_deterministic_and_label_aware(tmp_path: Path) -> None:
    raw = Path("MELD.Raw")
    items, _ = read_split(raw, "train", tmp_path)
    first = representative_subset(items, 14, seed=42)
    second = representative_subset(items, 14, seed=42)

    assert [item.clip_id for item in first] == [item.clip_id for item in second]
    assert set(item.emotion for item in first) == {
        "anger",
        "disgust",
        "fear",
        "joy",
        "neutral",
        "sadness",
        "surprise",
    }


def test_legacy_meld_punctuation_is_normalized() -> None:
    assert normalize_meld_text("I\u0092m here\u0097really") == "I'm here-really"
