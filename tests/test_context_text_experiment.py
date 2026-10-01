from scripts.context_text_experiment import contextualize
from sentient_signals.data import ManifestItem


def item(dialogue: int, utterance: int, speaker: str, text: str) -> ManifestItem:
    return ManifestItem("train", dialogue, utterance, speaker, text, "neutral", 4, "", "")


def test_context_is_causal_relative_and_dialogue_scoped() -> None:
    contexts = contextualize(
        [
            item(0, 0, "A", "first"),
            item(0, 1, "B", "second"),
            item(0, 2, "A", "third"),
            item(1, 0, "B", "new dialogue"),
        ]
    )

    assert contexts == [
        "CURRENT: first",
        "OTHER: first\nCURRENT: second",
        "SELF: first\nOTHER: second\nCURRENT: third",
        "CURRENT: new dialogue",
    ]
