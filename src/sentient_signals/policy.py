"""Deterministic, grounded reply and lamp-gesture policy."""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PolicyDecision:
    response_text: str
    policy_id: str
    gesture_id: str


_CUES = (
    "thank you",
    "thanks",
    "sorry",
    "help",
    "frustrated",
    "angry",
    "upset",
    "worried",
    "afraid",
    "scared",
    "sad",
    "happy",
    "excited",
)


def _cue(transcript: str) -> str | None:
    """Return a short, explicit, non-negated transcript cue."""
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", transcript.lower())
    for cue in _CUES:
        cue_words = cue.split()
        for index in range(len(words) - len(cue_words) + 1):
            if words[index : index + len(cue_words)] != cue_words:
                continue
            if index and words[index - 1] in {"no", "not", "never", "don't", "isn't", "wasn't"}:
                continue
            return cue
    return None


def choose_response(emotion: str, uncertain: bool, transcript: str = "") -> PolicyDecision:
    if uncertain:
        return PolicyDecision(
            "I'm not fully sure how that felt. Would you tell me a little more?",
            "clarify_uncertain_v1",
            "clarification",
        )
    mapping = {
        "anger": ("That sounds frustrating. I'm listening.", "acknowledge_anger_v1", "reassurance"),
        "disgust": ("That sounds deeply unpleasant.", "acknowledge_disgust_v1", "acknowledgment"),
        "fear": (
            "That sounds unsettling. We can take it slowly.",
            "reassure_fear_v1",
            "reassurance",
        ),
        "joy": ("That sounds wonderful!", "celebrate_joy_v1", "celebration"),
        "neutral": ("I hear you.", "acknowledge_neutral_v1", "acknowledgment"),
        "sadness": ("That sounds hard. I'm here with you.", "support_sadness_v1", "reassurance"),
        "surprise": ("That sounds unexpected!", "acknowledge_surprise_v1", "acknowledgment"),
    }
    text, policy_id, gesture_id = mapping[emotion]
    if cue := _cue(transcript):
        text = f'I heard "{cue}." {text}'
        policy_id = f"{policy_id}_cue_v1"
    return PolicyDecision(text, policy_id, gesture_id)
