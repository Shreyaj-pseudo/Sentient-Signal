import pytest
from pydantic import ValidationError

from sentient_signals.constants import EMOTIONS
from sentient_signals.policy import choose_response
from sentient_signals.schema import ClientEvent, PredictionResult, Timings


def test_uncertainty_keeps_reply_and_gesture_consistent() -> None:
    decision = choose_response("joy", uncertain=True)
    assert decision.policy_id == "clarify_uncertain_v1"
    assert decision.gesture_id == "clarification"
    assert "not fully sure" in decision.response_text


def test_response_cue_is_grounded_and_negation_is_not() -> None:
    assert '"frustrated."' in choose_response("anger", False, "I'm frustrated.").response_text
    assert '"happy."' not in choose_response("sadness", False, "I'm not happy.").response_text


def test_prediction_requires_normalized_seven_class_probabilities() -> None:
    probabilities = dict.fromkeys(EMOTIONS, 1 / 7)
    result = PredictionResult(
        turn_id="turn-1",
        transcript="hello",
        transcript_source="supplied",
        emotion="neutral",
        probabilities=probabilities,
        uncertain=True,
        response_text="I hear you.",
        policy_id="test",
        gesture_id="acknowledgment",
        timings=Timings(audio_ms=1, text_ms=1, classifier_ms=1, total_ms=3),
    )
    assert sum(result.probabilities.values()) == pytest.approx(1)

    probabilities["neutral"] = 0.9
    with pytest.raises(ValidationError):
        PredictionResult(
            turn_id="turn-2",
            transcript="hello",
            transcript_source="supplied",
            emotion="neutral",
            probabilities=probabilities,
            uncertain=False,
            response_text="I hear you.",
            policy_id="test",
            gesture_id="acknowledgment",
            timings=Timings(audio_ms=1, text_ms=1, classifier_ms=1, total_ms=3),
        )


def test_client_event_rejects_unknown_schema_version() -> None:
    with pytest.raises(ValidationError):
        ClientEvent(schema_version="2.0", event="start_turn", turn_id="turn-1")
