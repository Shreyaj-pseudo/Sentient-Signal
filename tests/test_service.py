import asyncio
import time
from pathlib import Path

import numpy as np
import pytest

from sentient_signals.audio import read_pcm16_wav
from sentient_signals.constants import EMOTIONS
from sentient_signals.schema import ClientEvent, PredictionResult, Timings
from sentient_signals.service import MAX_AUDIO_BYTES, TurnSession


def result(turn_id: str = "generated") -> PredictionResult:
    probabilities = {emotion: 0.0 for emotion in EMOTIONS}
    probabilities["neutral"] = 1.0
    return PredictionResult(
        turn_id=turn_id,
        transcript="hello",
        transcript_source="supplied",
        emotion="neutral",
        probabilities=probabilities,
        uncertain=False,
        response_text="I hear you.",
        policy_id="acknowledge_neutral_v1",
        gesture_id="acknowledgment",
        timings=Timings(audio_ms=1, text_ms=1, classifier_ms=1, total_ms=3),
    )


def test_turn_session_preserves_pcm_and_client_turn_id() -> None:
    expected = np.asarray([0, 1000, -1000], dtype="<i2")

    def predict(path: Path, text: str | None) -> PredictionResult:
        assert text == "hello"
        assert np.array_equal((read_pcm16_wav(path) * 32768).astype("<i2"), expected)
        return result()

    session = TurnSession(predict)
    session.start(ClientEvent(event="start_turn", turn_id="turn-1", transcript="hello"))
    session.append(expected.tobytes())
    actual = asyncio.run(session.finish("turn-1"))
    assert actual is not None
    assert actual.turn_id == "turn-1"


def test_cancel_suppresses_stale_worker_result() -> None:
    def slow_predict(_path: Path, _text: str | None) -> PredictionResult:
        time.sleep(0.05)
        return result()

    async def scenario() -> PredictionResult | None:
        session = TurnSession(slow_predict)
        session.start(ClientEvent(event="start_turn", turn_id="turn-1"))
        session.append(np.zeros(100, dtype="<i2").tobytes())
        task = asyncio.create_task(session.finish("turn-1"))
        await asyncio.sleep(0.01)
        session.cancel("turn-1")
        return await task

    assert asyncio.run(scenario()) is None


def test_audio_buffer_is_bounded_and_rejects_malformed_pcm() -> None:
    session = TurnSession(lambda _path, _text: result())
    session.start(ClientEvent(event="start_turn", turn_id="turn-1"))
    with pytest.raises(ValueError, match="even"):
        session.append(b"x")
    with pytest.raises(ValueError, match="ten-second"):
        session.append(bytes(MAX_AUDIO_BYTES + 2))


def test_app_serves_the_browser_interface() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sentient_signals.service import create_app

    response = TestClient(create_app(pipeline=object())).get("/")
    assert response.status_code == 200
    assert "SentientSignals" in response.text


def test_websocket_turn_reaches_result() -> None:
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from sentient_signals.service import create_app

    class FakePipeline:
        def predict(self, _path: Path, text: str | None = None) -> PredictionResult:
            return result().model_copy(update={"transcript": text or "asr"})

    with TestClient(create_app(pipeline=FakePipeline())).websocket_connect("/ws") as websocket:
        websocket.send_json(
            {
                "schema_version": "1.0",
                "event": "start_turn",
                "turn_id": "turn-1",
                "transcript": "hello",
            }
        )
        assert websocket.receive_json()["status"] == "listening"
        websocket.send_bytes(np.zeros(100, dtype="<i2").tobytes())
        websocket.send_json({"schema_version": "1.0", "event": "end_turn", "turn_id": "turn-1"})
        assert websocket.receive_json()["status"] == "processing"
        response = websocket.receive_json()
        assert response["event"] == "result"
        assert response["turn_id"] == "turn-1"
        assert response["transcript"] == "hello"
