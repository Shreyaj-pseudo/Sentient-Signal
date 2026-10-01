"""Bounded, cancellation-aware WebSocket service for live and replay turns."""

from __future__ import annotations

import asyncio
import os
import tempfile
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from sentient_signals.audio import pcm16_bytes_to_float32, write_pcm16_wav
from sentient_signals.constants import MAX_LIVE_SECONDS, SAMPLE_RATE
from sentient_signals.schema import ClientEvent, ErrorEvent, PredictionResult, StatusEvent

MAX_AUDIO_BYTES = int(MAX_LIVE_SECONDS * SAMPLE_RATE * 2)


class TurnSession:
    """Own one connection's audio buffer and suppress stale inference results."""

    def __init__(self, predictor: Callable[[Path, str | None], PredictionResult]) -> None:
        self.predictor = predictor
        self.turn_id: str | None = None
        self.transcript: str | None = None
        self.audio = bytearray()
        self.generation = 0
        self.worker_running = False

    def start(self, event: ClientEvent) -> StatusEvent:
        if self.worker_running:
            raise RuntimeError("The previous inference is still finishing")
        self.generation += 1
        self.turn_id = event.turn_id
        self.transcript = event.transcript
        self.audio.clear()
        return StatusEvent(turn_id=event.turn_id, status="listening")

    def append(self, chunk: bytes) -> None:
        if self.turn_id is None:
            raise RuntimeError("Send start_turn before audio")
        if len(self.audio) + len(chunk) > MAX_AUDIO_BYTES:
            raise ValueError("Audio exceeds the ten-second limit")
        if len(chunk) % 2:
            raise ValueError("PCM16 chunks must contain an even number of bytes")
        self.audio.extend(chunk)

    def cancel(self, turn_id: str) -> StatusEvent:
        if turn_id != self.turn_id:
            raise RuntimeError("Cannot cancel a stale turn")
        self.generation += 1
        self.turn_id = None
        self.audio.clear()
        return StatusEvent(turn_id=turn_id, status="cancelled")

    async def finish(self, turn_id: str) -> PredictionResult | None:
        if turn_id != self.turn_id:
            raise RuntimeError("Cannot finish a stale turn")
        self.worker_running = True
        try:
            samples = pcm16_bytes_to_float32(bytes(self.audio))
            generation = self.generation
            transcript = self.transcript
            self.audio.clear()
            runtime_directory = Path(".runtime")
            runtime_directory.mkdir(exist_ok=True)
            with tempfile.TemporaryDirectory(
                prefix="sentient-signals-", dir=runtime_directory
            ) as directory:
                audio_path = Path(directory) / "turn.wav"
                write_pcm16_wav(audio_path, samples)
                result = await asyncio.to_thread(self.predictor, audio_path, transcript)
        finally:
            self.worker_running = False
        if generation != self.generation or turn_id != self.turn_id:
            return None
        self.turn_id = None
        return result.for_turn(turn_id)


def create_app(pipeline=None):
    """Create the FastAPI app; injecting a pipeline keeps service tests lightweight."""
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.staticfiles import StaticFiles

    app = FastAPI(title="SentientSignals", version="0.2.0")
    static_dir = Path(__file__).resolve().parents[2] / "app" / "static"

    def get_pipeline():
        if not hasattr(app.state, "pipeline"):
            if pipeline is not None:
                app.state.pipeline = pipeline
            else:
                from sentient_signals.model import InferencePipeline

                checkpoint = Path(
                    os.environ.get("SENTIENT_CHECKPOINT", "checkpoints/day1_encoder_subset.pt")
                )
                app.state.pipeline = InferencePipeline(checkpoint, local_files_only=True)
        return app.state.pipeline

    async def websocket_endpoint(websocket) -> None:
        await websocket.accept()
        session = TurnSession(lambda path, text: get_pipeline().predict(path, text=text))
        inference_tasks: set[asyncio.Task] = set()

        async def finish_and_send(turn_id: str) -> None:
            try:
                result = await session.finish(turn_id)
                if result is not None:
                    await websocket.send_json(result.model_dump())
            except (ValueError, RuntimeError) as error:
                await websocket.send_json(
                    ErrorEvent(
                        turn_id=turn_id, code="invalid_audio", message=str(error)
                    ).model_dump()
                )
            except Exception:
                await websocket.send_json(
                    ErrorEvent(
                        turn_id=turn_id,
                        code="inference_failed",
                        message="Local inference failed. Check the server log and retry.",
                    ).model_dump()
                )

        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    raise WebSocketDisconnect(code=message.get("code", 1000))
                if message.get("bytes") is not None:
                    try:
                        session.append(message["bytes"])
                    except (RuntimeError, ValueError) as error:
                        turn_id = session.turn_id or "unknown"
                        await websocket.send_json(
                            ErrorEvent(
                                turn_id=turn_id, code="invalid_audio", message=str(error)
                            ).model_dump()
                        )
                    continue
                try:
                    event = ClientEvent.model_validate_json(message["text"])
                    if event.event == "start_turn":
                        await websocket.send_json(session.start(event).model_dump())
                    elif event.event == "cancel_turn":
                        await websocket.send_json(session.cancel(event.turn_id).model_dump())
                    elif event.event == "end_turn":
                        if session.worker_running:
                            raise RuntimeError("The previous inference is still finishing")
                        session.worker_running = True
                        await websocket.send_json(
                            StatusEvent(turn_id=event.turn_id, status="processing").model_dump()
                        )
                        task = asyncio.create_task(finish_and_send(event.turn_id))
                        inference_tasks.add(task)
                        task.add_done_callback(inference_tasks.discard)
                    else:
                        raise ValueError(f"Unsupported event: {event.event}")
                except (ValidationError, ValueError, RuntimeError) as error:
                    await websocket.send_json(
                        ErrorEvent(
                            turn_id=session.turn_id or "unknown",
                            code="invalid_event",
                            message=str(error),
                        ).model_dump()
                    )
        except WebSocketDisconnect:
            if session.turn_id is not None:
                session.cancel(session.turn_id)
            if inference_tasks:
                await asyncio.gather(*inference_tasks, return_exceptions=True)

    websocket_endpoint.__annotations__["websocket"] = WebSocket
    app.add_api_websocket_route("/ws", websocket_endpoint)
    app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    return app
