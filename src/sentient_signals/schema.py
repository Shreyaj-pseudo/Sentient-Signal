"""Validated boundary objects for prediction output."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from sentient_signals.constants import EMOTIONS, SCHEMA_VERSION


class ClientEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event: str
    turn_id: str = Field(min_length=1, max_length=128)
    transcript: str | None = Field(default=None, max_length=10_000)

    @model_validator(mode="after")
    def validate_schema_version(self) -> ClientEvent:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"Unsupported schema version: {self.schema_version}")
        return self


class StatusEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event: str = "status"
    turn_id: str
    status: str


class ErrorEvent(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event: str = "error"
    turn_id: str
    code: str
    message: str


class Timings(BaseModel):
    transcription_ms: float = Field(default=0, ge=0)
    audio_ms: float = Field(ge=0)
    text_ms: float = Field(ge=0)
    classifier_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class PredictionResult(BaseModel):
    schema_version: str = SCHEMA_VERSION
    event: str = "result"
    turn_id: str
    transcript: str
    transcript_source: str
    emotion: str
    probabilities: dict[str, float]
    uncertain: bool
    response_text: str
    policy_id: str
    gesture_id: str
    timings: Timings

    def for_turn(self, turn_id: str) -> PredictionResult:
        """Return a result carrying the caller's turn identifier."""
        return self.model_copy(update={"turn_id": turn_id})

    @model_validator(mode="after")
    def validate_probabilities(self) -> PredictionResult:
        if tuple(self.probabilities) != EMOTIONS:
            raise ValueError("Probabilities must use the stable seven-emotion order")
        if any(not 0 <= value <= 1 for value in self.probabilities.values()):
            raise ValueError("Probabilities must be in [0, 1]")
        if abs(sum(self.probabilities.values()) - 1.0) > 1e-5:
            raise ValueError("Probabilities must sum to one")
        if self.emotion not in self.probabilities:
            raise ValueError("Predicted emotion must be a known label")
        return self
