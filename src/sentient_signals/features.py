"""Frozen encoder and explicitly-labelled feasibility feature extraction."""

from __future__ import annotations

import hashlib
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sentient_signals.audio import read_pcm16_wav

TEXT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
AUDIO_MODEL = "facebook/wav2vec2-base"
EMOTION_AUDIO_MODEL = "superb/wav2vec2-base-superb-er"
TEXT_REVISION = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
AUDIO_REVISION = "0b5b8e868dd84f03fd87d01f9c4ff0f080fecfe8"
EMOTION_AUDIO_REVISION = "441a7599c3b22107314dcbd9166621c5c83f2cc5"
PROSODY_FEATURES = (
    "duration",
    "rms_mean",
    "rms_std",
    "rms_p10",
    "rms_p90",
    "rms_range",
    "rms_slope",
    "zcr_mean",
    "zcr_std",
    "centroid_mean",
    "centroid_std",
    "log_pitch_mean",
    "log_pitch_std",
    "log_pitch_p10",
    "log_pitch_p90",
    "log_pitch_range",
    "log_pitch_slope",
    "voiced_ratio",
    "pause_ratio",
    "pause_events_per_second",
    "longest_pause",
)


@dataclass(frozen=True)
class FeatureTimings:
    audio_ms: float
    text_ms: float


class FeatureExtractor:
    """One shared extraction implementation for cache and runtime inference."""

    def __init__(
        self,
        mode: str = "encoders",
        device: str = "cpu",
        local_files_only: bool = False,
        audio_encoder: str = "speech",
    ) -> None:
        if mode not in {"encoders", "feasibility"}:
            raise ValueError("mode must be 'encoders' or 'feasibility'")
        if audio_encoder not in {"speech", "emotion"}:
            raise ValueError("audio_encoder must be 'speech' or 'emotion'")
        self.mode = mode
        self.device = device
        self.local_files_only = local_files_only
        self.audio_encoder = audio_encoder
        self._text_model = None
        self._audio_processor = None
        self._audio_model = None

    @property
    def text_dimension(self) -> int:
        return 384 if self.mode == "encoders" else 256

    @property
    def audio_dimension(self) -> int:
        if self.mode == "feasibility":
            return 32
        return 512 if self.audio_encoder == "emotion" else 1536

    @property
    def audio_frame_dimension(self) -> int:
        if self.mode != "encoders":
            raise ValueError("Frame features require encoder mode")
        return 256 if self.audio_encoder == "emotion" else 768

    @property
    def audio_config(self) -> dict[str, str]:
        if self.audio_encoder == "emotion":
            return {
                "audio_encoder": "emotion",
                "audio_model": EMOTION_AUDIO_MODEL,
                "audio_revision": EMOTION_AUDIO_REVISION,
                "audio_pooling": "classification_projector_mean_population_std",
            }
        return {
            "audio_encoder": "speech",
            "audio_model": AUDIO_MODEL,
            "audio_revision": AUDIO_REVISION,
            "audio_pooling": "last_hidden_mean_population_std",
        }

    def _load_text(self) -> None:
        if self._text_model is None:
            from sentence_transformers import SentenceTransformer

            self._text_model = SentenceTransformer(
                TEXT_MODEL,
                revision=TEXT_REVISION,
                device=self.device,
                local_files_only=self.local_files_only,
            )

    def _load_audio(self) -> None:
        if self._audio_model is None:
            from transformers import AutoFeatureExtractor

            config = self.audio_config
            kwargs = {
                "revision": config["audio_revision"],
                "local_files_only": self.local_files_only,
            }
            self._audio_processor = AutoFeatureExtractor.from_pretrained(
                config["audio_model"], **kwargs
            )
            if self.audio_encoder == "emotion":
                from transformers import AutoModelForAudioClassification

                self._audio_model = AutoModelForAudioClassification.from_pretrained(
                    config["audio_model"], **kwargs
                )
            else:
                from transformers import Wav2Vec2Model

                self._audio_model = Wav2Vec2Model.from_pretrained(config["audio_model"], **kwargs)
            self._audio_model = self._audio_model.to(self.device).eval()

    def encode_text(self, text: str) -> np.ndarray:
        if self.mode == "feasibility":
            vector = np.zeros(self.text_dimension, dtype=np.float32)
            for token in re.findall(r"[\w']+", text.lower()):
                digest = hashlib.blake2b(token.encode(), digest_size=8).digest()
                value = int.from_bytes(digest, "little")
                vector[value % vector.size] += -1.0 if value & 1 else 1.0
            norm = np.linalg.norm(vector)
            return vector / norm if norm else vector
        self._load_text()
        return np.asarray(
            self._text_model.encode(text, normalize_embeddings=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def encode_audio(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("Audio must contain finite samples")
        if self.mode == "feasibility":
            return acoustic_summary(samples)
        hidden = self.encode_audio_frames(samples)
        mean = hidden.mean(axis=0, dtype=np.float64)
        variance = np.maximum(np.square(hidden, dtype=np.float64).mean(axis=0) - mean**2, 0)
        return np.concatenate([mean, np.sqrt(variance)]).astype(np.float32)

    def encode_audio_frames(self, samples: np.ndarray) -> np.ndarray:
        if samples.size == 0 or not np.isfinite(samples).all():
            raise ValueError("Audio must contain finite samples")
        if self.mode == "feasibility":
            raise ValueError("Frame features require encoder mode")
        self._load_audio()
        import torch

        if len(samples) < 400:
            samples = np.pad(samples, (0, 400 - len(samples)))

        windows = np.array_split(samples, max(1, math.ceil(len(samples) / 160_000)))
        frames = []
        with torch.inference_mode():
            for window in windows:
                inputs = self._audio_processor(
                    window, sampling_rate=16_000, return_tensors="pt"
                ).input_values.to(self.device)
                if self.audio_encoder == "emotion":
                    hidden = self._audio_model.wav2vec2(inputs).last_hidden_state
                    hidden = self._audio_model.projector(hidden)
                else:
                    hidden = self._audio_model(inputs).last_hidden_state
                frames.append(hidden[0].float().cpu().numpy())
        return np.concatenate(frames).astype(np.float32)

    def encode(self, text: str, audio_path: Path) -> tuple[np.ndarray, np.ndarray, FeatureTimings]:
        samples = read_pcm16_wav(audio_path)
        started = time.perf_counter()
        audio = self.encode_audio(samples)
        audio_ms = (time.perf_counter() - started) * 1000
        started = time.perf_counter()
        text_feature = self.encode_text(text)
        text_ms = (time.perf_counter() - started) * 1000
        return text_feature, audio, FeatureTimings(audio_ms, text_ms)


def acoustic_summary(samples: np.ndarray) -> np.ndarray:
    """Compact deterministic smoke-test features; not the planned audio encoder."""
    if samples.size == 0:
        raise ValueError("Cannot encode empty audio")
    chunks = np.array_split(samples, 8)
    values: list[float] = []
    for chunk in chunks:
        values.extend(
            [
                float(np.mean(chunk)),
                float(np.std(chunk)),
                float(np.sqrt(np.mean(np.square(chunk)))),
                float(np.mean(np.abs(np.diff(np.signbit(chunk))))),
            ]
        )
    return np.asarray(values, dtype=np.float32)


def temporal_audio_bins(frames: np.ndarray, bins: int = 32) -> tuple[np.ndarray, np.ndarray]:
    """Compress a variable frame sequence while preserving its temporal order."""
    if frames.ndim != 2 or not len(frames) or bins < 1 or not np.isfinite(frames).all():
        raise ValueError("frames must be a non-empty finite matrix and bins must be positive")
    used = min(len(frames), bins)
    output = np.zeros((bins, frames.shape[1]), dtype=np.float32)
    for index, chunk in enumerate(np.array_split(frames, used)):
        output[index] = chunk.mean(axis=0)
    mask = np.arange(bins) < used
    return output, mask


def prosody_features(samples: np.ndarray, sample_rate: int = 16_000) -> np.ndarray:
    """Dependency-free pitch, energy, rhythm, and spectral delivery features."""
    if samples.size == 0 or sample_rate < 1 or not np.isfinite(samples).all():
        raise ValueError("Audio must contain finite samples at a positive sample rate")
    frame_size, hop = round(sample_rate * 0.025), round(sample_rate * 0.01)
    if len(samples) < frame_size:
        samples = np.pad(samples, (0, frame_size - len(samples)))
    frames = np.lib.stride_tricks.sliding_window_view(samples, frame_size)[::hop]
    windowed = frames * np.hanning(frame_size)
    rms = np.sqrt(np.mean(np.square(frames, dtype=np.float64), axis=1))
    zcr = np.mean(np.diff(np.signbit(frames), axis=1), axis=1)
    spectrum = np.abs(np.fft.rfft(windowed, axis=1))
    frequencies = np.fft.rfftfreq(frame_size, 1 / sample_rate)
    centroid = (spectrum * frequencies).sum(1) / np.maximum(spectrum.sum(1), 1e-8)

    silence_threshold = max(float(rms.max()) * 0.05, 1e-4)
    pause = rms < silence_threshold
    transitions = np.diff(np.pad(pause.astype(np.int8), (1, 1)))
    boundaries = np.flatnonzero(transitions)
    pause_lengths = boundaries[1::2] - boundaries[::2]

    pitches = []
    pitch_positions = []
    minimum_lag, maximum_lag = sample_rate // 400, sample_rate // 50
    for index in range(0, len(windowed), 4):
        if rms[index] < silence_threshold:
            continue
        frame = windowed[index] - windowed[index].mean()
        spectrum_frame = np.fft.rfft(frame, n=1024)
        correlation = np.fft.irfft(np.abs(spectrum_frame) ** 2)
        if correlation[0] <= 1e-8:
            continue
        candidates = correlation[minimum_lag : maximum_lag + 1]
        lag = minimum_lag + int(candidates.argmax())
        if correlation[lag] / correlation[0] >= 0.3:
            pitches.append(np.log(sample_rate / lag))
            pitch_positions.append(index)

    def statistics(values: np.ndarray) -> list[float]:
        if not len(values):
            return [0.0] * 6
        slope = float(np.polyfit(np.arange(len(values)), values, 1)[0]) if len(values) > 1 else 0
        return [
            float(values.mean()),
            float(values.std()),
            float(np.percentile(values, 10)),
            float(np.percentile(values, 90)),
            float(np.ptp(values)),
            slope,
        ]

    log_pitch = np.asarray(pitches)
    duration = len(samples) / sample_rate
    features = [
        duration,
        *statistics(rms),
        float(zcr.mean()),
        float(zcr.std()),
        float(centroid.mean()),
        float(centroid.std()),
        *statistics(log_pitch),
        len(pitch_positions) / max(1, len(range(0, len(windowed), 4))),
        float(pause.mean()),
        len(pause_lengths) / duration,
        (float(pause_lengths.max()) * hop / sample_rate if len(pause_lengths) else 0.0),
    ]
    return np.asarray(features, dtype=np.float32)
