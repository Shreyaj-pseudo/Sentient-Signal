"""Shared audio decoding used by preparation and inference."""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from sentient_signals.constants import SAMPLE_RATE


def decode_audio(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    """Decode any PyAV-supported media into finite mono float32 samples."""
    import av

    chunks: list[np.ndarray] = []
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError(f"No audio stream in {path}")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(stream):
            converted = resampler.resample(frame)
            for output in converted if isinstance(converted, list) else [converted]:
                if output is not None:
                    chunks.append(output.to_ndarray().reshape(-1).astype(np.float32))
        for output in resampler.resample(None):
            chunks.append(output.to_ndarray().reshape(-1).astype(np.float32))
    if not chunks:
        raise ValueError(f"Decoded audio is empty: {path}")
    samples = np.concatenate(chunks)
    if not np.isfinite(samples).all():
        raise ValueError(f"Decoded audio contains non-finite samples: {path}")
    return np.clip(samples, -1.0, 1.0)


def write_pcm16_wav(path: Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).round().astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())


def pcm16_bytes_to_float32(data: bytes) -> np.ndarray:
    """Validate little-endian PCM16 bytes and convert them to normalized mono samples."""
    if not data:
        raise ValueError("Audio payload is empty")
    if len(data) % 2:
        raise ValueError("PCM16 payload must contain an even number of bytes")
    return np.frombuffer(data, dtype="<i2").astype(np.float32) / 32768.0


def read_pcm16_wav(path: Path, sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    with wave.open(str(path), "rb") as handle:
        if handle.getnchannels() != 1 or handle.getsampwidth() != 2:
            raise ValueError(f"Expected mono PCM16 WAV: {path}")
        if handle.getframerate() != sample_rate:
            raise ValueError(f"Expected {sample_rate} Hz WAV: {path}")
        return (
            np.frombuffer(handle.readframes(handle.getnframes()), dtype="<i2").astype(np.float32)
            / 32768
        )
