"""Local English speech transcription and simple transcript evaluation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sentient_signals.audio import read_pcm16_wav
from sentient_signals.data import normalize_meld_text

ASR_MODEL = "tiny.en"
WHISPER_TINY_EN_PARAMETERS = 37_760_256


@dataclass(frozen=True)
class Transcription:
    text: str
    language: str
    language_probability: float
    duration_seconds: float


class LocalTranscriber:
    """A reusable faster-whisper model configured for CPU INT8 inference."""

    def __init__(
        self,
        model_name: str = ASR_MODEL,
        cpu_threads: int = 8,
        local_files_only: bool = False,
    ) -> None:
        from faster_whisper import WhisperModel

        self.model_name = model_name
        self.model = WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",
            cpu_threads=cpu_threads,
            num_workers=1,
            local_files_only=local_files_only,
        )

    def transcribe(self, audio_path: Path) -> Transcription:
        samples = read_pcm16_wav(audio_path)
        segments, info = self.model.transcribe(
            samples,
            language="en",
            beam_size=1,
            best_of=1,
            condition_on_previous_text=False,
            vad_filter=False,
            word_timestamps=False,
        )
        text = " ".join(segment.text.strip() for segment in segments).strip()
        return Transcription(
            text=text,
            language=info.language,
            language_probability=float(info.language_probability),
            duration_seconds=float(info.duration),
        )


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Return word-level edit distance divided by the reference word count."""
    reference_words = re.findall(r"[\w']+", normalize_meld_text(reference).lower())
    hypothesis_words = re.findall(r"[\w']+", normalize_meld_text(hypothesis).lower())
    if not reference_words:
        return 0.0 if not hypothesis_words else 1.0
    previous = list(range(len(hypothesis_words) + 1))
    for row, reference_word in enumerate(reference_words, 1):
        current = [row]
        for column, hypothesis_word in enumerate(hypothesis_words, 1):
            substitution = previous[column - 1] + (reference_word != hypothesis_word)
            current.append(min(previous[column] + 1, current[column - 1] + 1, substitution))
        previous = current
    return previous[-1] / len(reference_words)
