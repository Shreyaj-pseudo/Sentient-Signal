from pathlib import Path

import numpy as np

from sentient_signals.audio import pcm16_bytes_to_float32, read_pcm16_wav, write_pcm16_wav


def test_pcm16_round_trip(tmp_path: Path) -> None:
    samples = np.linspace(-0.8, 0.8, 1600, dtype=np.float32)
    path = tmp_path / "clip.wav"

    write_pcm16_wav(path, samples)
    decoded = read_pcm16_wav(path)

    assert decoded.shape == samples.shape
    assert np.max(np.abs(samples - decoded)) < 1e-4


def test_pcm_bytes_require_nonempty_complete_samples() -> None:
    import pytest

    with pytest.raises(ValueError):
        pcm16_bytes_to_float32(b"")
    with pytest.raises(ValueError):
        pcm16_bytes_to_float32(b"x")


def test_real_meld_clip_decodes_to_mono_16khz() -> None:
    pytest = __import__("pytest")
    pytest.importorskip("av")
    from sentient_signals.audio import decode_audio

    path = Path("MELD.Raw/dev/dev_splits_complete/dia0_utt0.mp4")
    samples = decode_audio(path)

    assert samples.dtype == np.float32
    assert samples.ndim == 1
    assert 16_000 < len(samples) < 160_000
    assert np.isfinite(samples).all()
