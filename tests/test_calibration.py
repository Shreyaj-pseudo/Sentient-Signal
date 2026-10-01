import numpy as np
import pytest

from sentient_signals.calibration import (
    apply_temperature,
    dialogue_calibration_mask,
    fit_temperature,
    fuse_probabilities,
)


def test_calibration_split_keeps_dialogues_together_and_is_stable() -> None:
    clips = np.asarray(["dia1_utt0", "dia1_utt1", "dia2_utt0", "dia3_utt0", "dia4_utt0"])
    first = dialogue_calibration_mask(clips)
    second = dialogue_calibration_mask(clips)
    assert np.array_equal(first, second)
    assert first[0] == first[1]
    assert first.any() and not first.all()


def test_probability_fusion_is_normalized() -> None:
    text = np.asarray([[0.8, 0.2]], dtype=np.float32)
    audio = np.asarray([[0.1, 0.9]], dtype=np.float32)
    combined = fuse_probabilities(text, audio, 0.75)
    assert combined.sum() == pytest.approx(1)
    assert combined[0, 0] == pytest.approx(0.625)


def test_temperature_is_positive_and_outputs_probabilities() -> None:
    logits = np.asarray([[4.0, 0.0], [0.0, 4.0], [3.0, 1.0]], dtype=np.float32)
    labels = np.asarray([0, 1, 0])
    temperature = fit_temperature(logits, labels)
    probabilities = apply_temperature(logits, temperature)
    assert temperature > 0
    assert np.allclose(probabilities.sum(axis=1), 1)
