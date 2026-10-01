import numpy as np

from sentient_signals.features import (
    PROSODY_FEATURES,
    FeatureExtractor,
    acoustic_summary,
    prosody_features,
    temporal_audio_bins,
)


def test_feasibility_features_are_finite_and_repeatable() -> None:
    extractor = FeatureExtractor("feasibility")
    first = extractor.encode_text("This is surprisingly good")
    second = extractor.encode_text("This is surprisingly good")
    audio = acoustic_summary(np.sin(np.linspace(0, 50, 16_000)).astype(np.float32))

    np.testing.assert_array_equal(first, second)
    assert first.shape == (256,)
    assert audio.shape == (32,)
    assert np.isfinite(audio).all()


def test_emotion_audio_encoder_has_pinned_task_specific_features() -> None:
    extractor = FeatureExtractor(audio_encoder="emotion")

    assert extractor.audio_dimension == 512
    assert extractor.audio_config["audio_model"] == "superb/wav2vec2-base-superb-er"
    assert len(extractor.audio_config["audio_revision"]) == 40


def test_temporal_audio_bins_preserve_order_and_mask_padding() -> None:
    frames = np.arange(18, dtype=np.float32).reshape(6, 3)

    values, mask = temporal_audio_bins(frames, bins=4)

    np.testing.assert_array_equal(mask, [True, True, True, True])
    np.testing.assert_array_equal(values[0], frames[:2].mean(axis=0))
    np.testing.assert_array_equal(values[-1], frames[-1])


def test_prosody_features_are_finite_and_detect_pitch() -> None:
    time = np.arange(16_000, dtype=np.float32) / 16_000
    audio = 0.5 * np.sin(2 * np.pi * 200 * time)

    features = prosody_features(audio)

    assert features.shape == (len(PROSODY_FEATURES),)
    assert np.isfinite(features).all()
    assert features[11] > 0
    assert features[17] > 0
