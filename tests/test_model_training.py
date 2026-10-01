from pathlib import Path

import numpy as np
import torch

from sentient_signals.model import AttentiveAudioHead, GatedFusionHead, fit_head


def write_features(path: Path, examples: int) -> None:
    rng = np.random.default_rng(7)
    np.savez_compressed(
        path,
        text=rng.normal(size=(examples, 4)).astype(np.float32),
        audio=rng.normal(size=(examples, 6)).astype(np.float32),
        labels=np.arange(examples) % 7,
        clip_ids=np.asarray([f"dia{index // 2}_utt{index % 2}" for index in range(examples)]),
        feature_mode=np.asarray("feasibility"),
    )


def test_fit_head_supports_unimodal_weighted_training(tmp_path: Path) -> None:
    train = tmp_path / "train.npz"
    dev = tmp_path / "dev.npz"
    output = tmp_path / "head.pt"
    write_features(train, 28)
    write_features(dev, 28)

    report = fit_head(
        train,
        dev,
        output,
        epochs=1,
        modality="text",
        class_weighting="inverse_sqrt",
    )

    assert output.is_file()
    assert report["modality"] == "text"
    assert report["class_weighting"] == "inverse_sqrt"
    assert report["dev_examples"] < 28


def test_attentive_audio_head_ignores_padded_bins() -> None:
    torch.manual_seed(1)
    model = AttentiveAudioHead(3).eval()
    frames = torch.randn(2, 4, 3)
    mask = torch.tensor([[True, True, False, False], [True, True, True, True]])
    changed = frames.clone()
    changed[0, 2:] = 1_000

    with torch.inference_mode():
        original = model(frames, mask)
        modified = model(changed, mask)

    torch.testing.assert_close(original[0], modified[0])


def test_gated_fusion_returns_normalized_probabilities() -> None:
    model = GatedFusionHead(4, 6).eval()
    with torch.inference_mode():
        log_probabilities, _text, _audio, gate = model(torch.randn(3, 4), torch.randn(3, 6))

    torch.testing.assert_close(log_probabilities.exp().sum(1), torch.ones(3))
    assert torch.all((gate > 0) & (gate < 1))
