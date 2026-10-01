from pathlib import Path

import numpy as np
import pytest
import torch

from sentient_signals.calibration import apply_temperature, fuse_probabilities
from sentient_signals.model import FusionHead, InferencePipeline


def artifact(modality: str) -> dict:
    dim = {"text": 4, "audio": 6, "fusion": 10}[modality]
    return {
        "feature_mode": "feasibility",
        "modality": modality,
        "text_dimension": 4,
        "audio_dimension": 6,
        "normalization_mean": torch.zeros(dim),
        "normalization_scale": torch.ones(dim),
        "model_state": FusionHead(dim).eval().state_dict(),
    }


@pytest.mark.parametrize("modality", ["fusion", "probability"])
def test_deployed_calibration_matches_offline_distribution(tmp_path: Path, modality: str) -> None:
    torch.manual_seed(42)
    text = np.arange(4, dtype=np.float32)
    audio = np.arange(6, dtype=np.float32)
    if modality == "fusion":
        value = artifact("fusion")
        head = FusionHead(10).eval()
        head.load_state_dict(value["model_state"])
        with torch.inference_mode():
            raw = head(torch.from_numpy(np.concatenate([text, audio]))[None]).numpy()
    else:
        value = {
            "feature_mode": "feasibility",
            "modality": "probability",
            "text_dimension": 4,
            "audio_dimension": 6,
            "alpha": 0.75,
            "text_head": artifact("text"),
            "audio_head": artifact("audio"),
        }
        probabilities = []
        for mode, features in [("text", text), ("audio", audio)]:
            head = FusionHead(len(features)).eval()
            head.load_state_dict(value[f"{mode}_head"]["model_state"])
            with torch.inference_mode():
                probabilities.append(
                    apply_temperature(head(torch.from_numpy(features)[None]).numpy(), 1.0)
                )
        raw = np.log(fuse_probabilities(*probabilities, 0.75))
    value["temperature"] = 1.7
    path = tmp_path / "deployed.pt"
    torch.save(value, path)
    pipeline = InferencePipeline(path)
    actual = pipeline.predict_features(text, audio)
    assert np.allclose(actual, apply_temperature(raw, 1.7)[0], atol=1e-6)
    assert np.isfinite(actual).all()
    assert actual.sum() == pytest.approx(1)
