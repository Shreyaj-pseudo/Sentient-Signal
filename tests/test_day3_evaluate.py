import numpy as np

from scripts.day3_evaluate import dialogue_bootstrap


def test_dialogue_bootstrap_detects_better_fused_predictions() -> None:
    labels = np.asarray([0, 1, 0, 1])
    fused = np.eye(7)[labels]
    text = np.eye(7)[np.asarray([1, 0, 1, 0])]
    clip_ids = np.asarray(["dia0_utt0", "dia0_utt1", "dia1_utt0", "dia1_utt1"])

    result = dialogue_bootstrap(labels, fused, text, clip_ids, samples=20)

    assert result["ci95_low"] > 0
