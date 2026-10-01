from scripts.day3_accuracy_experiment import choose_weight


def test_choose_weight_uses_seed_mean_and_deterministic_tie_break() -> None:
    assert choose_weight({0.75: [0.4, 0.5], 0.8: [0.46, 0.46]}) == 0.8
    assert choose_weight({0.75: [0.4], 0.8: [0.4]}) == 0.8
