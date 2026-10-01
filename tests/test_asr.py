from sentient_signals.asr import word_error_rate


def test_word_error_rate_exact_match() -> None:
    assert word_error_rate("Hello, world!", "hello world") == 0


def test_word_error_rate_counts_substitution() -> None:
    assert word_error_rate("we are very happy", "we are very sad") == 0.25


def test_word_error_rate_handles_empty_reference() -> None:
    assert word_error_rate("", "") == 0
    assert word_error_rate("", "unexpected words") == 1
