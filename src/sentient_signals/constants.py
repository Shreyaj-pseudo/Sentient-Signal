"""Stable runtime and training constants."""

SCHEMA_VERSION = "1.0"
SAMPLE_RATE = 16_000
MAX_LIVE_SECONDS = 10.0
EMOTIONS = ("anger", "disgust", "fear", "joy", "neutral", "sadness", "surprise")
LABEL_TO_INDEX = {label: index for index, label in enumerate(EMOTIONS)}
INDEX_TO_LABEL = dict(enumerate(EMOTIONS))
