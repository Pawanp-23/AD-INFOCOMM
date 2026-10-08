"""Unified RAMEI label space and per-dataset label mappings (TRD §3.2)."""

EMOTIONS: tuple[str, ...] = ("neutral", "happy", "sad", "angry", "fear", "disgust", "surprise")
EMOTION_TO_ID: dict[str, int] = {e: i for i, e in enumerate(EMOTIONS)}
NUM_CLASSES = len(EMOTIONS)

# Raw dataset label -> unified label. Labels mapped to None are excluded.
DATASET_LABEL_MAPS: dict[str, dict[str, str | None]] = {
    # RAVDESS emotion code (3rd filename field)
    "ravdess": {
        "01": "neutral",
        "02": "neutral",  # calm -> neutral
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fear",
        "07": "disgust",
        "08": "surprise",
    },
    # CREMA-D emotion code (3rd filename field)
    "crema_d": {
        "NEU": "neutral",
        "HAP": "happy",
        "SAD": "sad",
        "ANG": "angry",
        "FEA": "fear",
        "DIS": "disgust",
    },
    # TESS emotion word (last filename token)
    "tess": {
        "neutral": "neutral",
        "happy": "happy",
        "sad": "sad",
        "angry": "angry",
        "fear": "fear",
        "disgust": "disgust",
        "ps": "surprise",  # pleasant surprise
    },
    # SAVEE emotion prefix
    "savee": {
        "n": "neutral",
        "h": "happy",
        "sa": "sad",
        "a": "angry",
        "f": "fear",
        "d": "disgust",
        "su": "surprise",
    },
}


def map_label(dataset: str, raw_label: str) -> str | None:
    """Map a dataset-specific label to the unified label space, or None if excluded."""
    try:
        mapping = DATASET_LABEL_MAPS[dataset]
    except KeyError as exc:
        raise ValueError(f"Unknown dataset '{dataset}'") from exc
    return mapping.get(raw_label)
