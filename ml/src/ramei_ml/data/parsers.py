"""Filename parsers that turn raw dataset files into unified records.

Each parser takes a file path and returns a ``Record`` or ``None`` (file not usable).
Speaker ids are prefixed with the dataset name so they are globally unique.
"""

import re
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path

from ramei_ml.data.labels import map_label

AUDIO_EXTS = {".wav", ".flac", ".mp3"}
VIDEO_EXTS = {".mp4", ".avi", ".flv"}


@dataclass(frozen=True)
class Record:
    path: str
    dataset: str
    speaker_id: str
    gender: str | None  # "male" | "female" | None (unknown)
    emotion: str
    raw_label: str
    media: str  # "audio" | "video"

    def to_dict(self) -> dict[str, str | None]:
        return asdict(self)


def _media_kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in AUDIO_EXTS:
        return "audio"
    if ext in VIDEO_EXTS:
        return "video"
    return None


# ----------------------------------------------------------------------------- RAVDESS
# 03-01-05-01-02-01-12.wav = modality-vocal_channel-emotion-intensity-statement-repetition-actor
# modality: 01 full-AV, 02 video-only, 03 audio-only; vocal_channel: 01 speech, 02 song
_RAVDESS_RE = re.compile(r"^(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})-(\d{2})$")


def parse_ravdess(path: Path) -> Record | None:
    media = _media_kind(path)
    m = _RAVDESS_RE.match(path.stem)
    if media is None or m is None:
        return None
    modality, vocal_channel, emotion_code, *_, actor = m.groups()
    if vocal_channel != "01":  # speech only; song is out of scope
        return None
    if modality == "02":  # video without audio is unusable for speech emotion
        return None
    emotion = map_label("ravdess", emotion_code)
    if emotion is None:
        return None
    gender = "male" if int(actor) % 2 == 1 else "female"
    return Record(str(path), "ravdess", f"ravdess:{actor}", gender, emotion, emotion_code, media)


# ----------------------------------------------------------------------------- CREMA-D
# 1001_DFA_ANG_XX.wav = actor_sentence_emotion_level
_CREMA_RE = re.compile(r"^(\d{4})_([A-Z]{3})_([A-Z]{3})_([A-Z]{2})$")


def parse_crema_d(path: Path, genders: dict[str, str] | None = None) -> Record | None:
    media = _media_kind(path)
    m = _CREMA_RE.match(path.stem)
    if media is None or m is None:
        return None
    actor, _sentence, code, _level = m.groups()
    emotion = map_label("crema_d", code)
    if emotion is None:
        return None
    gender = (genders or {}).get(actor)
    return Record(str(path), "crema_d", f"crema_d:{actor}", gender, emotion, code, media)


# ----------------------------------------------------------------------------- TESS
# OAF_back_angry.wav / YAF_dog_ps.wav  (two female speakers: OAF, YAF)
_TESS_RE = re.compile(r"^(OAF|YAF)_[A-Za-z]+_([A-Za-z]+)$", re.IGNORECASE)


def parse_tess(path: Path) -> Record | None:
    media = _media_kind(path)
    m = _TESS_RE.match(path.stem)
    if media is None or m is None:
        return None
    speaker, word = m.group(1).upper(), m.group(2).lower()
    emotion = map_label("tess", word)
    if emotion is None:
        return None
    return Record(str(path), "tess", f"tess:{speaker}", "female", emotion, word, media)


# ----------------------------------------------------------------------------- SAVEE
# Flat: DC_a01.wav   |   Original: AudioData/DC/a01.wav   (four male speakers)
_SAVEE_SPEAKERS = {"DC", "JE", "JK", "KL"}
_SAVEE_FLAT_RE = re.compile(r"^(DC|JE|JK|KL)_([a-z]+)\d+$")
_SAVEE_NESTED_RE = re.compile(r"^([a-z]+)\d+$")


def parse_savee(path: Path) -> Record | None:
    media = _media_kind(path)
    if media is None:
        return None
    m = _SAVEE_FLAT_RE.match(path.stem)
    if m:
        speaker, code = m.groups()
    else:
        m = _SAVEE_NESTED_RE.match(path.stem)
        if m is None or path.parent.name not in _SAVEE_SPEAKERS:
            return None
        speaker, code = path.parent.name, m.group(1)
    emotion = map_label("savee", code)
    if emotion is None:
        return None
    return Record(str(path), "savee", f"savee:{speaker}", "male", emotion, code, media)


PARSERS: dict[str, Callable[[Path], Record | None]] = {
    "ravdess": parse_ravdess,
    "crema_d": parse_crema_d,
    "tess": parse_tess,
    "savee": parse_savee,
}


def scan_dataset(name: str, root: Path, media: str | None = "audio") -> Iterator[Record]:
    """Yield records for every parseable file under ``root`` (optionally filtered by media)."""
    try:
        parser = PARSERS[name]
    except KeyError as exc:
        raise ValueError(f"No parser for dataset '{name}'. Known: {sorted(PARSERS)}") from exc
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        record = parser(path)
        if record is not None and (media is None or record.media == media):
            yield record
