from pathlib import Path

import pytest

from ramei_ml.data.labels import EMOTIONS, NUM_CLASSES, map_label
from ramei_ml.data.manifest import (
    assert_speaker_independent,
    build_manifest,
    split_summary,
)
from ramei_ml.data.parsers import (
    parse_crema_d,
    parse_ravdess,
    parse_savee,
    parse_tess,
    scan_dataset,
)


def test_label_space_has_seven_classes() -> None:
    assert NUM_CLASSES == 7
    assert EMOTIONS[0] == "neutral"


def test_map_label() -> None:
    assert map_label("ravdess", "02") == "neutral"  # calm folds into neutral
    assert map_label("tess", "ps") == "surprise"
    assert map_label("crema_d", "XYZ") is None
    with pytest.raises(ValueError):
        map_label("unknown_ds", "x")


@pytest.mark.parametrize(
    ("name", "emotion", "speaker", "gender", "media"),
    [
        ("03-01-05-01-02-01-12.wav", "angry", "ravdess:12", "female", "audio"),
        ("01-01-08-02-01-02-07.mp4", "surprise", "ravdess:07", "male", "video"),
    ],
)
def test_parse_ravdess(name: str, emotion: str, speaker: str, gender: str, media: str) -> None:
    rec = parse_ravdess(Path(name))
    assert rec is not None
    assert (rec.emotion, rec.speaker_id, rec.gender, rec.media) == (emotion, speaker, gender, media)


def test_parse_ravdess_rejects_song_and_silent_video() -> None:
    assert parse_ravdess(Path("03-02-05-01-02-01-12.wav")) is None  # song
    assert parse_ravdess(Path("02-01-05-01-02-01-12.mp4")) is None  # video-only, no audio
    assert parse_ravdess(Path("random.wav")) is None


def test_parse_other_datasets() -> None:
    crema = parse_crema_d(Path("1001_DFA_ANG_XX.wav"), genders={"1001": "male"})
    assert crema is not None and crema.emotion == "angry" and crema.gender == "male"

    tess = parse_tess(Path("OAF_back_ps.wav"))
    assert tess is not None and tess.emotion == "surprise" and tess.speaker_id == "tess:OAF"

    flat = parse_savee(Path("JK_sa05.wav"))
    nested = parse_savee(Path("AudioData/DC/su12.wav"))
    assert flat is not None and flat.emotion == "sad" and flat.speaker_id == "savee:JK"
    assert nested is not None and nested.emotion == "surprise" and nested.speaker_id == "savee:DC"


def test_speaker_independent_manifest(synthetic_ravdess: Path) -> None:
    df = build_manifest({"ravdess": synthetic_ravdess}, strategy="speaker", seed=1)
    assert len(df) == 12 * 7 * 2  # song files and notes.txt excluded
    assert set(df["split"]) == {"train", "val", "test"}
    assert_speaker_independent(df)
    # stratified by gender: every split has both genders
    assert (df.groupby("split")["gender"].nunique() == 2).all()
    summary = split_summary(df)
    assert summary["total"].sum() == len(df)


def test_manifest_is_deterministic(synthetic_ravdess: Path) -> None:
    a = build_manifest({"ravdess": synthetic_ravdess}, seed=7)
    b = build_manifest({"ravdess": synthetic_ravdess}, seed=7)
    assert a.equals(b)


def test_random_split_leaks_speakers(synthetic_ravdess: Path) -> None:
    df = build_manifest({"ravdess": synthetic_ravdess}, strategy="random", seed=1)
    with pytest.raises(AssertionError):
        assert_speaker_independent(df)


def test_crema_d_scan_reads_genders_from_demographics(tmp_path: Path) -> None:
    import soundfile as sf

    audio = tmp_path / "AudioWAV"
    audio.mkdir()
    for name in ("1001_DFA_ANG_XX.wav", "1002_IEO_HAP_HI.wav", "1003_ITH_CAL_XX.wav"):
        sf.write(audio / name, [0.0] * 160, 16_000)
    (tmp_path / "VideoDemographics.csv").write_text(
        "ActorID,Age,Sex,Race,Ethnicity\n1001,51,Male,Caucasian,Not Hispanic\n"
        "1002,21,Female,Caucasian,Not Hispanic\n"
    )
    records = {r.speaker_id: r for r in scan_dataset("crema_d", tmp_path)}
    assert set(records) == {"crema_d:1001", "crema_d:1002"}  # CAL is not a CREMA-D emotion
    assert records["crema_d:1001"].gender == "male"
    assert records["crema_d:1002"].gender == "female"
