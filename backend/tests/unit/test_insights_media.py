import numpy as np

from app.pipeline.media import SR, fixed_windows, speech_segments, words_in
from app.services.insights import build_summary, compute_alerts

MODS = {
    m: {
        "available": True,
        "reliability": 0.8,
        "weight": 1 / 3,
        "emotion": "angry",
        "confidence": 0.9,
    }
    for m in ("audio", "text", "visual")
}


def seg(i: int, emotion: str, valence: float = 0.0, abstained: bool = False) -> dict:
    return {
        "start_ms": i * 3000,
        "end_ms": (i + 1) * 3000,
        "emotion": emotion,
        "valence": valence,
        "abstained": abstained,
        "modalities": MODS,
    }


def test_sustained_anger_alert_fires_after_20_seconds() -> None:
    assert compute_alerts([seg(i, "angry", -0.6) for i in range(6)]) == []  # 18 s
    alerts = compute_alerts([seg(i, "angry", -0.6) for i in range(7)])  # 21 s
    assert [a["rule"] for a in alerts] == ["sustained_anger"]
    assert alerts[0]["severity"] == "critical"


def test_negative_valence_and_reliability_alerts() -> None:
    segs = [seg(i, "sad", -0.7) for i in range(10)]
    segs += [seg(10 + i, "uncertain", 0, True) for i in range(3)]
    assert {a["rule"] for a in compute_alerts(segs)} == {"negative_valence", "low_reliability"}


def test_summary() -> None:
    s = build_summary([seg(0, "happy", 0.8), seg(1, "happy", 0.7), seg(2, "neutral", 0.0)], [])
    assert s["dominant"] == "happy"
    assert abs(sum(s["distribution"].values()) - 1.0) < 1e-6
    assert build_summary([], [])["points"] == []


def test_speech_segments_split_on_silence() -> None:
    t = np.arange(int(1.5 * SR)) / SR
    tone = 0.3 * np.sin(2 * np.pi * 200 * t)
    gap = np.zeros(SR)
    audio = np.concatenate([gap, tone, gap, tone, gap]).astype(np.float32)
    windows = speech_segments(audio)
    assert len(windows) == 2
    assert 0.8 < windows[0].start_s < 1.2 and 3.3 < windows[1].start_s < 3.7


def test_long_speech_is_chunked_and_words_assigned() -> None:
    audio = (0.3 * np.sin(2 * np.pi * 200 * np.arange(14 * SR) / SR)).astype(np.float32)
    windows = speech_segments(audio)
    assert len(windows) == 3 and all(w.end_s - w.start_s <= 6.0 + 1e-6 for w in windows)
    words = [{"start": 0.1, "end": 0.4, "word": "hi"}, {"start": 5.0, "end": 5.4, "word": "x"}]
    assert [x["word"] for x in words_in(words, windows[0])] == ["hi"]
    assert len(fixed_windows(10.0)) == 3
