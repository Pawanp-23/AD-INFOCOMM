"""Session intelligence: alert rules, timeline and a data-driven summary (TRD §4.9)."""

from collections import Counter
from typing import Any

NEGATIVE = {"angry", "sad", "fear", "disgust"}


def compute_alerts(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rules over consecutive segments:
    - sustained_anger:   'angry' for >= 20 s of speech (critical)
    - negative_valence:  valence < -0.4 for >= 30 s (warning)
    - low_reliability:   >= 3 consecutive abstentions (info)
    """
    alerts: list[dict[str, Any]] = []

    def runs(pred):
        start = None
        for i, s in enumerate(segments + [None]):
            if s is not None and pred(s):
                start = i if start is None else start
            elif start is not None:
                yield segments[start], segments[i - 1], i - start
                start = None

    for first, last, _ in runs(lambda s: s["emotion"] == "angry"):
        if last["end_ms"] - first["start_ms"] >= 20_000:
            alerts.append(
                {
                    "rule": "sustained_anger",
                    "severity": "critical",
                    "start_ms": first["start_ms"],
                    "end_ms": last["end_ms"],
                    "message": "Sustained anger for "
                    f"{(last['end_ms'] - first['start_ms']) / 1000:.0f} s",
                }
            )
    for first, last, _ in runs(lambda s: (s.get("valence") or 0) < -0.4):
        if last["end_ms"] - first["start_ms"] >= 30_000:
            alerts.append(
                {
                    "rule": "negative_valence",
                    "severity": "warning",
                    "start_ms": first["start_ms"],
                    "end_ms": last["end_ms"],
                    "message": "Prolonged negative mood",
                }
            )
    for first, last, n in runs(lambda s: s["abstained"]):
        if n >= 3:
            alerts.append(
                {
                    "rule": "low_reliability",
                    "severity": "info",
                    "start_ms": first["start_ms"],
                    "end_ms": last["end_ms"],
                    "message": f"{n} consecutive uncertain segments — check audio/video",
                }
            )
    return sorted(alerts, key=lambda a: a["start_ms"])


def build_summary(segments: list[dict[str, Any]], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """Deterministic summary from the analysis (used when no LLM provider is configured)."""
    if not segments:
        return {"headline": "No speech or faces detected.", "points": [], "distribution": {}}
    decided = [s for s in segments if not s["abstained"]]
    dist = Counter(s["emotion"] for s in segments)
    total_ms = sum(s["end_ms"] - s["start_ms"] for s in segments)
    share = {
        e: round(
            sum(s["end_ms"] - s["start_ms"] for s in segments if s["emotion"] == e)
            / max(total_ms, 1),
            3,
        )
        for e in dist
    }
    dominant = max((e for e in share if e != "uncertain"), key=share.get, default="uncertain")
    neg = sum(v for e, v in share.items() if e in NEGATIVE)
    mean_val = sum(s.get("valence") or 0 for s in decided) / max(len(decided), 1)
    reliab = {
        m: round(
            sum(
                s["modalities"][m]["reliability"]
                for s in segments
                if s["modalities"][m]["available"]
            )
            / max(1, sum(s["modalities"][m]["available"] for s in segments)),
            3,
        )
        for m in ("audio", "text", "visual")
    }
    trend = "stable"
    if len(decided) >= 4:
        half = len(decided) // 2
        a = sum(s["valence"] for s in decided[:half]) / half
        b = sum(s["valence"] for s in decided[half:]) / (len(decided) - half)
        trend = "improving" if b - a > 0.15 else "worsening" if a - b > 0.15 else "stable"
    points = [
        f"Dominant emotion: {dominant} ({share.get(dominant, 0) * 100:.0f}% of speech time).",
        f"Negative emotions covered {neg * 100:.0f}% of the session; mood trend is {trend}.",
        f"{len(alerts)} alert(s) raised." if alerts else "No alerts raised.",
        "Average modality reliability — "
        + ", ".join(f"{m}: {v * 100:.0f}%" for m, v in reliab.items())
        + ".",
    ]
    if share.get("uncertain"):
        points.append(f"{share['uncertain'] * 100:.0f}% of segments were too uncertain to label.")
    tone = "positive" if mean_val > 0.15 else "negative" if mean_val < -0.15 else "neutral"
    return {
        "headline": f"Overall {tone} session, mostly {dominant}.",
        "points": points,
        "distribution": share,
        "dominant": dominant,
        "trend": trend,
        "mean_valence": round(mean_val, 3),
        "mean_reliability": reliab,
    }
