from app.core.config import get_settings
from app.services import narrative

SUMMARY = {
    "headline": "Overall negative session, mostly angry.",
    "distribution": {"angry": 0.6, "neutral": 0.4},
    "dominant": "angry",
    "trend": "worsening",
    "mean_valence": -0.4,
    "mean_reliability": {"audio": 0.8, "text": 0.6, "visual": 0.7},
}
ALERTS = [{"rule": "sustained_anger", "severity": "critical", "start_ms": 12_000, "end_ms": 40_000}]


def test_disabled_by_default_uses_rule_headline() -> None:
    out = narrative.add_narrative(SUMMARY, ALERTS)
    assert out["narrative"] == SUMMARY["headline"]
    assert out["narrative_source"] == "rules"
    assert out["dominant"] == "angry"  # original summary preserved


def test_provider_failure_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "llm_provider", "anthropic")

    def boom(_facts):
        raise RuntimeError("network down")

    monkeypatch.setattr(narrative, "_anthropic_narrative", boom)
    out = narrative.add_narrative(SUMMARY, ALERTS)
    assert out["narrative_source"] == "rules"
    assert out["narrative"] == SUMMARY["headline"]


def test_provider_success_and_only_aggregates_are_sent(monkeypatch) -> None:
    monkeypatch.setattr(get_settings(), "llm_provider", "anthropic")
    seen = {}

    def fake(facts):
        seen.update(facts)
        return "Mostly angry, worsening over time."

    monkeypatch.setattr(narrative, "_anthropic_narrative", fake)
    out = narrative.add_narrative(SUMMARY, ALERTS)
    assert out["narrative"] == "Mostly angry, worsening over time."
    assert out["narrative_source"] == get_settings().llm_model
    assert "headline" not in seen and "texts" not in seen
    assert seen["alerts"] == [{"rule": "sustained_anger", "severity": "critical", "start_s": 12}]
