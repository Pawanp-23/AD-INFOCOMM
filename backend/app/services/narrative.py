"""Optional LLM-written session narrative (FR-21, ADR-0008).

Disabled unless LLM_PROVIDER=anthropic. Only aggregate statistics leave the server — never
transcripts, audio or video — and any failure falls back to the deterministic summary.
"""

import json
import logging
from typing import Any

from app.core.config import get_settings

log = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You write short, neutral session notes for an emotion-analytics dashboard. "
    "You receive aggregate statistics produced by a speech/face emotion model. "
    "Write 2-3 plain sentences for a non-technical reader: the overall emotional picture, "
    "anything that changed over the session, and how trustworthy the signals were. "
    "Do not diagnose, give medical or psychological advice, or speculate about causes. "
    "Use only the numbers provided."
)


def _facts(summary: dict[str, Any], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """The only data sent to the provider: aggregates, no content."""
    return {
        "distribution_share_of_speech_time": summary.get("distribution", {}),
        "dominant_emotion": summary.get("dominant"),
        "mood_trend": summary.get("trend"),
        "mean_valence_minus1_to_1": summary.get("mean_valence"),
        "mean_modality_reliability_0_to_1": summary.get("mean_reliability", {}),
        "alerts": [
            {"rule": a["rule"], "severity": a["severity"], "start_s": a["start_ms"] // 1000}
            for a in alerts
        ],
    }


def _anthropic_narrative(facts: dict[str, Any]) -> str | None:
    import anthropic

    settings = get_settings()
    client = anthropic.Anthropic(
        api_key=settings.llm_api_key or None,  # None -> SDK resolves env/profile credentials
        timeout=settings.llm_timeout_s,
        max_retries=1,
    )
    response = client.beta.messages.create(
        model=settings.llm_model,
        max_tokens=1024,  # deliberately short: a 2-3 sentence note
        system=SYSTEM_PROMPT,
        output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        messages=[{"role": "user", "content": json.dumps(facts)}],
    )
    if response.stop_reason == "refusal":
        return None
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    return text or None


def add_narrative(summary: dict[str, Any], alerts: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the summary with `narrative` / `narrative_source` set (never raises)."""
    settings = get_settings()
    narrative, source = None, "rules"
    if settings.llm_provider == "anthropic" and summary.get("distribution"):
        try:
            narrative = _anthropic_narrative(_facts(summary, alerts))
            source = settings.llm_model if narrative else "rules"
        except Exception:  # noqa: BLE001 - the narrative is optional; never fail an analysis
            log.warning("LLM narrative failed; using rule-based summary", exc_info=True)
    return {
        **summary,
        "narrative": narrative or summary.get("headline", ""),
        "narrative_source": source,
    }
