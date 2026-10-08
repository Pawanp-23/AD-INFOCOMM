# ADR-0001: Reliability-gated cross-modal transformer for fusion

| Field | Value |
|---|---|
| Status | Accepted |
| Date | 2026-10-09 |
| Deciders | Project team, internship guide |

## Context

RAMEI must combine audio, text and visual emotion evidence (PRD FR-10, FR-11) such that:
- noisy or missing modalities do not corrupt the decision,
- the system works with any subset of modalities,
- the contribution of each modality can be shown to the user (FR-17).

## Options considered

| Option | Adapts per input | Handles missing | Explainable weights | Cost |
|---|---|---|---|---|
| Feature concatenation + MLP | No | Poorly (zero-filling) | No | Low |
| Static weighted late fusion | No | Yes (renormalize) | Yes (fixed) | Low |
| Tensor fusion (outer product) | Partially | Poorly | No | High |
| **Reliability-gated cross-modal transformer** | **Yes** | **Yes (masking)** | **Yes (attention mass)** | Medium |

## Decision

Use the Reliability-Gated Cross-Modal Transformer (RG-CMT, TRD §4.5 F3): modality tokens with
an additive `β·log(r_k + ε)` attention bias from the Reliability Estimation Module and a hard mask
for unavailable modalities. Keep reliability-weighted late fusion (F1) as the baseline and fallback.

## Consequences

- Positive: input-adaptive trust, exact missing-modality handling, per-segment weights for explanations.
- Negative: requires a separately trained reliability module and corruption-augmented training data.
- Follow-up: ablations E9–E11 (TRD §6.2) must show each component's contribution.
