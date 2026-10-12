"""Fusion model contracts (TRD §4.4-4.5): shapes, missing modalities, reliability gating."""

import numpy as np
import pytest
import torch

from ramei_ml.data.corruptions import add_noise, blur_faces, darken_faces, drop_words, occlude_faces
from ramei_ml.features.quality import audio_quality
from ramei_ml.fusion.model import RAMEIFusion, fusion_loss, uncertainty_features


def make_batch(b: int = 4, avail: list[list[int]] | None = None) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    return {
        "audio": torch.randn(b, 13, 1536),
        "q_audio": torch.randn(b, 5),
        "text_z": torch.randn(b, 768),
        "text_logits": torch.randn(b, 7),
        "q_text": torch.randn(b, 5),
        "visual": torch.randn(b, 16, 768),
        "visual_mask": torch.ones(b, 16, dtype=torch.bool),
        "q_visual": torch.randn(b, 6),
        "avail": torch.tensor(avail or [[1, 1, 1]] * b, dtype=torch.float32),
    }


def test_forward_shapes_and_weights_sum_to_one() -> None:
    model = RAMEIFusion().eval()
    out = model(make_batch())
    assert out["logits"].shape == (4, 7)
    assert out["branch_logits"].shape == (4, 3, 7)
    assert out["reliability"].shape == out["weights"].shape == (4, 3)
    torch.testing.assert_close(out["weights"].sum(-1), torch.ones(4))
    assert ((out["reliability"] >= 0) & (out["reliability"] <= 1)).all()


def test_missing_modality_gets_zero_weight_and_reliability() -> None:
    model = RAMEIFusion().eval()
    out = model(make_batch(avail=[[1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, 0]]))
    w, r = out["weights"], out["reliability"]
    assert w[0, 2] == 0 and w[1, 1] == 0 and w[2, 0] == 0
    assert w[3, 1] == 0 and w[3, 2] == 0 and torch.isclose(w[3, 0], torch.tensor(1.0))
    assert r[0, 2] == 0 and r[3, 1] == 0
    assert torch.isfinite(out["logits"]).all()


def test_low_reliability_lowers_fusion_weight() -> None:
    model = RAMEIFusion().eval()
    with torch.no_grad():  # force REM outputs: audio very unreliable, others reliable
        for m, bias in (("audio", -8.0), ("text", 4.0), ("visual", 4.0)):
            model.rem[m].net[-1].weight.zero_()
            model.rem[m].net[-1].bias.fill_(bias)
        gated = model(make_batch())["weights"][:, 0].mean()
        model.c.use_reliability = False
        ungated = model(make_batch())["weights"][:, 0].mean()
    assert gated < ungated * 0.5


def test_loss_ignores_missing_branches() -> None:
    model = RAMEIFusion()
    batch = make_batch(avail=[[1, 0, 0]] * 4)
    losses = fusion_loss(model(batch), torch.tensor([0, 1, 2, 3]), batch["avail"], None)
    assert torch.isfinite(losses["total"])
    losses["total"].backward()
    assert model.rem["audio"].net[0].weight.grad.abs().sum() > 0
    assert model.rem["text"].net[0].weight.grad is None or (
        model.rem["text"].net[0].weight.grad.abs().sum() == 0
    )


def test_uncertainty_features_range() -> None:
    u = uncertainty_features(torch.tensor([[10.0, 0, 0, 0, 0, 0, 0], [0.0] * 7]))
    assert u[0, 0] < 0.05 and u[0, 1] > 0.99  # confident: low entropy, high max-prob
    assert u[1, 0] == pytest.approx(1.0, abs=1e-5)  # uniform: max entropy


def test_corruptions_degrade_signals() -> None:
    rng = np.random.default_rng(0)
    t = np.arange(16_000) / 16_000
    speech = (0.3 * np.sin(2 * np.pi * 220 * t) * (np.sin(2 * np.pi * 2 * t) > 0)).astype(
        np.float32
    )
    clean_snr = audio_quality(speech)[0]
    noisy_snr = audio_quality(add_noise(speech, 0.0, rng))[0]
    assert noisy_snr < clean_snr - 10

    crops = rng.integers(0, 255, (4, 32, 32, 3), dtype=np.uint8)
    assert blur_faces(crops).std() < crops.std()
    assert darken_faces(crops).mean() < crops.mean() * 0.3
    assert (occlude_faces(crops, rng) != crops).any()
    assert len(drop_words("a b c d e f g h i j", 0.5, rng).split()) < 10
