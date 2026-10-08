from pathlib import Path

from omegaconf import OmegaConf

from ramei_ml.data.manifest import assert_speaker_independent, build_manifest
from ramei_ml.training.cv import fold_manifest, run_cv, speaker_folds


def test_folds_are_gender_balanced_and_cover_every_speaker(synthetic_ravdess: Path) -> None:
    df = build_manifest({"ravdess": synthetic_ravdess})
    k = 3
    fold_of = speaker_folds(df, k)
    assert set(fold_of) == set(df["speaker_id"])
    speakers = df[["speaker_id", "gender"]].drop_duplicates().set_index("speaker_id")["gender"]
    for fold in range(k):
        members = [s for s, f in fold_of.items() if f == fold]
        assert sorted(speakers[members].value_counts().tolist()) == [2, 2]  # 2 male + 2 female

    tested = []
    for fold in range(k):
        fm = fold_manifest(df, fold_of, fold, k)
        assert_speaker_independent(fm)
        assert set(fm["split"]) == {"train", "val", "test"}
        tested += list(fm.loc[fm["split"] == "test", "speaker_id"].unique())
    assert sorted(tested) == sorted(set(df["speaker_id"]))  # each speaker tested exactly once


def test_run_cv_end_to_end(synthetic_ravdess: Path, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.csv"
    build_manifest({"ravdess": synthetic_ravdess}).to_csv(manifest_path, index=False)
    cfg = OmegaConf.create(
        {
            "seed": 0,
            "n_jobs": 1,
            "cv_folds": 3,
            "cache_dir": str(tmp_path / "cache"),
            "output_root": str(tmp_path / "out"),
            "data": {
                "name": "synthetic",
                "manifest": str(manifest_path),
                "split_strategy": "speaker",
            },
            "experiment": {
                "name": "e0_test",
                "model": "mlp",
                "features": "stats",
                "mlp": {
                    "hidden_layer_sizes": [32],
                    "alpha": 0.01,
                    "batch_size": 32,
                    "learning_rate": "adaptive",
                    "max_iter": 300,
                },
            },
        }
    )
    result = run_cv(cfg)
    assert len(result["folds"]) == 3
    assert 0.0 <= result["summary"]["macro_f1"]["mean"] <= 1.0
    assert (tmp_path / "out" / "leaderboard.csv").exists()
