"""Build versioned, speaker-independent split manifests (TRD §3.3).

Manifest CSV columns: path, dataset, speaker_id, gender, emotion, raw_label, media, split

CLI:
    python -m ramei_ml.data.manifest --dataset ravdess=data/raw/ravdess --out ravdess_audio.csv
"""

import argparse
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ramei_ml.data.parsers import scan_dataset
from ramei_ml.paths import MANIFEST_DIR, ML_ROOT, resolve

SPLITS = ("train", "val", "test")
MANIFEST_COLUMNS = [
    "path",
    "dataset",
    "speaker_id",
    "gender",
    "emotion",
    "raw_label",
    "media",
    "split",
]


def _portable(path: str) -> str:
    """Store paths relative to ml/ when possible so tracked manifests work on every machine."""
    p = Path(path).resolve()
    try:
        return p.relative_to(ML_ROOT).as_posix()
    except ValueError:
        return p.as_posix()


def collect_records(datasets: Mapping[str, Path], media: str | None = "audio") -> pd.DataFrame:
    rows = [r.to_dict() for name, root in datasets.items() for r in scan_dataset(name, root, media)]
    if not rows:
        raise FileNotFoundError(f"No usable files found in {dict(datasets)}")
    df = pd.DataFrame(rows)
    df["path"] = df["path"].map(_portable)
    return df


def _allocate(
    items: list[str], ratios: tuple[float, float, float], rng: np.random.Generator
) -> dict[str, str]:
    """Assign items to train/val/test by ratio; every split gets ≥1 item when there are ≥3."""
    items = list(items)
    rng.shuffle(items)
    n = len(items)
    n_val = max(1, round(n * ratios[1])) if n >= 3 else 0
    n_test = max(1, round(n * ratios[2])) if n >= 3 else 0
    n_train = n - n_val - n_test
    assignment = {}
    for i, item in enumerate(items):
        assignment[item] = "train" if i < n_train else "val" if i < n_train + n_val else "test"
    return assignment


def speaker_independent_split(
    df: pd.DataFrame, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> pd.Series:
    """Split by speaker, stratified by (dataset, gender), so no speaker spans two splits.

    A stratum with fewer than 3 speakers (e.g. TESS has 2) cannot fill every split, so all of
    its speakers go to train; this is visible in ``split_summary``.
    """
    rng = np.random.default_rng(seed)
    speaker_split: dict[str, str] = {}
    strata = df[["dataset", "speaker_id", "gender"]].drop_duplicates()
    strata = strata.assign(gender=strata["gender"].fillna("unknown"))
    for _, group in strata.groupby(["dataset", "gender"], sort=True):
        speaker_split.update(_allocate(sorted(group["speaker_id"]), ratios, rng))
    return df["speaker_id"].map(speaker_split)


def random_split(
    df: pd.DataFrame, ratios: tuple[float, float, float] = (0.7, 0.15, 0.15), seed: int = 42
) -> pd.Series:
    """Utterance-level random split (speaker-DEPENDENT).

    Only for demonstrating leakage; never use it for reported results.
    """
    rng = np.random.default_rng(seed)
    draws = rng.random(len(df))
    cuts = np.cumsum(ratios)
    labels = np.where(draws < cuts[0], "train", np.where(draws < cuts[1], "val", "test"))
    return pd.Series(labels, index=df.index)


def assert_speaker_independent(df: pd.DataFrame) -> None:
    splits_per_speaker = df.groupby("speaker_id")["split"].nunique()
    leaking = splits_per_speaker[splits_per_speaker > 1]
    if not leaking.empty:
        raise AssertionError(f"Speakers appear in multiple splits: {list(leaking.index)[:10]}")


def build_manifest(
    datasets: Mapping[str, Path],
    strategy: str = "speaker",
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
    media: str | None = "audio",
) -> pd.DataFrame:
    df = collect_records(datasets, media)
    if strategy == "speaker":
        df["split"] = speaker_independent_split(df, ratios, seed)
        assert_speaker_independent(df)
    elif strategy == "random":
        df["split"] = random_split(df, ratios, seed)
    else:
        raise ValueError(f"Unknown split strategy '{strategy}' (expected 'speaker' or 'random')")
    return (
        df[MANIFEST_COLUMNS]
        .sort_values(["split", "dataset", "speaker_id", "path"])
        .reset_index(drop=True)
    )


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Utterance counts per split × emotion, plus speaker counts per split."""
    counts = pd.crosstab(df["split"], df["emotion"])
    counts["speakers"] = df.groupby("split")["speaker_id"].nunique()
    counts["total"] = df.groupby("split").size()
    return counts.reindex([s for s in SPLITS if s in counts.index])


def _parse_dataset_arg(value: str) -> tuple[str, Path]:
    name, _, root = value.partition("=")
    if not root:
        raise argparse.ArgumentTypeError("expected NAME=PATH, e.g. ravdess=data/raw/ravdess")
    return name, resolve(root)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset",
        action="append",
        type=_parse_dataset_arg,
        required=True,
        help="NAME=PATH (repeatable). Names: ravdess, crema_d, tess, savee",
    )
    parser.add_argument("--out", required=True, help="Output CSV name (written to data/manifests/)")
    parser.add_argument("--strategy", choices=["speaker", "random"], default="speaker")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--media", choices=["audio", "video", "all"], default="audio")
    args = parser.parse_args()

    df = build_manifest(
        dict(args.dataset),
        args.strategy,
        seed=args.seed,
        media=None if args.media == "all" else args.media,
    )
    out = MANIFEST_DIR / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(split_summary(df).to_string())
    print(f"\nWrote {len(df)} rows -> {out}")


if __name__ == "__main__":
    main()
